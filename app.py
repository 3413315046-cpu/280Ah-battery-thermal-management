"""Competition research website v7.0 — strictly no air-conditioning.
Install: pip install -r requirements.txt; streamlit run app.py
"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from thermal_engine import Settings, run_all, summarize, FAN_DB, h_specific, liquid_ratio, PCM_DENSITY, PCM_LATENT, PHASE_LO, PHASE_HI

ROOT=Path(__file__).parent
st.set_page_config(page_title='280Ah｜双状态预测与智能风冷协同',page_icon='🔋',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp{background:#F3F7FC;color:#17283D}
[data-testid="stHeader"]{background:#F3F7FC}
section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid #D6E2ED}
section[data-testid="stSidebar"] *{color:#233952 !important}
.block-container{max-width:1600px;padding-top:1.15rem;padding-bottom:3rem}
.stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption, p, span, label {color:#24364C}
h1,h2,h3{color:#132940 !important}
[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #D9E4EF;border-radius:16px;padding:17px 16px;box-shadow:0 4px 20px #183A5E0D}
[data-testid="stMetricLabel"] *, [data-testid="stMetricLabel"]{color:#526881 !important}
[data-testid="stMetricValue"] *, [data-testid="stMetricValue"]{color:#102A43 !important}
[data-testid="stMetricDelta"] *{font-weight:700}
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid #D5E1EB;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{background:#FFFFFF;color:#234158;border:1px solid #D6E2ED;border-radius:10px;padding:9px 15px;font-weight:650}
.stTabs [aria-selected="true"]{background:#E0F5F2;color:#087F78;border-color:#80CBC2}
[data-testid="stDataFrame"]{background:#FFFFFF;border:1px solid #DCE6F0;border-radius:12px}
.hero{padding:26px 32px;border-radius:19px;background:linear-gradient(113deg,#143D5C 0%,#146A79 60%,#149989 100%);box-shadow:0 11px 30px #1A53652A;margin-bottom:16px}
.hero h1{color:#FFFFFF !important;font-size:2.08rem;margin:0}.hero p{color:#E1F9F7;margin:8px 0 0}
.hint{padding:11px 14px;background:#E6F2FA;border-left:4px solid #2B85C6;border-radius:8px;color:#23466A}
.stButton>button[kind="primary"]{background:#0A8D86;color:white;border:none}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔋 280Ah储能电池舱 · 双状态预测与风冷智能协同</h1><p>三策略公平对照｜固定送风 · 固定阈值 · 双预测MPC｜电池热状态与PCM储热状态联动</p></div>',unsafe_allow_html=True)
st.markdown('<div class="hint">当前为12热区＋24个PCM焓节点的<strong>工程演示模型</strong>，144个电芯坐标用于空间展示。没有空调；温控和能耗差值由同一热模型计算，不预置优势。<strong>3℃是目标，不是保证值。</strong></div>',unsafe_allow_html=True)

with st.sidebar:
 st.header('⚙️ 运行工况')
 st.caption('相同环境、热负荷、PCM物性和设备参数用于全部三组。')
 ambient=st.slider('环境温度（℃）',10,45,30,1)
 initial=st.slider('电池初始温度（℃）',15,55,30,1)
 rate=st.slider('放电倍率（C）',.2,2.,1.,.1)
 hours=st.slider('仿真运行时长（h）',1,48,12,1)
 profile=st.selectbox('电池负荷时序',['完整充放电循环','高负荷间歇循环','单次放电后静置'])
 st.caption('循环负荷：按容量限制放电时长，配套0.5倍电流充电，再安排静置；连续48小时并非连续48小时1C放电。')
 with st.expander('电芯候选热参数（尚待标定）'):
  resistance=st.number_input('等效内阻（mΩ）',.05,2.,.25,.05)
  mass=st.number_input('电芯质量（kg）',1.,15.,5.49,.1)
  cp=st.number_input('电芯比热（J/(kg·K)）',500,2000,1000,50)
 st.header('🌀 仅风冷设备配置')
 fan_type=st.selectbox('舱级风机类型',list(FAN_DB))
 nfan=st.slider('舱级风机数量',1,12,4)
 fixed=st.slider('固定送风组风机转速（%）',20,100,70,5)
 valve=st.slider('基准风阀开度（%）',25,100,75,5)
 target=st.slider('最高温度目标（℃）',34.,43.,38.,.5)
 spread=st.slider('最大温差控制目标（℃）',2.,5.,3.,.5)
 pred_h=st.selectbox('MPC预测时域（分钟）',[10,20,30],index=1)
 ctrl_step=st.selectbox('滚动更新间隔（分钟）',[2,5],index=1)
 st.caption('固定送风组按设置转速从开始持续送风；另外两组根据当前温度或未来预测按需运行。')

# user-provided historical fixed PCM parameters (not WT-CPCM 502.2kJ/kg paper model)
geom=pd.read_csv(ROOT/'data'/'Stage15_v4_7_1_24个侧面PCM区域几何参数.csv')
pcm_volume=float(geom['Allocated_PCM_Volume_m3'].sum())
s=Settings(ambient=float(ambient),initial=float(initial),rate=float(rate),duration_h=hours,profile=profile,
 resistance_mohm=float(resistance),cell_mass=float(mass),cell_cp=float(cp), fans=nfan,fan_type=fan_type,
 fixed_speed=fixed/100.,fixed_valve=valve/100.,target_t=target,target_spread=spread,
 prediction_minutes=pred_h,update_minutes=ctrl_step,pcm_volume=pcm_volume)
@st.cache_data(show_spinner=False,max_entries=30)
def calc(settings): return run_all(settings)
with st.spinner('正在计算三组同工况热响应与控制轨迹…'):
 results=calc(s)
labels=list(results)
colors={'固定送风':'#77899C','固定阈值':'#357CBD','双预测MPC':'#009A87'}
metrics={label:summarize(results[label],s) for label in labels}
summary=pd.DataFrame(metrics)
M=summary.loc
save_e=100*(metrics['固定送风']['总设备能耗（kWh）']-metrics['双预测MPC']['总设备能耗（kWh）'])/max(metrics['固定送风']['总设备能耗（kWh）'],1e-8)
mt=metrics['双预测MPC']['最高温度 Tmax（℃）']
ms=metrics['双预测MPC']['最大温差 ΔTmax（℃）']
c1,c2,c3,c4=st.columns(4)
c1.metric('预测组最高温度',f'{mt:.2f}℃',f'{metrics["固定送风"]["最高温度 Tmax（℃）"]-mt:+.2f}℃ 相对固定送风')
c2.metric('预测组最大温差',f'{ms:.2f}℃',f'{metrics["固定送风"]["最大温差 ΔTmax（℃）"]-ms:+.2f}℃ 相对固定送风')
c3.metric('预测组累计能耗',f'{metrics["双预测MPC"]["总设备能耗（kWh）"]:.3f} kWh',f'相对固定送风节能 {save_e:+.1f}%',delta_color='off')
c4.metric('预测组风机运行',f'{metrics["双预测MPC"]["风机运行时长（h）"]:.2f} h',f'固定送风 {metrics["固定送风"]["风机运行时长（h）"]:.2f}h',delta_color='off')
if mt>target or ms>spread:
 st.info(f'本工况预测组计算最高温 {mt:.2f}℃、温差 {ms:.2f}℃，存在未达成控制目标的指标；请结合散热能力与设备功率查看，不对计算结果进行人为修饰。')


def lineplot(columns,title,ytitle,style=None,baseline=None,unit='',height=350):
 fig=go.Figure()
 for label,df in results.items():
  fig.add_trace(go.Scatter(x=df['time_h'],y=df[columns],mode='lines',name=label,
   line=dict(color=colors[label],width=3 if label=='双预测MPC' else 2.3)))
 if baseline is not None:fig.add_hline(y=baseline,line_dash='dash',line_color='#D78830',annotation_text='目标',annotation_position='top left')
 fig.update_layout(title=title,height=height,template='plotly_white',font=dict(color='#203A52',size=12),
  xaxis_title='运行时间（h）',yaxis_title=ytitle,margin=dict(l=20,r=20,b=20,t=55),
  legend=dict(orientation='h',y=-.24),paper_bgcolor='#FFFFFF',plot_bgcolor='#FFFFFF',
  hovermode='x unified',xaxis=dict(gridcolor='#E8EFF5'),yaxis=dict(gridcolor='#E8EFF5'))
 return fig

tabs=st.tabs(['01 综合评价','02 温度与温差','03 PCM储热','04 预测与调控','05 能耗与设备','06 三维结构'])
with tabs[0]:
 st.subheader('三组同工况综合评价')
 descriptions={
 '固定送风':'不预测、不反馈；风机从运行开始按固定转速持续送风，风阀固定。',
 '固定阈值':'无预测；按当前最高温和带回差的固定规则启停、切换风机档位。',
 '双预测MPC':'结合电池温度和PCM储热状态，前瞻计算并枚举风机档位、风阀开度与热区风量分配方案。'}
 for i,(label,desc) in enumerate(descriptions.items()):
  st.markdown(f'**<span style="color:{colors[label]}">● {i+1}. {label}</span>**　{desc}',unsafe_allow_html=True)
 st.dataframe(summary,use_container_width=True)
 col1,col2=st.columns(2)
 with col1:
  fig=go.Figure()
  for label in labels:
   fig.add_trace(go.Bar(x=[label],y=[metrics[label]['最高温度 Tmax（℃）']],name=label,marker_color=colors[label]))
  fig.update_layout(title='最高温度对比',template='plotly_white',showlegend=False,yaxis_title='℃',height=355,margin=dict(l=15,r=15,b=20,t=60))
  st.plotly_chart(fig,use_container_width=True)
 with col2:
  fig=go.Figure()
  for label in labels:
   fig.add_trace(go.Bar(x=[label],y=[metrics[label]['总设备能耗（kWh）']],name=label,marker_color=colors[label]))
  fig.update_layout(title='系统能耗对比（无空调）',template='plotly_white',showlegend=False,yaxis_title='kWh',height=355,margin=dict(l=15,r=15,b=20,t=60))
  st.plotly_chart(fig,use_container_width=True)
 st.plotly_chart(lineplot('Tmax_C','最高温度随时间变化','Tmax（℃）',baseline=target),use_container_width=True)
 st.caption('改善幅度会随工况变化，控制器没有内置实验组优于对照组的数值偏置。指标来自热网络的数值模拟，而非完整实测。')
with tabs[1]:
 st.subheader('三策略温度与温差变化')
 st.plotly_chart(lineplot('Tmax_C','电池热区最高温度','温度（℃）',baseline=target),use_container_width=True)
 st.plotly_chart(lineplot('dTmax_C','热区最大温差','ΔT（℃）',baseline=spread),use_container_width=True)
 st.plotly_chart(lineplot('Tmean_C','电池平均温度','平均温度（℃）'),use_container_width=True)
 st.caption('最大温差取12热区温度最大值减最小值；由于一个热区代表12个电芯，不能当作144芯逐芯真实最大温差。')
with tabs[2]:
 st.subheader('PCM相变储热与剩余容量')
 a,b,c,d,e=st.columns(5)
 for col,key,val in zip((a,b,c,d,e),('密度','显热比热','导热系数','相变温区','潜热'),('1100 kg/m³','1740 J/(kg·K)','3.5 W/(m·K)','38–43℃','120 kJ/kg')):
  col.metric(key,val)
 st.plotly_chart(lineplot('PCM_T_C','PCM平均温度','温度（℃）'),use_container_width=True)
 st.plotly_chart(lineplot('PCM_liquid','PCM体积分区等效液相率','液相率（0–1）'),use_container_width=True)
 st.plotly_chart(lineplot('PCM_remaining_MJ','PCM剩余潜热容量','MJ'),use_container_width=True)
 a,b=st.columns(2)
 a.metric('PCM几何区域体积',f'{pcm_volume:.5f} m³')
 b.metric('PCM估算质量',f'{pcm_volume*PCM_DENSITY:.1f} kg')
 st.caption('网页使用你提供的固定参数组：38–43℃、120 kJ/kg，与宽温域WT-CPCM论文的多阶段总吸热焓不是同一套已验证参数。液相率为24区域线性焓近似；储热余量显示剩余潜热，显热另计。')
with tabs[3]:
 st.subheader('预测窗口与控制动作对比')
 st.write(f'当前热状态更新：**60秒**；滚动决策：**每{ctrl_step}分钟**；MPC前瞻：**{pred_h}分钟**。')
 st.plotly_chart(lineplot('fan_speed','风机转速比例随时间变化','额定转速比例（0–1）'),use_container_width=True)
 st.plotly_chart(lineplot('valve_pct','风阀开度控制轨迹','开度（%）'),use_container_width=True)
 st.plotly_chart(lineplot('current_A','充放电电流时序','电流（A）'),use_container_width=True)
 st.markdown('**预测调控决策结构**')
 st.markdown('**负荷预测＋热状态** → 未来电池最高温/温差与PCM储热余量 → 枚举风机转速、风阀开度和区域流量分配 → 以温度约束、温差约束、能耗和调速代价评分 → 执行最优候选 → 下一时刻重算。')
 st.caption('这里的MPC是离散候选控制量的短时域滚动搜索；区域均衡风量是模型中的分配策略，现场实施仍需独立区域风阀。')
with tabs[4]:
 st.subheader('风机运行与累计能耗')
 st.plotly_chart(lineplot('fan_power_W','舱级风机即时功率','功率（W）'),use_container_width=True)
 st.plotly_chart(lineplot('energy_kWh','累计系统设备耗电量','能耗（kWh）'),use_container_width=True)
 st.plotly_chart(lineplot('fan_runtime_h','累计运行时间','h'),use_container_width=True)
 st.plotly_chart(lineplot('airflow_m3h','风量（安装工况估算）','m³/h'),use_container_width=True)
 st.markdown('**设备参考参数：**')
 st.dataframe(pd.DataFrame([{'风机':n,'额定风量(m³/h)':v['flow'],'额定功率(W)':v['power'],'最大静压(Pa)':v['pressure'],'额定转速(rpm)':v['rpm']} for n,v in FAN_DB.items()]),use_container_width=True,hide_index=True)
 st.caption('风机P∝转速³；总耗电包含风机功耗和按5W×15s/次近似的风阀执行器动作电耗；分区执行器数量为候选假设。舱体风路与P-Q曲线实际工作点尚未通过阻力曲线确定，风量为工程估算。')
with tabs[5]:
 st.subheader('电池柜CAD中心坐标与热区温度映射')
 dfcoord=pd.read_csv(ROOT/'data'/'Stage15_v4_7_1_144电芯整数CAD坐标映射.csv')
 selected=st.selectbox('展示哪种控制策略的末端温度',labels,index=2)
 view=dfcoord.copy()
 packs=sorted(view.Pack_ID.unique())
 # 6 packs × 2 parallel positions = 12 equivalent zones. Temperature shading is illustrative,
 # not the resolved 144-cell finite-volume temperature field.
 ix={(pack,branch):i*2+j for i,pack in enumerate(packs) for j,branch in enumerate(('A','B'))}
 lastT=results[selected].attrs['final_zone_T']
 view['温度（热区映射）']=view.apply(lambda r:float(lastT[ix[(r.Pack_ID,r.Parallel_Branch)]]),axis=1)
 fig=px.scatter_3d(view,x='CAD_X_center_mm_rounded',y='CAD_Y_center_mm_rounded',z='CAD_Z_center_mm_rounded',
  color='温度（热区映射）',color_continuous_scale='RdYlBu_r',hover_name='Cell_ID',hover_data=['Pack_ID','Parallel_Branch'])
 fig.update_traces(marker=dict(size=5,opacity=.9))
 fig.update_layout(template='plotly_white',height=600,scene=dict(aspectmode='data'),margin=dict(l=0,r=0,t=10,b=0))
 st.plotly_chart(fig,use_container_width=True)
 st.caption('每个热区的温度颜色复制到12个对应电芯中心点，仅用于检查几何与热区分组；绝非真实逐芯测温或144芯温度场。CAD映射A/B及SG方向仍有待核对。')

st.divider()
st.markdown('### 📤 导出全部三组原始曲线')
export=pd.concat([results[k].assign(方案=k) for k in labels],ignore_index=True)
st.download_button('下载三策略时序结果 CSV',data=export.to_csv(index=False).encode('utf-8-sig'),file_name='三策略_温度_PCM_风机能耗_演示数据.csv',mime='text/csv',type='primary')
st.caption('v7.0 · 三策略同模型对照 · 无空调 · 1–48小时 · 12热区/24PCM · 浅色高对比度 · 显示与导出一致数据 · 仅为工程计算演示')
