from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from thermal_engine import Settings, run_all, summarize, FAN_DB, AC_DB

ROOT=Path(__file__).parent
st.set_page_config(page_title="280Ah储能电池舱｜双预测MPC",page_icon="🔋",layout="wide")
st.markdown("""<style>
.stApp{background:#F3F7FC;color:#18324A}
[data-testid="stHeader"]{background:#F3F7FC}
section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid #D8E3EE}
section[data-testid="stSidebar"] *{color:#213B55 !important}
.block-container{max-width:1600px;padding-top:1rem}
h1,h2,h3{color:#173650 !important}
p,span,label,.stMarkdown{color:#263F57}
[data-testid="stMetric"]{background:white;border:1px solid #DCE6EF;border-radius:14px;padding:14px;box-shadow:0 4px 18px #1B46620A}
.hero{padding:25px 30px;border-radius:18px;background:linear-gradient(110deg,#173D5B,#167D83,#159B8D);margin-bottom:14px}
.hero h1{color:white!important;margin:0;font-size:2rem}.hero p{color:#E4F8F6;margin:8px 0 0}
.note{padding:11px 14px;border-left:4px solid #168B9A;background:#E8F3F8;border-radius:8px;color:#24445E}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔋 280Ah储能电池舱智能热管理平台</h1><p>固定送风 · 固定阈值反馈 · 电池温度/PCM状态双预测＋MPC｜风机、空调、风阀协同</p></div>',unsafe_allow_html=True)
st.markdown('<div class="note">三组共用同一电池—空气—PCM热模型、相同负荷与设备参数，仅控制策略不同。页面结果为降阶模型计算，不是实测或已完成CFD校准的结果。</div>',unsafe_allow_html=True)

with st.sidebar:
 st.header("运行工况")
 ambient=st.slider("环境温度（℃）",10,45,30,1)
 initial=st.slider("电池初始温度（℃）",15,55,30,1)
 rate=st.slider("放电倍率（C）",.2,2.,1.,.1)
 hours=st.slider("运行时长（h）",1,48,48,1)
 profile=st.selectbox("负荷时序",["完整充放电循环","高负荷间歇循环","单次放电后静置"])
 with st.expander("电池热参数（候选值，待辨识）"):
  resistance=st.number_input("等效内阻（mΩ）",.05,2.,.25,.05)
  mass=st.number_input("单体质量（kg）",1.,15.,5.49,.1)
  cp=st.number_input("单体比热（J/(kg·K)）",500,2000,1000,50)
 st.header("风机与风阀")
 fan_type=st.selectbox("舱级风机",list(FAN_DB))
 nfan=st.slider("风机数量",1,12,4)
 fixed_speed=st.slider("固定送风转速（%）",20,100,70,5)
 fixed_valve=st.slider("固定风阀开度（%）",25,100,75,5)
 st.header("空调配置")
 ac_type=st.selectbox("温控机组",list(AC_DB))
 ac_setpoint=st.slider("固定送风组空调回风设定（℃）",20.,32.,27.,.5)
 fixed_ac=st.slider("固定送风组压缩机负荷比例（%）",10,100,55,5)
 st.header("MPC设置")
 target=st.slider("电池最高温度目标（℃）",34.,43.,38.,.5)
 spread=st.slider("最大温差目标（℃）",2.,5.,3.,.5)
 pred=st.selectbox("预测时域（min）",[10,20,30],index=1)
 update=st.selectbox("滚动控制间隔（min）",[2,5],index=1)

geom=pd.read_csv(ROOT/"data"/"Stage15_v4_7_1_24个侧面PCM区域几何参数.csv")
pcm_volume=float(geom["Allocated_PCM_Volume_m3"].sum())
s=Settings(ambient=ambient,initial=initial,rate=rate,duration_h=hours,profile=profile,
 resistance_mohm=resistance,cell_mass=mass,cell_cp=cp,fans=nfan,fan_type=fan_type,
 fixed_speed=fixed_speed/100.,fixed_valve=fixed_valve/100.,target_t=target,target_spread=spread,
 prediction_minutes=pred,update_minutes=update,pcm_volume=pcm_volume,ac_type=ac_type,
 ac_setpoint=ac_setpoint,fixed_ac_fraction=fixed_ac/100.)
@st.cache_data(show_spinner=False,max_entries=24)
def calculate(settings):
 return run_all(settings,dt=60)
with st.spinner("正在计算0–48小时三组温度、控制动作和能耗轨迹…"):
 results=calculate(s)
labels=list(results)
colors={"固定送风":"#8797A8","固定阈值":"#367CB8","双预测MPC":"#07998C"}
metrics={k:summarize(v,s) for k,v in results.items()}
summary=pd.DataFrame(metrics)
base=metrics["固定阈值"]["系统总能耗（kWh）"]
mpc=metrics["双预测MPC"]["系统总能耗（kWh）"]
saving=(base-mpc)/max(base,1e-9)*100
m=metrics["双预测MPC"]
a,b,c,d=st.columns(4)
a.metric("MPC组最高温度",f'{m["电池最高温度（℃）"]:.2f}℃',f'较阈值组低 {metrics["固定阈值"]["电池最高温度（℃）"]-m["电池最高温度（℃）"]:.2f}℃')
b.metric("MPC组最大温差",f'{m["最大温差（℃）"]:.2f}℃',f'较阈值组低 {metrics["固定阈值"]["最大温差（℃）"]-m["最大温差（℃）"]:.2f}℃')
c.metric("MPC组系统总能耗",f'{mpc:.2f} kWh',f'较阈值组 {saving:+.1f}%',delta_color="inverse")
d.metric("MPC组超温累计时间",f'{m["超温累计时间（h）"]:.2f} h',f'阈值组 {metrics["固定阈值"]["超温累计时间（h）"]:.2f} h')
if m["最大温差（℃）"]>spread or m["电池最高温度（℃）"]>target:
 st.warning("当前MPC组仍未达到全部温控目标；请检查热负荷、设备容量及控制参数。结果按模型原值显示。")

def chart(col,title,ytitle,targetline=None):
 fig=go.Figure()
 for name,df in results.items():
  fig.add_trace(go.Scatter(x=df.time_h,y=df[col],name=name,mode="lines",line=dict(color=colors[name],width=3 if name=="双预测MPC" else 2)))
 if targetline is not None:fig.add_hline(y=targetline,line_dash="dash",line_color="#D58B36",annotation_text="控制目标")
 fig.update_layout(title=title,xaxis_title="运行时间（h）",yaxis_title=ytitle,height=360,template="plotly_white",
  font=dict(color="#203A52"),paper_bgcolor="white",plot_bgcolor="white",hovermode="x unified",
  legend=dict(orientation="h",y=-.24),margin=dict(l=18,r=18,t=55,b=35),
  xaxis=dict(gridcolor="#E8EEF4"),yaxis=dict(gridcolor="#E8EEF4"))
 return fig

tabs=st.tabs(["01 综合评价","02 全时段温度表现","03 双预测与控制动作","04 分项能耗与设备","05 PCM状态（内部状态）"])
with tabs[0]:
 st.subheader("三组48小时（或所选时长）性能对比")
 st.dataframe(summary,use_container_width=True)
 x,y=st.columns(2)
 x.plotly_chart(chart("Tmax_C","电池最高温度全过程","℃",target),use_container_width=True)
 y.plotly_chart(chart("dTmax_C","电池热区最大温差全过程","℃",spread),use_container_width=True)
 x,y=st.columns(2)
 x.plotly_chart(chart("total_energy_kWh","系统累计总能耗","kWh"),use_container_width=True)
 y.plotly_chart(chart("Tmean_C","电池平均温度全过程","℃"),use_container_width=True)
 st.caption("超温累计时间按Tmax超过侧栏设定目标累计；温差超限时间另行统计。")
with tabs[1]:
 st.subheader("0–48小时完整温度轨迹")
 st.plotly_chart(chart("Tmax_C","最高电池温度","℃",target),use_container_width=True)
 st.plotly_chart(chart("dTmax_C","最大温差","℃",spread),use_container_width=True)
 st.plotly_chart(chart("Tmean_C","平均电池温度","℃"),use_container_width=True)
 st.plotly_chart(chart("Tair_mean_C","舱内等效空气温度","℃"),use_container_width=True)
with tabs[2]:
 st.subheader("双状态预测参与控制的动作轨迹")
 st.write(f"状态更新60秒；MPC每{update}分钟滚动决策；预测时域{pred}分钟。")
 st.plotly_chart(chart("fan_speed","风机转速指令","额定转速比例"),use_container_width=True)
 st.plotly_chart(chart("ac_fraction","空调压缩机负荷指令","负荷比例"),use_container_width=True)
 st.plotly_chart(chart("valve_pct","风阀开度","%"),use_container_width=True)
 st.plotly_chart(chart("PCM_T_C","PCM平均温度（MPC内部状态）","℃"),use_container_width=True)
 st.plotly_chart(chart("PCM_liquid","PCM等效液相率（MPC内部状态）","0–1"),use_container_width=True)
 st.markdown("**控制逻辑：** 电池温度预测与PCM焓状态共同进入短时域滚动评价；MPC在候选风机转速、空调负荷、风阀开度和区域配风策略中选择综合代价较低的动作。PCM指标用于控制器状态，不列入最终性能评价表。")
with tabs[3]:
 st.subheader("风机、空调与总能耗")
 st.plotly_chart(chart("fan_energy_kWh","风机累计能耗","kWh"),use_container_width=True)
 st.plotly_chart(chart("ac_energy_kWh","空调累计能耗","kWh"),use_container_width=True)
 st.plotly_chart(chart("valve_energy_kWh","风阀及辅助设备累计能耗","kWh"),use_container_width=True)
 st.plotly_chart(chart("total_energy_kWh","系统总能耗（核心评价指标）","kWh"),use_container_width=True)
 st.plotly_chart(chart("fan_power_W","风机即时功率","W"),use_container_width=True)
 st.plotly_chart(chart("ac_power_W","空调即时输入功率","W"),use_container_width=True)
 st.plotly_chart(chart("fan_runtime_h","风机累计运行时间","h"),use_container_width=True)
 st.plotly_chart(chart("ac_runtime_h","空调累计运行时间","h"),use_container_width=True)
 st.markdown("**设备额定参数**")
 st.dataframe(pd.DataFrame([{"风机型号":k,"额定风量(m³/h)":v["flow"],"额定功率(W)":v["power"],"最大静压(Pa)":v["pressure"]} for k,v in FAN_DB.items()]),use_container_width=True)
 st.dataframe(pd.DataFrame([{"空调型号":k,"额定制冷量(kW)":v["cooling_kw"],"额定输入功率(kW)":v["rated_input_kw"]} for k,v in AC_DB.items()]),use_container_width=True)
with tabs[4]:
 st.subheader("PCM储热状态预测（控制器内部状态，不作为最终评价指标）")
 st.plotly_chart(chart("PCM_T_C","PCM平均温度","℃"),use_container_width=True)
 st.plotly_chart(chart("PCM_liquid","PCM等效液相率","0–1"),use_container_width=True)
 st.caption("PCM末端液相率和剩余储热能力不列入三组最终评价表，但PCM焓状态持续参与MPC前瞻决策。材料参数仍需与用户指定WT-CPCM文献及实物材料最终核对。")

st.divider()
st.subheader("导出数据与复现")
export=pd.concat([df.assign(strategy=name) for name,df in results.items()],ignore_index=True)
st.download_button("下载三组0–48小时逐分钟原始结果CSV",export.to_csv(index=False).encode("utf-8-sig"),
 file_name="280Ah_三策略_风机空调MPC_全过程结果.csv",mime="text/csv",type="primary")
st.caption("v8.0｜风机＋空调＋风阀｜三策略同工况｜12热区＋24 PCM焓节点｜降阶模型演示，待真实280Ah参数及CFD/实验校准。")
