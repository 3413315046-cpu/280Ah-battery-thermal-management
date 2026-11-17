import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE=Path(__file__).parent
def loadj(n): return json.loads((BASE/"data"/n).read_text(encoding="utf-8"))
topo=loadj("target_baseline.json"); pcm=loadj("wt_cpcm_reference.json")
cell=loadj("lf280k_reference.json"); sources=loadj("data_sources.json")
coords=pd.read_csv(BASE/"data"/"Stage15_v4_7_1_144电芯整数CAD坐标映射.csv")
pcmgeo=pd.read_csv(BASE/"data"/"Stage15_v4_7_1_24个侧面PCM区域几何参数.csv")
FAN={
 "model":"ebm-papst W3G500-KK07-G2 (EC axial, 500 mm)",
 "flow_m3h":4110.0,"pressure_Pa":82.0,"power_W":230.0,
 "nominal_power_W":250.0,"rpm":970.0,"voltage_V":230.0,
 "url":"https://www.ebmpapst.com/us/en/products/axial-fans/p/W3G500KK07G2.html"
}
st.set_page_config(page_title="280Ah电池柜热管理数字孪生",page_icon="🔋",layout="wide")
st.markdown("""<style>
.block-container{max-width:1550px;padding-top:1.2rem;padding-bottom:2rem}
.hero{padding:24px 28px;border-radius:16px;background:linear-gradient(115deg,#10263b,#153a37 60%,#263a32);border:1px solid #31566b}
.hero h1{color:#f5f8fb;margin:0;font-size:2rem}.hero p{color:#b7c9d8;margin:7px 0 0}
[data-testid="stMetric"]{background:#142331;padding:13px 16px;border:1px solid #2d4354;border-radius:12px}
.note{color:#9fb1c2;font-size:.88rem}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔋 280Ah 磷酸铁锂电池柜智能热管理数字孪生平台</h1><p>电芯热源估算 · WT-CPCM储热 · 空冷风机调速 · 有风/无风对比</p></div>',unsafe_allow_html=True)
st.warning("工程估算版本 v3.0：当前运行的是可调参数的集中参数热模型，不是已验证的192状态真实柜体预测。所有结果均受候选电芯、等效热阻、换热效率及PCM未知物性影响。页面将明确标注估算假设，不作为安全认证或设备选型定案。")

# core input sidebar
st.sidebar.header("运行工况与参数")
ambient=st.sidebar.slider("环境温度（℃）", -10, 50, 30, 1)
t0=st.sidebar.slider("电池初始温度（℃）", 10, 60, 30, 1)
rate=st.sidebar.slider("放电倍率（C）",0.2,2.0,1.0,0.1)
hours=st.sidebar.slider("仿真时长（h）",1,12,6,1)
r_mohm=st.sidebar.number_input("单体等效内阻（mΩ，暂定）",0.05,2.0,0.25,0.05)
cell_mass=st.sidebar.number_input("单体质量（kg，候选参考）",1.0,15.0,5.49,0.1)
cp_cell=st.sidebar.number_input("单体比热（J/(kg·K)，暂定）",500,2000,1000,50)
num_fans=st.sidebar.slider("风机数量",1,8,4,1)
speed_pct=st.sidebar.slider("风机转速比例（%）",20,100,70,5)
bypass=st.sidebar.slider("有效风量利用率（%）",1,30,8,1)
pcm_density=st.sidebar.number_input("PCM密度（kg/m³，假设值）",400,1600,900,25)
cp_pcm=st.sidebar.number_input("PCM显热比热（J/(kg·K)，假设值）",500,3000,1500,50)
pcm_volume=float(pcmgeo["Allocated_PCM_Volume_m3"].sum())
pcm_mass=pcm_volume*pcm_density
pcm_contact=st.sidebar.slider("电芯至PCM等效耦合比例（%）",1,80,20,1)/100
st.sidebar.caption("带“假设/暂定”的参数可按后续实测或厂家资料替换。")

# fan affinity laws
n=speed_pct/100
q_one=FAN["flow_m3h"]*n
q_total=q_one*num_fans
p_total=FAN["power_W"]*num_fans*n**3
rho_air=1.2; cp_air=1005
mdot=rho_air*q_total/3600
# rectangular cabin frontal section; approximate mean bulk velocity
area=2.6*2.438
v_air=(q_total/3600)/area
# simplified heat source, Joule heat only; reversible/entropic heat excluded
I=280*rate
Q_cell=I**2*(r_mohm/1000)
Q_total=144*Q_cell
C_b=144*cell_mass*cp_cell
C_pcm=pcm_mass*cp_pcm
# PCM staged enthalpy curve. Sensible cp assumed; paper's enthalpies represented as heating-only apparent Cp over reported ranges.
ranges=pcm["stage_temperature_ranges_C"]; enthalpies=pcm["stage_enthalpy_J_kg"]
def pcm_h(T):
    T=np.asarray(T,dtype=float)
    h=cp_pcm*(T-ambient)
    for (lo,hi),latent in zip(ranges,enthalpies):
        h += latent*np.clip((T-lo)/(hi-lo),0,1)
    return h
# invert specific enthalpy with monotone table
Tgrid=np.linspace(-10,160,3401); Hgrid=pcm_h(Tgrid)
def pcm_temp(H):
    return np.interp(H,Hgrid,Tgrid)
# use natural convection conductance and fan-enhanced air conductance, bypass factor exposed
G_nat=28.0
G_fan=mdot*cp_air*(bypass/100)
G_air=G_nat+G_fan
# thermal conductance cell-to-PCM and PCM-to-air
G_bp=max(5.0,pcm_contact*pcmgeo["Cell_PCM_Direct_Contact_Area_m2"].sum()*120.0)
G_pa=max(2.0,mdot*cp_air*0.025 + 8.0)
dt=30.0; times=np.arange(0,hours*3600+dt,dt)
def simulate(fan_on):
    Tb=np.zeros(len(times)); Tp=np.zeros(len(times)); H=np.zeros(len(times)); stored=np.zeros(len(times))
    Tb[0]=t0; Tp[0]=ambient; H[0]=pcm_mass*float(pcm_h(ambient))
    for k in range(1,len(times)):
        ta=ambient; tb=Tb[k-1]; tp=Tp[k-1]
        qfan=(G_air*(tb-ta)) if fan_on else G_nat*(tb-ta)
        qbp=G_bp*(tb-tp)
        qpa=(G_pa*(tp-ta)) if fan_on else 8.0*(tp-ta)
        dTb=(Q_total-qfan-qbp)/max(C_b,1)
        dH=qbp-qpa
        Tb[k]=tb+dTb*dt
        H[k]=H[k-1]+dH*dt
        Tp[k]=float(pcm_temp(H[k]/max(pcm_mass,1e-6)))
        stored[k]=max(0.0,H[k]-H[0])
    return pd.DataFrame({"time_h":times/3600,"battery_C":Tb,"pcm_C":Tp,"pcm_stored_MJ":stored/1e6})
no=simulate(False); yes=simulate(True)
# Display tabs
tabs=st.tabs(["运行控制台","有风/无风温度对比","PCM储热状态","风机功率与风速","三维结构","模型依据与限制"])
with tabs[0]:
    st.subheader("当前工况")
    a,b,c,d=st.columns(4)
    a.metric("总焦耳热源",f"{Q_total/1000:.2f} kW")
    b.metric("电池等效热容",f"{C_b/1000:.1f} kJ/K")
    c.metric("PCM估算质量",f"{pcm_mass:.1f} kg")
    d.metric("风机总输入功率",f"{p_total:.0f} W")
    st.markdown("#### 当前控制设定")
    st.write(f"环境 {ambient}℃ · 初始电池 {t0}℃ · {rate:.1f}C · {hours}h · 风机 {num_fans}台、转速 {speed_pct}%")
    st.write(f"估算总风量 {q_total:,.0f} m³/h · 舱体截面平均风速约 {v_air:.2f} m/s · 估算PCM质量 {pcm_mass:.1f} kg")
    st.info("热源目前只计算 I²R 焦耳热，未计入可逆熵热、母排/连接件发热、柜体热容及空间温度不均匀。")
    st.subheader("模型输出摘要（集中参数估算）")
    a,b,c=st.columns(3)
    a.metric("无风机末端电池温度",f"{no.battery_C.iloc[-1]:.1f} ℃")
    b.metric("有风机末端电池温度",f"{yes.battery_C.iloc[-1]:.1f} ℃")
    c.metric("风机工况末端PCM温度",f"{yes.pcm_C.iloc[-1]:.1f} ℃")
    st.caption("这些是当前假设参数下的计算值，不是CFD或实验实测结果。")
with tabs[1]:
    st.subheader("有风机 / 无风机电池温度变化对比")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=no.time_h,y=no.battery_C,name="无风机（自然对流）",line=dict(width=3)))
    fig.add_trace(go.Scatter(x=yes.time_h,y=yes.battery_C,name=f"有风机（{num_fans}台，{speed_pct}%转速）",line=dict(width=3)))
    fig.add_hline(y=55,line_dash="dash",annotation_text="55℃文献阶段边界（参考）")
    fig.update_layout(height=470,xaxis_title="时间（h）",yaxis_title="电池平均温度（℃）",legend=dict(orientation="h"))
    st.plotly_chart(fig,use_container_width=True)
    delta=no.battery_C.iloc[-1]-yes.battery_C.iloc[-1]
    st.metric("末端平均温度差（无风机－有风机）",f"{delta:.1f} ℃")
    st.caption("模型假设：144芯同等发热、集中热容、统一环境温度；不解析柜内局部热点和风道分配。")
with tabs[2]:
    st.subheader("PCM储热状态")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=yes.time_h,y=yes.pcm_C,name="有风机：PCM温度"))
    fig.add_trace(go.Scatter(x=no.time_h,y=no.pcm_C,name="无风机：PCM温度"))
    fig.update_layout(height=400,xaxis_title="时间（h）",yaxis_title="PCM等效温度（℃）")
    st.plotly_chart(fig,use_container_width=True)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=yes.time_h,y=yes.pcm_stored_MJ,name="有风机：PCM净储热"))
    fig2.add_trace(go.Scatter(x=no.time_h,y=no.pcm_stored_MJ,name="无风机：PCM净储热"))
    fig2.update_layout(height=400,xaxis_title="时间（h）",yaxis_title="相对初始状态的净储热（MJ）")
    st.plotly_chart(fig2,use_container_width=True)
    st.metric("风机工况PCM净储热",f"{yes.pcm_stored_MJ.iloc[-1]:.2f} MJ")
    st.caption("采用论文三阶段吸热焓构造升温焓曲线；阶段II/III按温区线性分配仅为数值近似，不代表反应动力学或冷却可逆性已验证。")
with tabs[3]:
    st.subheader("风机选型参考与调速")
    st.markdown(f"**候选参考：[{FAN['model']}]({FAN['url']})**")
    st.write("厂家公开工作点：500 mm，4110 m³/h，82 Pa，约230 W，970 rpm；额定最大输入功率约250 W。")
    st.dataframe(pd.DataFrame([
      ["单台额定参考风量",f"{FAN['flow_m3h']:,.0f} m³/h"],
      ["单台参考静压",f"{FAN['pressure_Pa']:.0f} Pa"],
      ["单台参考输入功率",f"{FAN['power_W']:.0f} W"],
      ["数量",f"{num_fans} 台"],
      ["当前转速比例",f"{speed_pct}%"],
      ["当前总风量（相似定律估算）",f"{q_total:,.0f} m³/h"],
      ["当前总功率（立方定律估算）",f"{p_total:,.0f} W"],
      ["舱体截面平均风速",f"{v_air:.2f} m/s"],
    ],columns=["指标","估算值"]),use_container_width=True,hide_index=True)
    st.info("风机相似定律：风量约与转速成正比、压力约与转速平方成正比、功率约与转速立方成正比。实际工作点还受系统阻力曲线、安装方式、过滤网和导流板影响。")
    st.caption("此型号厂家注明主要用于工业蒸发器；在本项目中仅作公开性能参考候选，必须由系统阻力计算与厂家复核后才能定型。")
with tabs[4]:
    st.subheader("144芯CAD坐标点位浏览")
    packs=["全部"]+sorted(coords.Pack_ID.unique().tolist())
    pick=st.selectbox("电池包筛选",packs)
    view=coords if pick=="全部" else coords[coords.Pack_ID==pick]
    fig=px.scatter_3d(view,x="CAD_X_center_mm_rounded",y="CAD_Y_center_mm_rounded",z="CAD_Z_center_mm_rounded",
        color="Pack_ID",symbol="Parallel_Branch",hover_name="Cell_ID",
        hover_data=["Series_Group","Parallel_Branch","CAD_X_center_mm_rounded","CAD_Y_center_mm_rounded","CAD_Z_center_mm_rounded"])
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=600,margin=dict(l=0,r=0,t=20,b=0),scene=dict(aspectmode="data"))
    st.plotly_chart(fig,use_container_width=True)
    st.warning("该图是CAD坐标点位可视化，不是STEP实体级三维渲染；A/B与SG方向仍需原生SolidWorks核对。")
    st.dataframe(pd.DataFrame([
      ["电池舱","8200×2600×2438 mm"],["电池柜","1410×675×1987 mm"],
      ["电芯","295×248×104 mm"],["电芯间距","7 mm"],["电池包间距","29 mm"],
      ["包至柜体侧壁","72 mm"],["前后/顶部/底部间隙","29 mm"],["PCM厚度","7 mm"]
    ],columns=["参数","基线"]),use_container_width=True,hide_index=True)
with tabs[5]:
    st.subheader("模型参数来源与适用性")
    st.markdown(f"- WT-CPCM论文：DOI `{pcm['doi']}`；总吸热焓 {pcm['total_endothermic_enthalpy_J_kg']/1000:.2f} kJ/kg。")
    st.markdown(f"- 风机公开资料：[ebm-papst W3G500-KK07-G2]({FAN['url']})。")
    st.markdown(f"- 候选电芯：{cell['selected_reference']}；容量{cell['capacity_Ah']}Ah，尺寸与本项目CAD尺寸不一致。")
    st.markdown(f"- 由CAD区域文件求得PCM几何总体积约 **{pcm_volume:.5f} m³**；密度为用户可调假设，因此质量随假设变化。")
    st.dataframe(pd.DataFrame([
      ["144芯热源","I²R焦耳热估算","未含熵热，内阻待实测/文献匹配"],
      ["单体质量/比热","LF280K候选/可调值","目标电芯型号未确认"],
      ["PCM密度/比热","可调假设值","论文未提供完整数据"],
      ["PCM阶段焓","王佳论文三阶段吸热焓","高温阶段动力学及可逆性未验证"],
      ["风机性能","厂家公开工作点+相似定律","系统阻力/安装工作点待校核"],
      ["风道流速","总风量/舱体截面","截面平均速度，不是局部风速"],
      ["温度场","集中参数平均温度","非CFD，不解析局部热点"],
    ],columns=["项目","采用方法","限制"]),use_container_width=True,hide_index=True)
    st.warning("本网站用于方案估算、参数敏感性分析和控制策略演示。不能用于热失控安全认证、实际风机采购定型或替代CFD/实验验证。")
st.divider()
st.caption("版本 v3.0 · 有风/无风对比、PCM储热状态及风机调速估算已接入；真实192状态多节点预测与MPC闭环仍待参数校准。")
