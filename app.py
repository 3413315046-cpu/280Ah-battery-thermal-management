import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

BASE=Path(__file__).parent
def loadj(n): return json.loads((BASE/"data"/n).read_text(encoding="utf-8"))
coords=pd.read_csv(BASE/"data"/"Stage15_v4_7_1_144电芯整数CAD坐标映射.csv")
pcmgeo=pd.read_csv(BASE/"data"/"Stage15_v4_7_1_24个侧面PCM区域几何参数.csv")

# User-confirmed PCM parameters from the uploaded parameter table
PCM={"density":1100.0,"cp":1740.0,"k":3.5,"phase_lo":38.0,"phase_hi":43.0,"latent":120000.0}
FAN={"name":"ebm-papst W3G500-KK07-G2（候选参考）","flow":4110.0,"power":230.0,"rpm":970.0,
     "url":"https://www.ebmpapst.com/us/en/products/axial-fans/p/W3G500KK07G2.html"}

st.set_page_config(page_title="280Ah储能电池舱智能热管理",page_icon="🔋",layout="wide")
st.markdown("""
<style>
.stApp{background:linear-gradient(145deg,#07111e 0%,#0b1725 48%,#101d2b 100%);color:#e8f0f8}
.block-container{max-width:1580px;padding-top:1.1rem;padding-bottom:2rem}
.hero{padding:26px 30px;border:1px solid #28445b;border-radius:18px;background:linear-gradient(110deg,#102b43,#153a3b 65%,#253b39);box-shadow:0 12px 35px #0005}
.hero h1{color:#f4f8fc;font-size:2.1rem;margin:0}.hero p{color:#b9cbd9;margin:8px 0 0}
[data-testid="stMetric"]{background:#102236;border:1px solid #29445a;border-radius:14px;padding:14px 16px}
div[data-testid="stTabs"] button{font-weight:650}
.section{font-size:1.2rem;font-weight:700;color:#e9f2fb}
div[data-testid="stAlert"]{border-radius:12px}
</style>""",unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🔋 280Ah 磷酸铁锂储能电池舱</h1><p>电芯温度预测 · WT-CPCM相变储热 · 自适应风机协同控制</p></div>',unsafe_allow_html=True)
st.caption("智能热管理交互平台 · 144芯 / 24个PCM区域 / 目标最大温差 ≤ 3℃")

with st.sidebar:
    st.markdown("## ⚙️ 工况设置")
    ambient=st.slider("环境温度（℃）", -10, 50, 30)
    initial=st.slider("电池初始温度（℃）", 10, 55, 30)
    rate=st.slider("放电倍率（C）",0.2,2.0,1.0,0.1)
    duration=st.slider("运行时长（h）",1,12,6)
    r_mohm=st.number_input("单体等效内阻（mΩ，待校准）",0.05,2.0,0.25,0.05)
    cell_mass=st.number_input("单体质量（kg，候选值）",1.0,15.0,5.49,0.1)
    cell_cp=st.number_input("单体比热（J/(kg·K)，暂定）",500,2000,1000,50)
    nfan=st.slider("风机台数",1,8,4)
    target=st.select_slider("最大温差控制目标（℃）",options=[2.0,2.5,3.0,3.5,4.0],value=3.0)
    st.markdown("---")
    st.markdown("**自动控制模式**")
    st.success("AUTO · 温差反馈调速")
    st.caption("控制器根据12个热区预测温差自动调节风机转速，不提供手动风速栏目。")

# Derived geometry and thermal parameters
pcm_volume=float(pcmgeo["Allocated_PCM_Volume_m3"].sum())
pcm_mass=pcm_volume*PCM["density"]
pcm_area=float(pcmgeo["Cell_PCM_Direct_Contact_Area_m2"].sum())
# 12 lumped thermal zones represent 144 cells; nonuniform heat and cooling are explicitly modeled as assumptions
nz=12; cells_per_zone=12
heat_factor=np.array([0.92,0.96,1.00,1.04,1.08,1.02,0.94,0.98,1.06,1.10,0.97,1.03])
air_factor=np.array([0.78,0.84,0.91,0.98,1.05,1.12,0.80,0.88,0.95,1.02,1.10,1.18])
Cb=cells_per_zone*cell_mass*cell_cp
mpz=pcm_mass/nz
# Effective PCM cp includes latent heat spread over 38-43 C interval
def pcm_cp_eff(T):
    return PCM["cp"]+PCM["latent"]/(PCM["phase_hi"]-PCM["phase_lo"]) if PCM["phase_lo"]<=T<=PCM["phase_hi"] else PCM["cp"]
# candidate fan affinity curve; system airflow utilization is an explicit calibration parameter
util=0.08
q_rated=nfan*FAN["flow"]
Qtotal=144*(280*rate)**2*(r_mohm/1000)
qzone=Qtotal*heat_factor/heat_factor.sum()
# Thermal network coefficients; engineering assumptions pending CFD/measurement
G_cell_pcm=max(4.0,pcm_area*18.0/nz) # W/K per zone
G_pcm_air_nat=1.0
G_pcm_air_fan=16.0
G_cell_air_nat=0.35
G_cell_air_fan=12.0
G_inter=18.0
dt=20.0; ts=np.arange(0,duration*3600+dt,dt)
def simulate(auto=True,fan_off=False):
    Tb=np.ones((len(ts),nz))*initial
    Tp=np.ones((len(ts),nz))*ambient
    speed=np.zeros(len(ts)); pwr=np.zeros(len(ts)); spread=np.zeros(len(ts))
    # Initial PCM enthalpy is referenced to ambient; sensible+latent apparent heat capacity used in time integration.
    for k in range(1,len(ts)):
        oldb=Tb[k-1].copy(); oldp=Tp[k-1].copy()
        d=oldb.max()-oldb.min()
        # Feedback controller: minimum speed demand rises with spread and hottest-zone temperature.
        if fan_off: u=0.0
        elif auto:
            u=np.clip(0.18+0.34*max(0,d-0.7)+0.025*max(0,oldb.max()-(ambient+5)),0.20,1.0)
            if d>target: u=1.0
        else: u=0.65
        speed[k]=u*100
        # Affinity law estimate: P~N^3, flow~N; not a measured installed fan curve.
        pwr[k]=nfan*FAN["power"]*u**3
        qfan=(nfan*FAN["flow"]*u/3600)*1.2*1005*util
        # zone-specific air cooling and PCM coupling
        q_ba=(G_cell_air_nat+G_cell_air_fan*u)*air_factor*(oldb-ambient)
        q_bp=G_cell_pcm*(oldb-oldp)
        # axial neighbor mixing reduces spatial temperature spread; periodic ends avoided
        mix=np.zeros(nz)
        mix[1:]+=G_inter*(oldb[:-1]-oldb[1:])
        mix[:-1]+=G_inter*(oldb[1:]-oldb[:-1])
        # distribute the fan's effective air-side capability across zones
        q_ba += (qfan/nz)*np.maximum(oldb-ambient,0)/(np.mean(np.maximum(oldb-ambient,0))+1e-6)
        dTb=(qzone-q_ba-q_bp+mix)/Cb
        # PCM rejects heat to air; latent apparent heat capacity buffers the 38-43C transition
        Gpa=G_pcm_air_nat+G_pcm_air_fan*u
        q_pa=Gpa*air_factor*(oldp-ambient)
        dTp=(q_bp-q_pa)/(mpz*PCM["cp"] + (mpz*PCM["latent"]/(PCM["phase_hi"]-PCM["phase_lo"]) if PCM["phase_lo"]<=oldp.mean()<=PCM["phase_hi"] else 0))
        Tb[k]=oldb+dTb*dt
        Tp[k]=oldp+dTp*dt
        speed[k]=u*100
        pwr[k]=nfan*FAN["power"]*u**3
        spread[k]=Tb[k].max()-Tb[k].min()
    spread[0]=0
    return Tb,Tp,speed,pwr,spread
Tb_on,Tp_on,sp_on,pw_on,sp_on_delta=simulate(auto=True)
Tb_off,Tp_off,sp_off,pw_off,sp_off_delta=simulate(fan_off=True)
fan_speed_avg=float(np.mean(sp_on[1:]))
max_delta=float(np.max(sp_on_delta))
peak=float(np.max(Tb_on))
final_pcm=float(np.mean(Tp_on[-1]))
stored=(mpz* nz*PCM["cp"]*(final_pcm-ambient))/1e6
time_h=ts/3600
tabs=st.tabs(["数字孪生总览","有风 / 无风温度对比","PCM储热状态","三维电池结构"])
with tabs[0]:
    st.markdown("### 实时工况与自动控制结果")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("最高电芯温度",f"{peak:.1f} ℃")
    c2.metric("最大温差",f"{max_delta:.2f} ℃",delta=f"目标 ≤ {target:.1f} ℃",delta_color="normal" if max_delta<=target else "inverse")
    c3.metric("风机平均调速指令",f"{fan_speed_avg:.0f}%")
    c4.metric("末端PCM平均温度",f"{final_pcm:.1f} ℃")
    if max_delta<=target:
        st.success(f"本次集中参数估算中，温差控制目标达到：最大温差 {max_delta:.2f}℃ ≤ {target:.1f}℃。")
    else:
        st.error(f"本次估算未达到温差目标：最大温差 {max_delta:.2f}℃ > {target:.1f}℃。")
    st.caption("这是12热区降阶工程估算的控制结果；并非144芯逐芯实测/CFD验证。控制目标不能据此宣称真实电池舱已保证≤3℃。")
    fig=go.Figure()
    for i in range(nz):
        fig.add_trace(go.Scatter(x=time_h,y=Tb_on[:,i],mode="lines",line=dict(width=1.2),name=f"热区{i+1}",showlegend=i in [0,5,11]))
    fig.add_hline(y=ambient+5,line_dash="dot",annotation_text="环境温度+5℃参考线")
    fig.update_layout(height=390,xaxis_title="时间（h）",yaxis_title="热区电池温度（℃)",template="plotly_dark",margin=dict(l=10,r=10,t=25,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### 自动风机控制轨迹")
    f=go.Figure()
    f.add_trace(go.Scatter(x=time_h,y=sp_on,name="自动风机调速指令",line=dict(color="#31c7b5",width=3)))
    f.update_layout(height=240,xaxis_title="时间（h）",yaxis_title="转速指令（%）",template="plotly_dark",margin=dict(l=10,r=10,t=20,b=10),yaxis=dict(range=[0,105]))
    st.plotly_chart(f,use_container_width=True)
with tabs[1]:
    st.markdown("### 相同工况下有风机与无风机对比")
    avg_on=Tb_on.mean(axis=1); avg_off=Tb_off.mean(axis=1)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=time_h,y=avg_off,name="无风机 / 自然冷却",line=dict(color="#f39b62",width=3)))
    fig.add_trace(go.Scatter(x=time_h,y=avg_on,name="自动风机控制",line=dict(color="#32c7b4",width=3)))
    fig.update_layout(height=430,xaxis_title="时间（h）",yaxis_title="电池平均温度（℃）",template="plotly_dark",margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)
    d1,d2,d3=st.columns(3)
    d1.metric("无风机末端温度",f"{avg_off[-1]:.1f} ℃")
    d2.metric("自动风机末端温度",f"{avg_on[-1]:.1f} ℃")
    d3.metric("末端温度改善",f"{avg_off[-1]-avg_on[-1]:.1f} ℃")
    st.caption("两条曲线使用相同热源、初始条件和PCM参数；温度是12区降阶模型的平均值。")
with tabs[2]:
    st.markdown("### WT-CPCM固定材料参数")
    a,b,c,d,e=st.columns(5)
    a.metric("密度","1100 kg/m³")
    b.metric("比热容","1740 J/(kg·K)")
    c.metric("导热系数","3.5 W/(m·K)")
    d.metric("相变温区","38–43 ℃")
    e.metric("相变潜热","120 kJ/kg")
    st.markdown("### PCM温度与储热状态")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=time_h,y=Tp_on.mean(axis=1),name="自动风机：PCM平均温度",line=dict(color="#b58cff",width=3)))
    fig.add_trace(go.Scatter(x=time_h,y=Tp_off.mean(axis=1),name="无风机：PCM平均温度",line=dict(color="#f2bd65",width=3)))
    fig.add_hrect(y0=38,y1=43,fillcolor="#e5b94b",opacity=.13,line_width=0,annotation_text="相变温区 38–43℃")
    fig.update_layout(height=390,xaxis_title="时间（h）",yaxis_title="PCM平均温度（℃）",template="plotly_dark",margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.metric("PCM几何估算总体积",f"{pcm_volume:.5f} m³")
    st.metric("按1100 kg/m³计算的PCM质量",f"{pcm_mass:.1f} kg")
    st.caption("PCM体积来自当前Stage15 v4.7.1的24区几何映射；仍需确认实际填充率、封装体积与有效材料体积。")
with tabs[3]:
    st.markdown("### 144芯CAD坐标点位")
    packlist=["全部电池包"]+sorted(coords.Pack_ID.astype(str).unique().tolist())
    choice=st.selectbox("电池包筛选",packlist)
    v=coords if choice=="全部电池包" else coords[coords.Pack_ID.astype(str)==choice]
    fig=px.scatter_3d(v,x="CAD_X_center_mm_rounded",y="CAD_Y_center_mm_rounded",z="CAD_Z_center_mm_rounded",
      color="Pack_ID",symbol="Parallel_Branch",hover_name="Cell_ID",
      hover_data=["Series_Group","Parallel_Branch","CAD_X_center_mm_rounded","CAD_Y_center_mm_rounded","CAD_Z_center_mm_rounded"])
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=620,template="plotly_dark",scene=dict(aspectmode="data"),margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig,use_container_width=True)
    st.caption("这是基于Stage15 v4.7.1 CAD中心坐标的三维点位图，不是STEP实体渲染。")

st.divider()
st.caption("平台 v4.0 · 自动风机温差反馈控制 · 固定WT-CPCM参数 · 有风/无风对比 · 12热区降阶工程估算")
