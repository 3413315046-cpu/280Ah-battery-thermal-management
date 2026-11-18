
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

BASE=Path(__file__).parent
def loadj(n):
    return json.loads((BASE/"data"/n).read_text(encoding="utf-8"))

coords=pd.read_csv(BASE/"data"/"Stage15_v4_7_1_144电芯整数CAD坐标映射.csv")
pcmgeo=pd.read_csv(BASE/"data"/"Stage15_v4_7_1_24个侧面PCM区域几何参数.csv")

# 用户确认的 WT-CPCM 参数
PCM={"density":1100.0,"cp":1740.0,"k":3.5,"phase_lo":38.0,"phase_hi":43.0,"latent":120000.0}

# 临时工程候选参数：后续拿到真实风机/空调/风口参数后只替换这里
FAN={"name":"轴流风机候选参数","flow_m3h":4110.0,"power_W":230.0}
HVAC={"name":"舱级制冷单元临时参数","capacity_kW":5.0,"cop":3.2}
VENT={"open_min":0.40,"open_max":1.00}

st.set_page_config(page_title="280Ah储能电池舱预测调控平台",page_icon="🔋",layout="wide")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 15% 0%,#13263b 0%,#07111d 38%,#060b12 100%);color:#edf5fb}
.block-container{max-width:1580px;padding-top:1.0rem;padding-bottom:2.2rem}
.hero{padding:28px 32px;border:1px solid #31506a;border-radius:20px;
background:linear-gradient(120deg,#102c45 0%,#123a3b 55%,#1a293c 100%);
box-shadow:0 18px 55px #0007}
.hero h1{font-size:2.15rem;margin:0;color:#fff}.hero p{color:#b9cad9;margin:8px 0 0}
[data-testid="stMetric"]{background:linear-gradient(145deg,#0e1d2c,#112536);border:1px solid #29475d;border-radius:15px;padding:14px 16px}
.card{padding:16px 18px;border:1px solid #29445b;border-radius:14px;background:#0d1a28}
.badge{display:inline-block;padding:5px 10px;border-radius:999px;background:#17364d;color:#8fd9ff;font-size:.86rem}
.small{color:#9fb1c2;font-size:.88rem}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🔋 280Ah 磷酸铁锂储能电池舱 · 预测调控数字孪生</h1>
<p>WT-CPCM 被动储热 + 风机/空调/风口协同调节 + 预测控制对照评价</p>
</div>
""",unsafe_allow_html=True)

st.caption("核心评价：在相同工况下，对比“固定控制对照组”和“预测调控实验组”，同时看温度、均温性、PCM利用率与总能耗。")

with st.sidebar:
    st.markdown("## ⚙️ 工况")
    ambient=st.slider("环境温度（℃）",-10,50,30)
    initial=st.slider("初始电池温度（℃）",10,55,30)
    rate=st.slider("放电倍率（C）",0.2,2.0,1.0,0.1)
    duration=st.slider("运行时长（h）",1,12,6)
    r_mohm=st.number_input("单体等效内阻（mΩ，待校准）",0.05,2.0,0.25,0.05)
    cell_mass=st.number_input("单体质量（kg，候选值）",1.0,15.0,5.49,0.1)
    cell_cp=st.number_input("单体比热（J/(kg·K)，暂定）",500,2000,1000,50)
    nfan=st.slider("风机台数",1,8,4)
    target=st.select_slider("最大温差目标（℃）",[2.0,2.5,3.0,3.5,4.0],value=3.0)
    st.markdown("---")
    st.markdown("### 控制组定义")
    st.info("对照组：风机、空调、风口保持固定开度/功率。\n\n实验组：预测调控器滚动预测温度与温差，联合选择风机、空调和风口控制量。")
    st.markdown("### 当前临时设备参数")
    st.write(f"风机：{FAN['flow_m3h']:.0f} m³/h，{FAN['power_W']:.0f} W/台")
    st.write(f"空调：{HVAC['capacity_kW']:.1f} kW 冷量，COP {HVAC['cop']:.1f}")
    st.write("风口：40–100% 开度")
    st.caption("以上设备参数仅用于网站演示，后续收到真实设备参数后替换。")

pcm_volume=float(pcmgeo["Allocated_PCM_Volume_m3"].sum())
pcm_mass=pcm_volume*PCM["density"]
pcm_area=float(pcmgeo["Cell_PCM_Direct_Contact_Area_m2"].sum())

nz=12; cells_per_zone=12
heat_factor=np.array([.92,.96,1,1.04,1.08,1.02,.94,.98,1.06,1.10,.97,1.03])
air_factor=np.array([.78,.84,.91,.98,1.05,1.12,.80,.88,.95,1.02,1.10,1.18])
Cb=cells_per_zone*cell_mass*cell_cp
mpz=pcm_mass/nz
Qtotal=144*(280*rate)**2*(r_mohm/1000)
qzone=Qtotal*heat_factor/heat_factor.sum()
G_cell_pcm=max(4.0,pcm_area*18.0/nz)
G_inter=18.0
dt=20.0
ts=np.arange(0,duration*3600+dt,dt)
time_h=ts/3600

def pcm_cp(T):
    if PCM["phase_lo"]<=T<=PCM["phase_hi"]:
        return PCM["cp"]+PCM["latent"]/(PCM["phase_hi"]-PCM["phase_lo"])
    return PCM["cp"]

def step_state(oldb,oldp,fan,ac,vent):
    # fan/vent jointly determine air-side heat rejection
    q_ba=(0.35 + 12.0*fan)*(0.55+0.45*vent)*air_factor*(oldb-ambient)
    q_bp=G_cell_pcm*(oldb-oldp)
    mix=np.zeros(nz)
    mix[1:]+=G_inter*(oldb[:-1]-oldb[1:])
    mix[:-1]+=G_inter*(oldb[1:]-oldb[:-1])
    q_ac=HVAC["capacity_kW"]*1000*ac
    # distribute active cooling preferentially to hotter zones
    weights=np.maximum(oldb-ambient,0)+0.15
    weights=weights/weights.sum()
    q_ba += q_ac*weights
    Gpa=1.0+16.0*fan*(0.55+0.45*vent)
    q_pa=Gpa*air_factor*(oldp-ambient)
    dTb=(qzone-q_ba-q_bp+mix)/Cb
    dTp=(q_bp-q_pa)/(mpz*pcm_cp(float(oldp.mean())))
    return oldb+dTb*dt, oldp+dTp*dt

def energy_per_step(fan,ac,vent):
    # electrical fan power follows affinity-law approximation; AC power from COP
    fan_w=nfan*FAN["power_W"]*(fan**3)
    ac_w=HVAC["capacity_kW"]*1000*ac/HVAC["cop"]
    # vent is passive, but a larger opening slightly increases damper/air handling burden
    vent_w=30.0*max(0,vent-0.7)
    return fan_w+ac_w+vent_w

def liquid_fraction(T):
    # 0 below phase range, linear in 38–43, 1 above
    return np.clip((T-PCM["phase_lo"])/(PCM["phase_hi"]-PCM["phase_lo"]),0,1)

def mpc_action(oldb,oldp,target):
    # Receding-horizon surrogate MPC:
    # choose a control triple by rolling the thermal model forward several steps.
    candidates=[]
    for fan in (0.25,0.50,0.75,1.00):
        for ac in (0.0,0.25,0.50,0.75,1.00):
            for vent in (0.45,0.65,0.85,1.00):
                pb=oldb.copy(); pp=oldp.copy(); e=0.0
                for _ in range(6): # 2 min short prediction horizon; action updated every 5 min
                    pb,pp=step_state(pb,pp,fan,ac,vent)
                    e+=energy_per_step(fan,ac,vent)*dt
                tmax=pb.max(); spread=pb.max()-pb.min()
                # temperature safety dominates, then uniformity, then energy
                penalty=0
                if spread>target:
                    penalty += 250000*(spread-target)**2
                if tmax>ambient+18:
                    penalty += 50000*(tmax-(ambient+18))**2
                score=penalty + 9000*tmax + 70000*spread + 0.25*e
                candidates.append((score,fan,ac,vent))
    _,fan,ac,vent=min(candidates,key=lambda x:x[0])
    return fan,ac,vent

def simulate(mode):
    Tb=np.ones((len(ts),nz))*initial
    Tp=np.ones((len(ts),nz))*ambient
    fan=np.zeros(len(ts)); ac=np.zeros(len(ts)); vent=np.zeros(len(ts))
    power=np.zeros(len(ts)); energy=np.zeros(len(ts))
    control_every=int(300/dt) # 5 min
    last=(0.60,0.40,0.70) if mode=="baseline" else (0.25,0.0,0.45)
    for k in range(1,len(ts)):
        if (k-1)%control_every==0:
            if mode=="baseline":
                last=(0.60,0.40,0.70)
            else:
                last=mpc_action(Tb[k-1],Tp[k-1],target)
        fan[k],ac[k],vent[k]=last
        Tb[k],Tp[k]=step_state(Tb[k-1],Tp[k-1],*last)
        power[k]=energy_per_step(*last)
        energy[k]=energy[k-1]+power[k]*dt/3.6e6
    fan[0],ac[0],vent[0]=last
    return Tb,Tp,fan,ac,vent,power,energy

Tb_b,Tp_b,fan_b,ac_b,vent_b,pow_b,en_b=simulate("baseline")
Tb_m,Tp_m,fan_m,ac_m,vent_m,pow_m,en_m=simulate("mpc")

def metrics(Tb,Tp,en,fan,ac,vent):
    tmax=float(Tb.max())
    dmax=float(np.max(Tb.max(axis=1)-Tb.min(axis=1)))
    liquid=float(liquid_fraction(float(Tp.mean(axis=1)[-1])))
    pcm_remaining=max(0.0,1-liquid)
    runtime=float(np.sum(fan>0.05)*dt/3600)
    total=float(en[-1])
    avg=float(Tb.mean())
    return dict(tmax=tmax,dmax=dmax,liquid=liquid,remaining=pcm_remaining,runtime=runtime,total=total,avg=avg,
                fan=float(fan.mean()),ac=float(ac.mean()),vent=float(vent.mean()))

mb=metrics(Tb_b,Tp_b,en_b,fan_b,ac_b,vent_b)
mm=metrics(Tb_m,Tp_m,en_m,fan_m,ac_m,vent_m)

# Header KPIs
st.markdown("### 🎯 同工况对照：固定控制 vs 预测调控")
k1,k2,k3,k4=st.columns(4)
k1.metric("预测调控组最高温度",f"{mm['tmax']:.1f} ℃",f"{mb['tmax']-mm['tmax']:+.1f} ℃")
k2.metric("预测调控组最大温差",f"{mm['dmax']:.2f} ℃",f"{mb['dmax']-mm['dmax']:+.2f} ℃")
k3.metric("预测调控总能耗",f"{mm['total']:.2f} kWh",f"{mb['total']-mm['total']:+.2f} kWh")
k4.metric("总能耗降低",f"{max(0,(mb['total']-mm['total'])/max(mb['total'],1e-9)*100):.1f}%")

tabs=st.tabs(["① 对照结论","② 温度与温差","③ PCM储热","④ 控制动作","⑤ 三维电池结构"])

with tabs[0]:
    st.markdown("#### 两组控制策略")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="card"><span class="badge">方案1 · 对照组</span><h3>固定控制</h3><p>风机60% · 空调40% · 风口70%，整个运行周期保持不变。</p><p class="small">用于模拟传统定值通风/制冷策略。</p></div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><span class="badge">方案2 · 实验组</span><h3>预测调控</h3><p>每5 min滚动预测未来温度状态，联合选择风机、空调和风口控制量。</p><p class="small">目标：温度安全与均匀性优先，同时降低不必要的主动冷却能耗。</p></div>',unsafe_allow_html=True)
    st.markdown("#### 核心评价指标")
    table=pd.DataFrame({
        "指标":["电池最高温度 Tmax","电池组最大温差 ΔTmax","PCM液相率（末端）","PCM剩余储热能力（相对）","风机累计运行时间","系统总能耗","平均电池温度"],
        "固定控制对照组":[f"{mb['tmax']:.2f} ℃",f"{mb['dmax']:.2f} ℃",f"{mb['liquid']*100:.1f} %",f"{mb['remaining']*100:.1f} %",f"{mb['runtime']:.2f} h",f"{mb['total']:.2f} kWh",f"{mb['avg']:.2f} ℃"],
        "预测调控实验组":[f"{mm['tmax']:.2f} ℃",f"{mm['dmax']:.2f} ℃",f"{mm['liquid']*100:.1f} %",f"{mm['remaining']*100:.1f} %",f"{mm['runtime']:.2f} h",f"{mm['total']:.2f} kWh",f"{mm['avg']:.2f} ℃"],
    })
    st.dataframe(table,use_container_width=True,hide_index=True)
    if mm["dmax"]<=target and mm["total"]<=mb["total"]:
        st.success("当前临时参数下，预测调控组同时满足温差目标，并表现出更低的系统总能耗。")
    elif mm["dmax"]<=target:
        st.info("当前临时参数下，预测调控组满足温差目标；总能耗优势仍取决于后续真实设备参数和控制权重校准。")
    else:
        st.warning("当前临时参数下，预测调控组尚未满足温差目标，需要继续校准设备参数或控制权重。")
    st.caption("重要：这里的设备参数是临时工程候选值，模型属于网站演示用降阶热网络；最终论文/竞赛定量结论必须用真实设备参数、CFD或实验数据验证。")

with tabs[1]:
    st.markdown("#### 舱内温度变化对比")
    avg_b=Tb_b.mean(axis=1); avg_m=Tb_m.mean(axis=1)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=time_h,y=avg_b,name="固定控制对照组",line=dict(width=3)))
    fig.add_trace(go.Scatter(x=time_h,y=avg_m,name="预测调控实验组",line=dict(width=3)))
    fig.update_layout(template="plotly_dark",height=390,xaxis_title="时间（h）",yaxis_title="平均电池温度（℃）",margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### 最大温差对比")
    spread_b=Tb_b.max(axis=1)-Tb_b.min(axis=1)
    spread_m=Tb_m.max(axis=1)-Tb_m.min(axis=1)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=time_h,y=spread_b,name="固定控制",line=dict(width=2.5)))
    fig2.add_trace(go.Scatter(x=time_h,y=spread_m,name="预测调控",line=dict(width=2.5)))
    fig2.add_hline(y=target,line_dash="dash",annotation_text=f"目标 {target:.1f}℃")
    fig2.update_layout(template="plotly_dark",height=330,xaxis_title="时间（h）",yaxis_title="最大温差 ΔT（℃）",margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig2,use_container_width=True)
    q1,q2,q3=st.columns(3)
    q1.metric("对照组峰值温度",f"{mb['tmax']:.1f} ℃")
    q2.metric("实验组峰值温度",f"{mm['tmax']:.1f} ℃")
    q3.metric("实验组温差目标","达标" if mm["dmax"]<=target else "未达标")

with tabs[2]:
    st.markdown("#### WT-CPCM固定参数")
    a,b,c,d,e=st.columns(5)
    a.metric("密度","1100 kg/m³"); b.metric("比热容","1740 J/(kg·K)")
    c.metric("导热系数","3.5 W/(m·K)"); d.metric("相变温区","38–43 ℃"); e.metric("潜热","120 kJ/kg")
    avgpb=Tp_b.mean(axis=1); avgpm=Tp_m.mean(axis=1)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=time_h,y=avgpb,name="固定控制 PCM",line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=time_h,y=avgpm,name="预测调控 PCM",line=dict(width=2.5)))
    fig.add_hrect(y0=38,y1=43,fillcolor="#e5b94b",opacity=.13,line_width=0,annotation_text="相变区 38–43℃")
    fig.update_layout(template="plotly_dark",height=360,xaxis_title="时间（h）",yaxis_title="PCM平均温度（℃)",margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)
    liq_b=liquid_fraction(avgpb); liq_m=liquid_fraction(avgpm)
    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=time_h,y=liq_b*100,name="固定控制",line=dict(width=2.5)))
    fig3.add_trace(go.Scatter(x=time_h,y=liq_m*100,name="预测调控",line=dict(width=2.5)))
    fig3.update_layout(template="plotly_dark",height=300,xaxis_title="时间（h）",yaxis_title="PCM液相率（%）",yaxis=dict(range=[0,100]),margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig3,use_container_width=True)
    st.caption(f"当前几何映射得到PCM体积约 {pcm_volume:.5f} m³、按1100 kg/m³估算质量约 {pcm_mass:.1f} kg。")

with tabs[3]:
    st.markdown("#### 预测调控动作")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=time_h,y=fan_m*100,name="风机控制量",line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=time_h,y=ac_m*100,name="空调控制量",line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=time_h,y=vent_m*100,name="风口开度",line=dict(width=2.5)))
    fig.update_layout(template="plotly_dark",height=360,xaxis_title="时间（h）",yaxis_title="控制量（%）",yaxis=dict(range=[0,105]),margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig,use_container_width=True)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=time_h,y=en_b,name="固定控制累计能耗",line=dict(width=2.5)))
    fig2.add_trace(go.Scatter(x=time_h,y=en_m,name="预测调控累计能耗",line=dict(width=2.5)))
    fig2.update_layout(template="plotly_dark",height=330,xaxis_title="时间（h）",yaxis_title="累计电能（kWh）",margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig2,use_container_width=True)
    a,b,c=st.columns(3)
    a.metric("实验组平均风机控制量",f"{mm['fan']*100:.0f}%")
    b.metric("实验组平均空调控制量",f"{mm['ac']*100:.0f}%")
    c.metric("实验组平均风口开度",f"{mm['vent']*100:.0f}%")

with tabs[4]:
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
    st.caption("基于Stage15 v4.7.1 CAD中心坐标的三维点位图。")

st.divider()
st.caption("平台 v5.0 · 固定控制对照组 vs 预测调控实验组 · 风机/空调/风口协同 · WT-CPCM固定参数 · 多指标综合评价")
