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

PCM={"density":1100.0,"cp":1740.0,"k":3.5,"phase_lo":38.0,"phase_hi":43.0,"latent":120000.0}
FANS={
 "EC轴流500 mm":{"rated_flow":4500.0,"rated_pressure":350.0,"rated_power":200.0,"rated_rpm":1800.0,"voltage":"48 VDC",
 "pq":[(0,350),(1000,330),(2000,290),(3000,230),(4000,150),(4500,0)],"diameter":450,"area":0.159},
 "EC离心355 mm":{"rated_flow":1800.0,"rated_pressure":1442.0,"rated_power":500.0,"rated_rpm":2600.0,"voltage":"230 VAC",
 "pq":[(0,1442),(400,1350),(800,1200),(1200,1000),(1600,700),(1800,0)],"diameter":355,"area":0.099}
}
HVACS={
 "壁挂式7.5 kW":{"capacity":7.5,"input":3.2,"cop":7.5/3.2,"airflow":2000,"type":"fixed","heat":8.0},
 "顶装变频20 kW":{"capacity":20.0,"input":9.1,"cop":20/9.1,"airflow":6000,"type":"variable","heat":22.0}
}
SMALL={"8038型":{"power":5.0,"flow":130.0,"voltage":"12 VDC"},"6025型":{"power":3.24,"flow":67.9,"voltage":"12 VDC"}}
DAMPER={"area":0.10,"actuator_power":5.0,"zeta":[(0,float("inf")),(25,18),(50,6),(75,2.2),(100,.5)]}

st.set_page_config(page_title="280Ah储能电池舱预测调控平台",page_icon="🔋",layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 15% 0%,#13263b 0%,#07111d 38%,#060b12 100%);color:#edf5fb}
.block-container{max-width:1600px;padding-top:1rem;padding-bottom:2.2rem}
.hero{padding:28px 32px;border:1px solid #31506a;border-radius:20px;background:linear-gradient(120deg,#102c45,#123a3b 55%,#1a293c);box-shadow:0 18px 55px #0007}
.hero h1{font-size:2.1rem;margin:0;color:#fff}.hero p{color:#b9cad9;margin:8px 0 0}
[data-testid="stMetric"]{background:linear-gradient(145deg,#0e1d2c,#112536);border:1px solid #29475d;border-radius:15px;padding:14px 16px}
.card{padding:16px 18px;border:1px solid #29445b;border-radius:14px;background:#0d1a28}
.small{color:#9fb1c2;font-size:.88rem}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🔋 280Ah 磷酸铁锂储能电池舱 · 智能预测调控</h1><p>固定控制对照组 vs 预测调控实验组｜WT-CPCM储热｜风机·空调·风阀协同</p></div>',unsafe_allow_html=True)
st.caption("设备参数已按本轮提供的规格表接入；对照组保持固定控制，实验组采用预测调控并叠加设备保护阈值。")

with st.sidebar:
    st.markdown("## ⚙️ 工况输入")
    ambient=st.slider("环境温度（℃）",-10,50,30)
    initial=st.slider("初始电池温度（℃）",10,55,30)
    rate=st.slider("放电倍率（C）",.2,2.,1.,.1)
    duration=st.slider("运行时长（h）",1,12,6)
    r_mohm=st.number_input("单体等效内阻（mΩ，待校准）",.05,2.,.25,.05)
    cell_mass=st.number_input("单体质量（kg，暂定）",1.,15.,5.49,.1)
    cell_cp=st.number_input("单体比热（J/(kg·K)，暂定）",500,2000,1000,50)
    target=st.select_slider("最大温差控制目标（℃）",[2.,2.5,3.,3.5,4.],value=3.)
    st.markdown("---")
    st.markdown("## 🌀 设备配置")
    fan_type=st.selectbox("舱级风机",list(FANS.keys()),index=0)
    nfan=st.number_input("舱级风机数量",1,12,4)
    hvac_type=st.selectbox("空调机组",list(HVACS.keys()),index=1)
    small_type=st.selectbox("模组级DC风扇",["不配置",*SMALL.keys()],index=1)
    nsmall=st.number_input("小型DC风扇数量",0,144,24)
    st.markdown("---")
    st.markdown("## 🧰 对照组固定设定")
    base_fan=st.slider("对照组风机转速（%）",0,100,60,5)
    base_ac=st.slider("对照组空调输出（%）",0,100,40,5)
    base_vent=st.slider("对照组风阀开度（%）",25,100,70,5)
    fire=st.toggle("消防信号触发（25% LEL）",False)
    fan_fault=st.toggle("主风机故障信号",False)
    compressor_overload=st.toggle("空调压缩机过载",False)

fan=FANS[fan_type]; hvac=HVACS[hvac_type]
pcm_volume=float(pcmgeo["Allocated_PCM_Volume_m3"].sum())
pcm_mass=pcm_volume*PCM["density"]
pcm_area=float(pcmgeo["Cell_PCM_Direct_Contact_Area_m2"].sum())
nz=12; cells_per_zone=12
heat_factor=np.array([.92,.96,1,1.04,1.08,1.02,.94,.98,1.06,1.10,.97,1.03])
air_factor=np.array([.78,.84,.91,.98,1.05,1.12,.80,.88,.95,1.02,1.10,1.18])
Cb=cells_per_zone*cell_mass*cell_cp; mpz=pcm_mass/nz
Qtotal=144*(280*rate)**2*(r_mohm/1000)
qzone=Qtotal*heat_factor/heat_factor.sum()
G_cell_pcm=max(4.,pcm_area*18./nz); G_inter=18.
dt=30.; ts=np.arange(0,duration*3600+dt,dt); time_h=ts/3600

def zeta_at(opening):
    xs=np.array([p[0] for p in DAMPER["zeta"][1:]],float)
    ys=np.array([p[1] for p in DAMPER["zeta"][1:]],float)
    return float(np.interp(np.clip(opening,25,100),xs,ys))
def fan_operating_point(speed,opening):
    if speed<=0: return 0.,0.,0.
    # Parallel fan curves: interpolate each fan P-Q curve, then scale by affinity laws.
    qgrid=np.linspace(0,fan["rated_flow"]*nfan*speed,300)
    q_each=qgrid/nfan/max(speed,1e-6)
    pfan=np.interp(q_each,[x[0] for x in fan["pq"]],[x[1] for x in fan["pq"]])*speed**2
    area=max(.01,DAMPER["area"]*max(opening/100,.05))
    velocity=(qgrid/3600)/area
    z=zeta_at(opening)
    psys=z*1.2*velocity**2/2
    idx=int(np.argmin(np.abs(pfan-psys)))
    q=float(qgrid[idx]); dp=float(pfan[idx])
    return q,dp,velocity[idx]
def equipment(fan_cmd,ac_cmd,vent_cmd,small_cmd=0):
    q,dp,vel=fan_operating_point(fan_cmd,vent_cmd)
    fan_power=nfan*fan["rated_power"]*fan_cmd**3
    small_power=nsmall*(SMALL[small_type]["power"] if small_type!="不配置" else 0)*small_cmd
    if hvac["type"]=="variable":
        freq=0 if ac_cmd<=0 else 30+30*ac_cmd
        cap=0 if freq==0 else float(np.interp(freq,[30,40,50,60],[12,15,20,22]))
        ac_power=0 if freq==0 else hvac["input"]*1000*(freq/50)**1.5
    else:
        freq=50 if ac_cmd>0 else 0
        cap=hvac["capacity"]*ac_cmd
        ac_power=hvac["input"]*1000*ac_cmd
    actuator=DAMPER["actuator_power"] if vent_cmd>0 else 0
    total=fan_power+small_power+ac_power+actuator
    return dict(q=q,dp=dp,vel=vel,fan_w=fan_power,small_w=small_power,ac_w=ac_power,act_w=actuator,
                total_w=total,cooling_kw=cap,frequency=freq)

def pcm_cp(T):
    return PCM["cp"]+PCM["latent"]/(PCM["phase_hi"]-PCM["phase_lo"]) if PCM["phase_lo"]<=T<=PCM["phase_hi"] else PCM["cp"]
def step_state(oldb,oldp,fan_cmd,ac_cmd,vent_cmd):
    eq=equipment(fan_cmd,ac_cmd,vent_cmd,1.0 if small_type!="不配置" else 0.0)
    # Air-side exchange scales with actual fan flow and valve opening; HVAC cools return-air zone.
    air_gain=np.clip(eq["q"]/max(nfan*fan["rated_flow"],1)*12.,0,18.)
    q_ba=(.35+air_gain)*(0.55+0.45*vent_cmd/100)*air_factor*(oldb-ambient)
    q_bp=G_cell_pcm*(oldb-oldp)
    mix=np.zeros(nz); mix[1:]+=G_inter*(oldb[:-1]-oldb[1:]); mix[:-1]+=G_inter*(oldb[1:]-oldb[:-1])
    q_ac=eq["cooling_kw"]*1000
    weights=np.maximum(oldb-ambient,0)+.15; weights=weights/weights.sum()
    q_ba+=q_ac*weights
    Gpa=1.+16.*(eq["q"]/max(nfan*fan["rated_flow"],1))*(.55+.45*vent_cmd/100)
    q_pa=Gpa*air_factor*(oldp-ambient)
    dTb=(qzone-q_ba-q_bp+mix)/Cb
    dTp=(q_bp-q_pa)/(mpz*pcm_cp(float(oldp.mean())))
    return oldb+dTb*dt,oldp+dTp*dt,eq

def energy_step(eq):
    return eq["total_w"]*dt/3.6e6

def liquid_fraction(T):
    return float(np.clip((T-PCM["phase_lo"])/(PCM["phase_hi"]-PCM["phase_lo"]),0,1))
def threshold_guard(action,oldb,return_temp):
    f,a,v=action; alarms=[]
    spread=float(oldb.max()-oldb.min())
    if fire:
        alarms.append("消防联锁：空调关闭、风阀关闭、风机停机")
        return (0.,0.,0.),alarms
    if return_temp>32:
        f=max(f,.0) if fan_fault else 1.; a=1.; alarms.append("回风温度>32℃：风机100%、空调全功率")
    elif return_temp>28:
        f=max(f,.5); a=max(a,.35); alarms.append("回风温度>28℃：空调制冷、风机至少50%")
    if spread>5:
        v=min(100,v+20); f=min(1.,f+.2); alarms.append("模组温差>5℃：风阀开度+20%、风机提速")
    if compressor_overload and a>0:
        if hvac["type"]=="variable":
            a=min(a,0.0 if 30<=30 else .0) # 30Hz minimum-frequency fallback
            a=.01
        else:
            a=min(a,.5)
        alarms.append("空调压缩机过载：降载运行并报警")
    if fan_fault:
        f=min(f,.5); alarms.append("主风机故障：切换备用风机策略并报警（备用容量待补充）")
    return (f,a,v),alarms

def mpc_action(oldb,oldp):
    # Receding-horizon surrogate MPC. Equipment power is computed from supplied ratings and fan affinity laws.
    candidates=[]
    for f in (.25,.5,.75,1.):
        for a in (0.,.25,.5,.75,1.):
            for v in (25.,50.,75.,100.):
                pb=oldb.copy(); pp=oldp.copy(); e=0.
                for _ in range(4):
                    pb,pp,eq=step_state(pb,pp,f,a,v)
                    e+=energy_step(eq)
                spread=pb.max()-pb.min(); tmax=pb.max()
                penalty=0.
                if spread>target: penalty+=250000*(spread-target)**2
                if tmax>ambient+18: penalty+=50000*(tmax-(ambient+18))**2
                score=penalty+9000*tmax+70000*spread+.25*e
                candidates.append((score,f,a,v))
    _,f,a,v=min(candidates,key=lambda x:x[0])
    return f,a,v

def simulate(mode):
    Tb=np.ones((len(ts),nz))*initial; Tp=np.ones((len(ts),nz))*ambient
    fc=np.zeros(len(ts)); ac=np.zeros(len(ts)); vc=np.zeros(len(ts))
    fanw=np.zeros(len(ts)); acw=np.zeros(len(ts)); smallw=np.zeros(len(ts)); actw=np.zeros(len(ts))
    airflow=np.zeros(len(ts)); pressure=np.zeros(len(ts)); velocity=np.zeros(len(ts)); cooling=np.zeros(len(ts))
    power=np.zeros(len(ts)); energy=np.zeros(len(ts)); alarms=[]
    update_every=max(1,int(300/dt))
    last=(base_fan/100,base_ac/100,base_vent) if mode=="baseline" else (.25,0.,45.)
    for k in range(1,len(ts)):
        if (k-1)%update_every==0:
            last=(base_fan/100,base_ac/100,base_vent) if mode=="baseline" else mpc_action(Tb[k-1],Tp[k-1])
        ret=float(Tb[k-1].mean())
        if mode=="mpc":
            last,al=threshold_guard(last,Tb[k-1],ret)
            if al: alarms.extend(al)
        f,a,v=last
        Tb[k],Tp[k],eq=step_state(Tb[k-1],Tp[k-1],f,a,v)
        fc[k],ac[k],vc[k]=f,a,v
        fanw[k],acw[k],smallw[k],actw[k]=eq["fan_w"],eq["ac_w"],eq["small_w"],eq["act_w"]
        airflow[k],pressure[k],velocity[k],cooling[k]=eq["q"],eq["dp"],eq["vel"],eq["cooling_kw"]
        power[k]=eq["total_w"]; energy[k]=energy[k-1]+energy_step(eq)
    fc[0],ac[0],vc[0]=last
    return dict(Tb=Tb,Tp=Tp,fan=fc,ac=ac,vent=vc,fanw=fanw,acw=acw,smallw=smallw,actw=actw,
                airflow=airflow,pressure=pressure,velocity=velocity,cooling=cooling,power=power,energy=energy,alarms=sorted(set(alarms)))

base=simulate("baseline"); pred=simulate("mpc")
def metrics(s):
    b=s["Tb"]; p=s["Tp"]
    return {"tmax":float(b.max()),"dmax":float(np.max(b.max(axis=1)-b.min(axis=1))),
    "avg":float(b.mean()),"liquid":liquid_fraction(float(p.mean(axis=1)[-1])),
    "remaining":1-liquid_fraction(float(p.mean(axis=1)[-1])),
    "runtime":float(np.sum(s["fan"]>.05)*dt/3600),"energy":float(s["energy"][-1]),
    "fan_energy":float(np.sum(s["fanw"])*dt/3.6e6),"ac_energy":float(np.sum(s["acw"])*dt/3.6e6)}
mb=metrics(base); mm=metrics(pred)
st.markdown("### 🎯 同工况对照：固定控制 vs 预测调控")
k1,k2,k3,k4=st.columns(4)
k1.metric("预测组最高温度",f"{mm['tmax']:.1f} ℃",f"{mb['tmax']-mm['tmax']:+.1f} ℃")
k2.metric("预测组最大温差",f"{mm['dmax']:.2f} ℃",f"{mb['dmax']-mm['dmax']:+.2f} ℃")
k3.metric("预测组总能耗",f"{mm['energy']:.2f} kWh",f"{mb['energy']-mm['energy']:+.2f} kWh")
k4.metric("预测组风机累计运行",f"{mm['runtime']:.2f} h")

tabs=st.tabs(["① 对照评价","② 温度与温差","③ PCM储热","④ 设备控制与能耗","⑤ 设备参数库","⑥ 三维电池结构"])
with tabs[0]:
    c1,c2=st.columns(2)
    with c1: st.markdown('<div class="card"><h3>方案1 · 固定控制对照组</h3><p>风机、空调和风阀维持用户设定的固定输出，不根据温度动态调整。</p></div>',unsafe_allow_html=True)
    with c2: st.markdown('<div class="card"><h3>方案2 · 预测调控实验组</h3><p>每5分钟滚动预测热状态并优化设备控制量，同时执行温度阈值和消防/故障保护。</p></div>',unsafe_allow_html=True)
    rows=["电池最高温度 Tmax","电池组最大温差 ΔTmax","PCM液相率（末端）","PCM剩余储热能力（相对）","风机累计运行时间","系统总能耗","平均电池温度"]
    def fmt(m): return [f"{m['tmax']:.2f} ℃",f"{m['dmax']:.2f} ℃",f"{m['liquid']*100:.1f} %",f"{m['remaining']*100:.1f} %",f"{m['runtime']:.2f} h",f"{m['energy']:.2f} kWh",f"{m['avg']:.2f} ℃"]
    st.dataframe(pd.DataFrame({"评价指标":rows,"固定控制对照组":fmt(mb),"预测调控实验组":fmt(mm)}),use_container_width=True,hide_index=True)
    if mm["dmax"]<=target and mm["energy"]<mb["energy"]: st.success("当前模型工况下，预测组满足温差目标且总能耗较低。")
    elif mm["dmax"]<=target: st.info("预测组满足当前温差目标；能耗优势需结合实际设备参数和控制权重继续优化。")
    else: st.warning("预测组当前未满足温差目标，需进一步校准控制参数。")
    if pred["alarms"]: st.warning("控制/保护事件："+ "；".join(pred["alarms"]))
    st.caption("结果来自集中参数降阶模型和你提供的设备规格数据；并非实测或CFD结果。")
with tabs[1]:
    avg0=base["Tb"].mean(axis=1); avg1=pred["Tb"].mean(axis=1)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=time_h,y=avg0,name="固定控制对照组",line=dict(width=3)))
    fig.add_trace(go.Scatter(x=time_h,y=avg1,name="预测调控实验组",line=dict(width=3)))
    fig.update_layout(template="plotly_dark",height=390,xaxis_title="时间（h）",yaxis_title="平均电池温度（℃)")
    st.plotly_chart(fig,use_container_width=True)
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=time_h,y=base["Tb"].max(axis=1)-base["Tb"].min(axis=1),name="固定控制"))
    fig2.add_trace(go.Scatter(x=time_h,y=pred["Tb"].max(axis=1)-pred["Tb"].min(axis=1),name="预测调控"))
    fig2.add_hline(y=target,line_dash="dash",annotation_text=f"目标 {target}℃")
    fig2.update_layout(template="plotly_dark",height=340,xaxis_title="时间（h）",yaxis_title="最大温差 ΔT（℃)")
    st.plotly_chart(fig2,use_container_width=True)
with tabs[2]:
    st.markdown("#### WT-CPCM 固定材料参数")
    cols=st.columns(5)
    for col,(name,val) in zip(cols,[("密度","1100 kg/m³"),("比热容","1740 J/(kg·K)"),("导热系数","3.5 W/(m·K)"),("相变温区","38–43 ℃"),("相变潜热","120 kJ/kg")]): col.metric(name,val)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=time_h,y=base["Tp"].mean(axis=1),name="固定控制 PCM"))
    fig.add_trace(go.Scatter(x=time_h,y=pred["Tp"].mean(axis=1),name="预测调控 PCM"))
    fig.add_hrect(y0=38,y1=43,fillcolor="#e5b94b",opacity=.14,line_width=0,annotation_text="相变区")
    fig.update_layout(template="plotly_dark",height=350,xaxis_title="时间（h）",yaxis_title="PCM平均温度（℃)")
    st.plotly_chart(fig,use_container_width=True)
    lf0=np.array([liquid_fraction(x) for x in base["Tp"].mean(axis=1)])
    lf1=np.array([liquid_fraction(x) for x in pred["Tp"].mean(axis=1)])
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=time_h,y=lf0*100,name="固定控制"))
    fig2.add_trace(go.Scatter(x=time_h,y=lf1*100,name="预测调控"))
    fig2.update_layout(template="plotly_dark",height=300,xaxis_title="时间（h）",yaxis_title="PCM液相率（%）",yaxis=dict(range=[0,100]))
    st.plotly_chart(fig2,use_container_width=True)
    st.write(f"PCM几何总体积：{pcm_volume:.5f} m³；按1100 kg/m³估算质量：{pcm_mass:.1f} kg。")
with tabs[3]:
    fig=go.Figure()
    for arr,name in [(pred["fan"]*100,"风机转速指令"),(pred["ac"]*100,"空调输出指令"),(pred["vent"],"风阀开度")]:
        fig.add_trace(go.Scatter(x=time_h,y=arr,name=name))
    fig.update_layout(template="plotly_dark",height=340,xaxis_title="时间（h）",yaxis_title="控制量（%）",yaxis=dict(range=[0,105]))
    st.plotly_chart(fig,use_container_width=True)
    fig2=go.Figure()
    for arr,name in [(pred["fanw"],"舱级风机功率"),(pred["acw"],"空调输入功率"),(pred["smallw"],"模组风扇功率"),(pred["actw"],"风阀执行器功率")]:
        fig2.add_trace(go.Scatter(x=time_h,y=arr,name=name))
    fig2.update_layout(template="plotly_dark",height=330,xaxis_title="时间（h）",yaxis_title="功率（W）")
    st.plotly_chart(fig2,use_container_width=True)
    eq_last=equipment(float(pred["fan"][-1]),float(pred["ac"][-1]),float(pred["vent"][-1]),1)
    a,b,c,d=st.columns(4)
    a.metric("风量估算",f"{eq_last['q']:.0f} m³/h")
    b.metric("风机工作静压",f"{eq_last['dp']:.0f} Pa")
    c.metric("风口平均风速",f"{eq_last['vel']:.2f} m/s")
    d.metric("空调制冷量",f"{eq_last['cooling_kw']:.1f} kW")
    st.markdown("#### 累计能耗分项")
    en=pd.DataFrame({"设备":["舱级风机","空调","模组级风扇"],"固定控制(kWh)":[mb["fan_energy"],mb["ac_energy"],float(np.sum(base["smallw"])*dt/3.6e6)],"预测调控(kWh)":[mm["fan_energy"],mm["ac_energy"],float(np.sum(pred["smallw"])*dt/3.6e6)]})
    st.dataframe(en,use_container_width=True,hide_index=True)
    st.caption("风机工作点按提供的P-Q离散点插值，并用风阀阻力关系估算；管网阻力、风道分配及系统曲线仍需实测/CFD校核。")
with tabs[4]:
    st.markdown("#### 风机设备库")
    for name,obj in FANS.items():
        with st.expander(name):
            st.write(f"额定风量 {obj['rated_flow']} m³/h · 静压 {obj['rated_pressure']} Pa · 功率 {obj['rated_power']} W · 转速 {obj['rated_rpm']} rpm · {obj['voltage']}")
            st.dataframe(pd.DataFrame(obj["pq"],columns=["风量 (m³/h)","静压 (Pa)"]),use_container_width=True,hide_index=True)
    st.markdown("#### 空调设备库")
    st.dataframe(pd.DataFrame([{"型号":k,"额定制冷量(kW)":v["capacity"],"额定输入功率(kW)":v["input"],"EER/COP":round(v["cop"],2),"内风量(m³/h)":v["airflow"]} for k,v in HVACS.items()]),use_container_width=True,hide_index=True)
    st.markdown("#### 小型DC风扇与电动风阀")
    st.dataframe(pd.DataFrame([{"型号":k,"功率(W)":v["power"],"风量(m³/h)":v["flow"],"电压":v["voltage"]} for k,v in SMALL.items()]),use_container_width=True,hide_index=True)
    st.write("电动风阀：400×300 mm，有效面积0.10 m²，24 VAC，执行器5 W；阻力系数按开度插值。")
    st.markdown("#### 控制阈值与保护")
    st.dataframe(pd.DataFrame([
    ["回风温度>28℃","空调制冷；风机≥50%"],
    ["回风温度>32℃","空调全功率；风机100%"],
    ["回风温度<18℃","配置制热时启动制热"],
    ["模组温差>5℃","对应区域风阀+20%；风机提速"],
    ["消防信号25%LEL","关闭空调与风阀，风机停机"],
    ["风机故障","切换备用风机并报警"],
    ["压缩机过载","变频机组降至30Hz附近并报警"]],columns=["触发条件","动作"]))
with tabs[5]:
    packlist=["全部电池包"]+sorted(coords.Pack_ID.astype(str).unique().tolist())
    choice=st.selectbox("电池包筛选",packlist)
    v=coords if choice=="全部电池包" else coords[coords.Pack_ID.astype(str)==choice]
    fig=px.scatter_3d(v,x="CAD_X_center_mm_rounded",y="CAD_Y_center_mm_rounded",z="CAD_Z_center_mm_rounded",color="Pack_ID",symbol="Parallel_Branch",hover_name="Cell_ID",hover_data=["Series_Group","Parallel_Branch"])
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=620,template="plotly_dark",scene=dict(aspectmode="data"))
    st.plotly_chart(fig,use_container_width=True)

st.divider()
st.caption("v6.0 · 用户设备参数接入 · 风机P-Q曲线插值 · 风阀阻力 · 空调变频功率 · 设备分项能耗 · 固定控制与预测调控对比")
