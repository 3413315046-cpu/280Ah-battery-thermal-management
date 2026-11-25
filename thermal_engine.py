"""Three-strategy coupled battery/PCM/air/HVAC thermal demonstrator.
All strategies share the same plant model; only control policies differ.
The model is a reduced-order engineering simulation, not validated CFD or field data.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

PCM_DENSITY=1100.0
PCM_CP=1740.0
PCM_LATENT=120000.0
PHASE_LO,PHASE_HI=38.0,43.0
NZ,NPCM,NCELLS=12,24,144
CELL_PER_ZONE=12
FAN_DB={
 'EC轴流500mm':{'flow':4500.,'power':200.,'pressure':350.,'rpm':1800.,'pq':[(0,350),(1000,330),(2000,290),(3000,230),(4000,150),(4500,0)]},
 'EC离心355mm':{'flow':1800.,'power':500.,'pressure':1442.,'rpm':2600.,'pq':[(0,1442),(400,1350),(800,1200),(1200,1000),(1600,700),(1800,0)]},
}
AC_DB={
 '壁挂式7.5kW':{'cooling_kw':7.5,'rated_input_kw':3.2,'min_hz':30.,'rated_hz':50.,'max_hz':60.},
 '顶装式20kW':{'cooling_kw':20.,'rated_input_kw':9.1,'min_hz':30.,'rated_hz':50.,'max_hz':60.},
}
HEAT_SHAPE=np.array([.89,.93,.97,1.03,1.08,1.12,.90,.96,1.04,1.10,1.14,.84]);HEAT_SHAPE/=HEAT_SHAPE.mean()
AIR_SHAPE=np.array([.72,.83,.90,1.02,1.15,1.28,.75,.86,.95,1.08,1.20,1.26])

@dataclass(frozen=True)
class Settings:
 ambient:float=30.
 initial:float=30.
 rate:float=1.
 duration_h:int=48
 profile:str='完整充放电循环'
 resistance_mohm:float=.25
 cell_mass:float=5.49
 cell_cp:float=1000.
 fans:int=4
 fan_type:str='EC轴流500mm'
 fixed_speed:float=.70
 fixed_valve:float=.75
 target_t:float=38.
 target_spread:float=3.
 update_minutes:int=5
 prediction_minutes:int=20
 pcm_volume:float=.04651029
 ac_type:str='壁挂式7.5kW'
 ac_setpoint:float=27.0
 fixed_ac_fraction:float=.55
 def validate(self):
  assert 10<=self.ambient<=45 and 15<=self.initial<=55
  assert .1<=self.rate<=2 and 1<=self.duration_h<=48
  assert .05<=self.resistance_mohm<=2 and 1<=self.cell_mass<=15 and 500<=self.cell_cp<=2000
  assert 1<=self.fans<=12 and self.fan_type in FAN_DB and self.ac_type in AC_DB
  assert 0<self.fixed_speed<=1 and .25<=self.fixed_valve<=1
  assert self.profile in ('完整充放电循环','高负荷间歇循环','单次放电后静置')
  assert self.update_minutes in (2,5) and self.prediction_minutes in (10,20,30)
  assert 18<=self.ac_setpoint<=32 and .1<=self.fixed_ac_fraction<=1

def heat_current(t_s,s):
 hr=t_s/3600.;d=1./s.rate;ch=2./s.rate
 if s.profile=='单次放电后静置':return 280.*s.rate if hr<d else 0.
 rest=1.0 if s.profile=='完整充放电循环' else .5
 phase=hr%(d+ch+rest)
 return 280.*s.rate if phase<d else (-140.*s.rate if phase<d+ch else 0.)

def h_specific(T):
 T=np.asarray(T,float)
 return PCM_CP*T+PCM_LATENT*np.clip((T-PHASE_LO)/(PHASE_HI-PHASE_LO),0,1)
def t_from_h(h):
 h=np.asarray(h,float);hl=PCM_CP*PHASE_LO;hh=PCM_CP*PHASE_HI+PCM_LATENT
 return np.where(h<hl,h/PCM_CP,np.where(h<=hh,PHASE_LO+(h-hl)/(PCM_CP+PCM_LATENT/5.),(h-PCM_LATENT)/PCM_CP))
def liquid_ratio(T):return np.clip((np.asarray(T)-PHASE_LO)/(PHASE_HI-PHASE_LO),0,1)
def pcm_remaining_MJ(H,mp):
 T=t_from_h(H/mp)
 return float(np.sum(mp*PCM_LATENT*(1-liquid_ratio(T)))/1e6)

def plant_initial(s):
 mp=np.full(NPCM,s.pcm_volume*PCM_DENSITY/NPCM)
 return np.full(NZ,s.initial),mp*h_specific(np.full(NPCM,s.ambient)),np.full(NZ,s.ambient)

def air_distribution(T,valve,mode):
 hot=np.maximum(T-T.mean(),0.)
 weights=1+.85*hot/(np.max(hot)+.1) if mode=='balance' else np.ones(NZ)
 weights/=weights.mean()
 return AIR_SHAPE*weights*max(.0,valve)**.65

def calc_airflow(s,fan,valve):
 return s.fans*FAN_DB[s.fan_type]['flow']*fan*(.55+.3*np.sqrt(valve))

def fan_power(s,fan):return s.fans*FAN_DB[s.fan_type]['power']*fan**3
def ac_capacity_kw(s,fraction):
 db=AC_DB[s.ac_type]
 if fraction<=0:return 0.
 # fraction maps compressor frequency 30-60Hz; capacity curve is interpolated from supplied 20kW unit.
 hz=db['min_hz']+fraction*(db['max_hz']-db['min_hz'])
 if s.ac_type=='顶装式20kW':
  cap=np.interp(hz,[30,40,50,60],[12,15,20,22])
  cap=min(cap,db['cooling_kw']*1.10)
 else:
  cap=db['cooling_kw']*fraction
 return cap
def ac_power_kw(s,fraction):
 if fraction<=0:return 0.
 db=AC_DB[s.ac_type]
 # variable compressor input; rated input corresponds nominal 50Hz
 hz=db['min_hz']+fraction*(db['max_hz']-db['min_hz'])
 return db['rated_input_kw']*(hz/db['rated_hz'])**1.5

def plant_step(T,H,Ta,s,t_s,dt,fan,valve,mode,ac):
 fan=float(np.clip(fan,0,1));valve=float(np.clip(valve,.25,1));ac=float(np.clip(ac,0,1))
 mp=np.full(NPCM,s.pcm_volume*PCM_DENSITY/NPCM);Tp=t_from_h(H/mp);Tp_avg=Tp.reshape(NZ,2).mean(axis=1)
 I=heat_current(t_s,s)
 Qcell=(I**2)*(s.resistance_mohm*.001)*CELL_PER_ZONE*HEAT_SHAPE
 Cb=CELL_PER_ZONE*s.cell_mass*s.cell_cp
 # Battery-to-air exchange; fan increases convective conductance.
 ga=(.8+15.5*fan**.78)*air_distribution(T,valve,mode)
 q_ba=ga*(T-Ta)
 # Battery-to-PCM direct conduction.
 q_bp=5.5*(T.repeat(2)-Tp)
 # Neighbour-zone conduction is conservative.
 mix=np.zeros(NZ);edge=T[:-1]-T[1:];mix[:-1]-=9.*edge;mix[1:]+=9.*edge
 dT=(Qcell-q_ba-q_bp.reshape(NZ,2).sum(axis=1)+mix)/Cb
 # Air-node energy balance: battery + PCM heat, outdoor exchange, HVAC cooling.
 q_pa=(.35+1.5*fan**.75)*np.repeat(AIR_SHAPE,2)*(Tp-s.ambient)
 q_pcm_air=q_pa.reshape(NZ,2).sum(axis=1)
 g_vent=(3.0+17.0*fan**.8)*air_distribution(T,valve,mode)
 q_vent=g_vent*(Ta-s.ambient)
 q_ac=ac_capacity_kw(s,ac)*1000./NZ
 Cair=18000. # J/K per reduced air zone, explicit reduced-order assumption
 dTa=(q_ba+q_pcm_air-q_vent-q_ac)/Cair
 dH=q_bp-q_pa
 Tn=T+dt*dT;Hn=H+dt*dH;Tan=Ta+dt*dTa
 if not(np.all(np.isfinite(Tn)) and np.all(np.isfinite(Hn)) and np.all(np.isfinite(Tan))):
  raise FloatingPointError('Non-finite thermal state')
 return Tn,Hn,Tan

def metrics_state(T,H,s):
 mp=np.full(NPCM,s.pcm_volume*PCM_DENSITY/NPCM);Tp=t_from_h(H/mp)
 return float(T.max()),float(T.max()-T.min()),float(T.mean()),float(liquid_ratio(Tp).mean()),pcm_remaining_MJ(H,mp),float(Tp.mean())

def predict_score(T,H,Ta,s,t0,fan,valve,mode,ac,prevfan,prevac):
 dt=300;t=T.copy();h=H.copy();ta=Ta.copy();mx=spmx=0.;energy=0.
 for i in range(s.prediction_minutes//5):
  t,h,ta=plant_step(t,h,ta,s,t0+i*dt,dt,fan,valve,mode,ac)
  tm,sp,*_=metrics_state(t,h,s);mx=max(mx,tm);spmx=max(spmx,sp)
  energy+=(fan_power(s,fan)+ac_power_kw(s,ac)*1000)*dt/3.6e6
 # Penalize peak temperature, spread, overcooling, energy and control chattering.
 return (90*max(0,mx-(s.target_t-0.8))**2+
         55*max(0,spmx-s.target_spread)**2+
         3.0*max(0,(s.target_t-4)-float(t.mean()))**2+
         18.0*max(0,float(t.mean())-(s.target_t-2.2))**2+
         100*energy+1.5*abs(fan-prevfan)+2.0*abs(ac-prevac)+.02*fan+.08*ac)

def mpc_action(T,H,Ta,s,t0,prevfan,prevac):
 candidates=[]
 for fan in (0.,.35,.60,.85,1.):
  for ac in (0.,.35,.65,1.):
   for valve in (.55,.75,1.):
    for mode in (('uniform','balance') if fan>.01 else ('uniform',)):
     score=predict_score(T,H,Ta,s,t0,fan,valve,mode,ac,prevfan,prevac)
     candidates.append((score,fan,valve,mode,ac))
 return min(candidates,key=lambda z:z[0])[1:]

def run_one(s,strategy,dt=60):
 s.validate();n=int(round(s.duration_h*3600/dt));times=np.arange(n+1)*dt
 T,H,Ta=plant_initial(s);fan=0.;ac=0.;valve=s.fixed_valve;mode='uniform'
 prevfan=0.;prevac=0.;prevvalve=valve;prevmode=mode;ac_on=False;fan_on=False
 rec=[];ef=ea=ev=0.;runtime=acruntime=high=0.;temp_over3=0.;fan_cmd=ac_cmd=0
 step_select=int(s.update_minutes*60/dt)
 for k,ti in enumerate(times):
  if k<n and k%step_select==0:
   hottest=float(T.max());spread=float(T.max()-T.min());airhot=float(Ta.mean())
   if strategy=='固定送风':
    fan=s.fixed_speed;ac=s.fixed_ac_fraction if airhot>s.ac_setpoint else 0.;valve=s.fixed_valve;mode='uniform'
   elif strategy=='固定阈值':
    if fan_on:
     if hottest<34.5:fan_on=False
    elif hottest>=35.5:fan_on=True
    fan=(.65 if hottest<39 else 1.) if fan_on else 0.
    if ac_on:
     if hottest<34.5:ac_on=False
    elif hottest>=37.0:ac_on=True
    ac=1.0 if ac_on and hottest>=40 else (.65 if ac_on else 0.)
    valve=.75 if hottest<38 else 1.0;mode='uniform'
   else:
    fan,valve,mode,ac=mpc_action(T,H,Ta,s,ti,prevfan,prevac)
   if abs(fan-prevfan)>.01:fan_cmd+=1
   if abs(ac-prevac)>.01:ac_cmd+=1
   damper_changes=(abs(valve-prevvalve)>.01)+(12 if mode!=prevmode else 0)
   e_event=damper_changes*5.*15./3.6e6
   ev+=e_event
   prevvalve,prevmode=valve,mode;prevfan,prevac=fan,ac
  tmax,spread,tmean,liq,remain,tpcm=metrics_state(T,H,s)
  pf=fan_power(s,fan);pa=ac_power_kw(s,ac)*1000.
  ef_now=ef;ea_now=ea;ev_now=ev
  rec.append((ti/3600,tmax,spread,tmean,float(Ta.mean()),tpcm,liq,remain,fan,pf,ac,pa,ef_now,ea_now,ev_now,ef_now+ea_now+ev_now,runtime,acruntime,
              calc_airflow(s,fan,valve),valve*100,heat_current(ti,s),1 if mode=='balance' else 0))
  if k==n:break
  T,H,Ta=plant_step(T,H,Ta,s,ti,dt,fan,valve,mode,ac)
  ef+=pf*dt/3.6e6;ea+=pa*dt/3.6e6
  if fan>.02:runtime+=dt/3600.
  if ac>.02:acruntime+=dt/3600.
  if tmax>s.target_t:high+=dt/3600.
  if spread>s.target_spread:temp_over3+=dt/3600.
 df=pd.DataFrame(rec,columns=['time_h','Tmax_C','dTmax_C','Tmean_C','Tair_mean_C','PCM_T_C','PCM_liquid','PCM_remaining_MJ','fan_speed','fan_power_W','ac_fraction','ac_power_W','fan_energy_kWh','ac_energy_kWh','valve_energy_kWh','total_energy_kWh','fan_runtime_h','ac_runtime_h','airflow_m3h','valve_pct','current_A','flow_balanced'])
 df.attrs.update(final_zone_T=T.copy(),final_pcm_H=H.copy(),final_air_T=Ta.copy(),over_target_h=high,over_spread_h=temp_over3,fan_commands=fan_cmd,ac_commands=ac_cmd)
 return df

def run_all(s,dt=60):return {name:run_one(s,name,dt) for name in ('固定送风','固定阈值','双预测MPC')}
def summarize(df,s):
 return {'电池最高温度（℃）':round(float(df.Tmax_C.max()),2),
 '最大温差（℃）':round(float(df.dTmax_C.max()),2),
 '平均电池温度（℃）':round(float(df.Tmean_C.mean()),2),
 '超温累计时间（h）':round(float(df.attrs['over_target_h']),2),
 '温差超限累计时间（h）':round(float(df.attrs['over_spread_h']),2),
 '风机累计能耗（kWh）':round(float(df.fan_energy_kWh.iloc[-1]),3),
 '空调累计能耗（kWh）':round(float(df.ac_energy_kWh.iloc[-1]),3),
 '风阀及辅助能耗（kWh）':round(float(df.valve_energy_kWh.iloc[-1]),4),
 '系统总能耗（kWh）':round(float(df.total_energy_kWh.iloc[-1]),3),
 '风机运行时间（h）':round(float(df.fan_runtime_h.iloc[-1]),2),
 '空调运行时间（h）':round(float(df.ac_runtime_h.iloc[-1]),2)}
