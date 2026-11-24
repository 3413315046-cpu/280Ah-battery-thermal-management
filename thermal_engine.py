"""Three-strategy, no-air-conditioner research demonstrator.

Twelve temperature zones / 24 PCM enthalpy nodes; target cabinet has 144 cells.
All scenarios call the SAME plant dynamics; only actions differ.
No demonstrated calibration to the target CAD/CFD. Assumptions documented in README.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

PCM_DENSITY = 1100.0  # kg/m3, user-provided historic fixed demo material
PCM_CP = 1740.0       # J/(kg K)
PCM_LATENT = 120000.0 # J/kg
PHASE_LO, PHASE_HI = 38.0, 43.0
NZ, NPCM, NCELLS = 12, 24, 144
CELL_PER_ZONE = NCELLS // NZ
FAN_DB = {
 'EC轴流500mm': {'flow':4500., 'power':200., 'pressure':350., 'rpm':1800.,
              'pq':[(0,350),(1000,330),(2000,290),(3000,230),(4000,150),(4500,0)]},
 'EC离心355mm': {'flow':1800., 'power':500., 'pressure':1442., 'rpm':2600.,
              'pq':[(0,1442),(400,1350),(800,1200),(1200,1000),(1600,700),(1800,0)]},
}
HEAT_SHAPE = np.array([.89,.93,.97,1.03,1.08,1.12,.90,.96,1.04,1.10,1.14,.84]); HEAT_SHAPE/=HEAT_SHAPE.mean()
AIR_SHAPE = np.array([.72,.83,.90,1.02,1.15,1.28,.75,.86,.95,1.08,1.20,1.26])

@dataclass(frozen=True)
class Settings:
 ambient:float=30.0
 initial:float=30.0
 rate:float=1.0
 duration_h:int=12
 profile:str='完整充放电循环'
 resistance_mohm:float=.25
 cell_mass:float=5.49
 cell_cp:float=1000.
 fans:int=4
 fan_type:str='EC轴流500mm'
 fixed_speed:float=.70
 fixed_valve:float=.75
 target_t:float=38.0
 target_spread:float=3.0
 update_minutes:int=5
 prediction_minutes:int=20
 pcm_volume:float=.04651029

 def validate(self):
  assert 10<=self.ambient<=45 and 15<=self.initial<=55
  assert .1<=self.rate<=2 and 1<=self.duration_h<=48
  assert .05<=self.resistance_mohm<=2 and 1<=self.cell_mass<=15 and 500<=self.cell_cp<=2000
  assert 1<=self.fans<=12 and self.fan_type in FAN_DB
  assert 0<self.fixed_speed<=1 and .25<=self.fixed_valve<=1
  assert self.profile in ('完整充放电循环','高负荷间歇循环','单次放电后静置')
  assert self.update_minutes in (2,5) and self.prediction_minutes in (10,20,30)


def heat_current(t_s, s:Settings):
 """SOC-consistent illustrative periodic load, with charging replenishing Ah.

 At user rate r: discharge I=280*r for 1/r h, recharge I=-140*r for 2/r h;
 an idle period follows. No OCV/electrochem validation; heat source is I^2 R.
 """
 hr=t_s/3600.
 d=1.0/s.rate; ch=2.0/s.rate
 if s.profile=='单次放电后静置':
  return 280.*s.rate if hr<d else 0.
 rest=1.0 if s.profile=='完整充放电循环' else .5
 phase=hr%(d+ch+rest)
 return 280.*s.rate if phase<d else (-140.*s.rate if phase<d+ch else 0.)

def h_specific(T):
 T=np.asarray(T,float)
 return PCM_CP*T+PCM_LATENT*np.clip((T-PHASE_LO)/(PHASE_HI-PHASE_LO),0,1)

def t_from_h(h):
 h=np.asarray(h,float)
 hl=PCM_CP*PHASE_LO
 hh=PCM_CP*PHASE_HI+PCM_LATENT
 return np.where(h<hl,h/PCM_CP,np.where(h<=hh,PHASE_LO+(h-hl)/(PCM_CP+PCM_LATENT/5.),(h-PCM_LATENT)/PCM_CP))

def liquid_ratio(T):
 return np.clip((np.asarray(T)-PHASE_LO)/(PHASE_HI-PHASE_LO),0,1)

def pcm_remaining_MJ(H,mp):
 T=t_from_h(H/mp)
 return float(np.sum(mp*PCM_LATENT*(1-liquid_ratio(T)))/1e6)


def plant_initial(s):
 mp=np.full(NPCM,s.pcm_volume*PCM_DENSITY/NPCM)
 return np.full(NZ,s.initial), mp*h_specific(np.full(NPCM,s.ambient))

def air_distribution(T, fan, valve, mode):
 # fixed valve = uniform fractions; 'balance' directs more flow to warmer zones
 hot=np.maximum(T-T.mean(),0.)
 if mode=='balance' and np.max(hot)>0.01:
  weights=1+.95*hot/(np.max(hot)+.1)
 else:
  weights=np.ones(NZ)
 weights/=weights.mean()
 return (AIR_SHAPE*weights*max(.0,valve)**.65)

def calc_airflow(s, fan, valve):
 # The chosen installed airflow derating (~55-85%) is an explicit uncalibrated
 # duct estimate, NOT a claim of solving the full pressure-flow network.
 unit=FAN_DB[s.fan_type]
 return s.fans*unit['flow']*fan*(.55+.3*np.sqrt(valve))

def plant_step(T,H, s:Settings, t_s, dt, fan, valve, mode):
 """Conservative heat network: battery Q, battery-air, battery-PCM,
 PCM-air. Cells split into 12 zones, each has 2 equivalent PCM regions.
 Cooling never subtracts a constant 'rated cooling power' from cell states.
 """
 assert dt>0
 fan=float(np.clip(fan,0,1)); valve=float(np.clip(valve,.25,1.))
 mass_pcm=np.full(NPCM,s.pcm_volume*PCM_DENSITY/NPCM)
 Tp=t_from_h(H/mass_pcm)
 Tp_avg=Tp.reshape(NZ,2).mean(axis=1)
 Qcell= (heat_current(t_s,s)**2)*(s.resistance_mohm*.001)*CELL_PER_ZONE*HEAT_SHAPE
 C_b=CELL_PER_ZONE*s.cell_mass*s.cell_cp
 # Natural air + fan air, with mild spatial variation based on air path location
 g_air=(.7+17.0*fan**.75)*air_distribution(T,fan,valve,mode)
 q_to_air=g_air*(T-s.ambient)
 # Two PCM regions for each zone. Direct contact effective G deliberately tunable later.
 g_bp=5.5
 q_to_pcm=g_bp*(T.repeat(2)-Tp)
 # Adjacent zone mixing conserves heat (no implicit sink)
 g_neigh=9.0
 mix=np.zeros(NZ)
 dT_pair=T[:-1]-T[1:]
 mix[:-1]-=g_neigh*dT_pair
 mix[1:]+=g_neigh*dT_pair
 dT=(Qcell-q_to_air-q_to_pcm.reshape(NZ,2).sum(axis=1)+mix)/C_b
 # PCM-to-air: convects toward ambient, bidirectional
 g_pcm_air=(.35+1.75*fan**.75)*np.repeat(AIR_SHAPE,2)
 dH=q_to_pcm-g_pcm_air*(Tp-s.ambient)
 Tnext=T+dt*dT; Hnext=H+dt*dH
 if not(np.all(np.isfinite(Tnext)) and np.all(np.isfinite(Hnext))):
  raise FloatingPointError('Thermal state became non-finite')
 return Tnext,Hnext


def metrics_state(T,H,s):
 mass_pcm=np.full(NPCM,s.pcm_volume*PCM_DENSITY/NPCM)
 Tp=t_from_h(H/mass_pcm)
 return float(T.max()),float(T.max()-T.min()),float(T.mean()),float(liquid_ratio(Tp).mean()),pcm_remaining_MJ(H,mass_pcm),float(Tp.mean())

def fan_power(s,fan):
 return s.fans*FAN_DB[s.fan_type]['power']*fan**3


def predict_objective(T,H,s,t0,fan,valve,mode,prevfan):
 """Candidate screening by 10-30 min forward thermal prediction. No future
 real temperature is used. Larger penalty for hot spots/uniformity and stored
 PCM depletion, with separate fan-power/switching costs."""
 dtt=300
 t=T.copy(); h=H.copy()
 maxT,maxSpread=0.,0.
 total_kwh=0.
 for i in range(s.prediction_minutes//5):
  ti=t0+i*dtt
  t,h=plant_step(t,h,s,ti,dtt,fan,valve,mode)
  tm,sp,*_=metrics_state(t,h,s)
  maxT=max(maxT,tm);maxSpread=max(maxSpread,sp)
  total_kwh+=fan_power(s,fan)*dtt/3.6e6
 _,_,tmean,liq,remaining,_=metrics_state(t,h,s)
 return (35.*max(0,maxT-s.target_t)**2 + 10.*max(0,maxSpread-s.target_spread)**2
         +.4*max(0,tmean-(s.target_t-1))**2 + 1.8*max(0,liq-.75)**2
         + 55.*total_kwh +.25*abs(fan-prevfan)+.02*fan)

def mpc_action(T,H,s,t0,prevfan):
 # The optional 'balance' is a thermal-zone flow-allocation policy;
 # it requires matching damper geometry/actuators for real hardware.
 if float(np.max(T))<s.target_t-2.5 and float(np.max(T)-np.min(T))<s.target_spread-1:
  # Still inspect forecast to allow pre-emptive activation (no strict startup gate)
  j_off=predict_objective(T,H,s,t0,0.,s.fixed_valve,'uniform',prevfan)
  if j_off<.45: return 0.,s.fixed_valve,'uniform'
 candidates=[]
 for fan in (0.,.30,.50,.70,1.):
  for valve in (.50,.75,1.):
   for mode in ('uniform','balance') if fan>0 else ('uniform',):
    if fan==0 and valve!=.75:continue
    j=predict_objective(T,H,s,t0,fan,valve,mode,prevfan)
    candidates.append((j,fan,valve,mode))
 _,f,v,m=min(candidates,key=lambda x:x[0])
 return f,v,m

def run_one(s:Settings,strategy,dt=60):
 s.validate()
 assert strategy in ('固定送风','固定阈值','双预测MPC')
 nsteps=int(round(s.duration_h*3600/dt))
 times=np.arange(nsteps+1)*dt
 T,H=plant_initial(s)
 fan=0.;valve=s.fixed_valve; mode='uniform'
 prevfan=0.;prev_valve=s.fixed_valve;prev_mode="uniform";on=False
 records=[]
 energy=0.;fan_time=0.; valve_energy=0.; changes=0;high_time=0.
 select_every=int(s.update_minutes*60/dt)
 for k,ti in enumerate(times):
  if k<nsteps and k%select_every==0:
   if strategy=='固定送风': fan,valve,mode=s.fixed_speed,s.fixed_valve,'uniform'
   elif strategy=='固定阈值':
    # Hysteresis against rapid switching. No model prediction.
    hottest=float(T.max())
    if on:
     if hottest<34.0:on=False
    elif hottest>=35.0:on=True
    fan=(1. if hottest>39.0 else .60) if on else 0.
    valve=s.fixed_valve;mode='uniform'
   else:
    fan,valve,mode=mpc_action(T,H,s,ti,prevfan)
   if abs(fan-prevfan)>.01:changes+=1
   # Electric damper actuator: 5W for 15s per change (single command),
   # and a provisional 12-zone actuator bank when balanced distribution is requested.
   damper_moves=(1 if abs(valve-prev_valve)>.01 else 0)+(12 if mode!=prev_mode else 0)
   event_kwh=damper_moves*5.0*15.0/3.6e6
   energy+=event_kwh;valve_energy+=event_kwh
   prev_valve,prev_mode=valve,mode
   prevfan=fan
  tmax,spread,tmean,liquid,remaining,Tpcm=metrics_state(T,H,s)
  records.append((ti/3600,tmax,spread,tmean,Tpcm,liquid,remaining,fan,fan_power(s,fan),energy,fan_time,
                 calc_airflow(s,fan,valve),valve*100,heat_current(ti,s),1 if mode=='balance' else 0,valve_energy))
  if k==nsteps:break
  T,H=plant_step(T,H,s,ti,dt,fan,valve,mode)
  if fan>0.02:fan_time+=dt/3600
  energy+=fan_power(s,fan)*dt/3.6e6
  # Actuator consumes 5W for 15s only when valve setting changes; NOT continuously.
  # Valve action count is not tracked for independent zones pending hardware model.
  if tmax>s.target_t:high_time+=dt/3600
 df=pd.DataFrame(records,columns=['time_h','Tmax_C','dTmax_C','Tmean_C','PCM_T_C','PCM_liquid','PCM_remaining_MJ',
                                   'fan_speed','fan_power_W','energy_kWh','fan_runtime_h','airflow_m3h','valve_pct','current_A','flow_balanced','valve_energy_kWh'])
 df['strategy']=strategy
 df.attrs['max_high_time_h']=high_time
 df.attrs['fan_commands']=changes
 df.attrs['final_zone_T']=T.copy()
 df.attrs['final_pcm_H']=H.copy()
 return df

def run_all(s:Settings,dt=60):
 return {name:run_one(s,name,dt=dt) for name in ('固定送风','固定阈值','双预测MPC')}

def summarize(df,s):
 return {'最高温度 Tmax（℃）':round(float(df.Tmax_C.max()),2),
         '最大温差 ΔTmax（℃）':round(float(df.dTmax_C.max()),2),
         '平均温度（℃）':round(float(df.Tmean_C.mean()),2),
         '超温累计时间（h）':round(float(df.attrs['max_high_time_h']),2),
         'PCM末端液相率（%）':round(float(df.PCM_liquid.iloc[-1]*100),1),
         'PCM剩余潜热（MJ）':round(float(df.PCM_remaining_MJ.iloc[-1]),2),
         '风机运行时长（h）':round(float(df.fan_runtime_h.iloc[-1]),2),
         '风机累计能耗（kWh）':round(float(df.energy_kWh.iloc[-1]-df.valve_energy_kWh.iloc[-1]),3),
         '风阀执行器能耗（kWh）':round(float(df.valve_energy_kWh.iloc[-1]),4),
         '总设备能耗（kWh）':round(float(df.energy_kWh.iloc[-1]),3),
         '转速变更次数':int(df.attrs['fan_commands'])}
