"""
280Ah储能电池舱热管理仿真引擎 v2
三组同工况分区等效热模型。结果为模型仿真，不是实测或CFD验证数据。
PCM采用焓法跟踪显热与潜热，液相率和剩余潜热储能由同一状态量计算。
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

FAN_DB = {
    "EC轴流500mm": {
        "rated_power_W": 200.0, "rated_airflow_m3h": 4500.0,
        "rated_pressure_Pa": 350.0, "rated_rpm": 1800.0,
        "voltage": "48VDC", "control": "0-10V / PWM",
        "pq": [(0,350),(1000,330),(2000,290),(3000,230),(4000,150),(4500,0)]
    },
    "EC离心355mm": {
        "rated_power_W": 500.0, "rated_airflow_m3h": 1800.0,
        "rated_pressure_Pa": 1442.0, "rated_rpm": 2600.0,
        "voltage": "230VAC", "control": "0-10V / RS485",
        "pq": [(0,1442),(400,1350),(800,1200),(1200,1000),(1600,700),(1800,0)]
    }
}
AC_DB = {
    "壁挂式7.5kW": {
        "cooling_capacity_kW": 7.5, "rated_power_kW": 3.2,
        "airflow_m3h": 2000.0, "eer": 2.34, "voltage": "380V/50Hz"
    },
    "顶装式20kW": {
        "cooling_capacity_kW": 20.0, "rated_power_kW": 9.1,
        "airflow_m3h": 6000.0, "eer": 2.20, "voltage": "380V/50Hz",
        "frequency_capacity_curve": [(30,12),(40,15),(50,20),(60,22)]
    }
}

@dataclass(frozen=True)
class Settings:
    ambient: float = 30.0
    initial: float = 30.0
    rate: float = 1.0
    duration_h: int = 48
    profile: str = "完整充放电循环"
    resistance_mohm: float = 0.25
    cell_mass: float = 5.49
    cell_cp: float = 1000.0
    fans: int = 4
    fan_type: str = "EC轴流500mm"
    fixed_speed: float = 0.70
    fixed_valve: float = 0.75
    target_t: float = 38.0
    target_spread: float = 3.0
    update_minutes: int = 5
    prediction_minutes: int = 20
    pcm_volume: float = 0.04651029
    ac_type: str = "壁挂式7.5kW"
    ac_setpoint: float = 27.0
    fixed_ac_fraction: float = 0.55
    cells: int = 280
    pack_heat_capacity_kJ_K: float = 4200.0
    heat_gain_kW_at_1C: float = 5.0
    passive_loss_kW_K: float = 0.22
    pcm_density: float = 1100.0
    pcm_cp: float = 1740.0
    pcm_k: float = 3.5
    pcm_latent_kJ_kg: float = 120.0
    pcm_t_start: float = 38.0
    pcm_t_end: float = 43.0

def _clip(x, lo=0.0, hi=1.0):
    return float(np.clip(x, lo, hi))

def _fan_spec(s):
    return FAN_DB.get(s.fan_type, FAN_DB["EC轴流500mm"])

def _ac_spec(s):
    return AC_DB.get(s.ac_type, AC_DB["壁挂式7.5kW"])

def _heat_load(s, hour):
    phase = hour % 4.0
    if s.profile == "恒定放电":
        factor = 1.0
    elif s.profile == "间歇运行":
        factor = 0.20 if phase < 1.0 else (0.85 if phase < 3.0 else 0.15)
    else:
        factor = 1.0 if phase < 2.0 else (0.58 if phase < 3.0 else 0.18)
    return max(0.0, s.heat_gain_kW_at_1C * max(0.1, s.rate)**1.65 * factor)

def _pcm_state(s, enthalpy_kwh, mass):
    """焓法状态映射：焓(kWh) -> PCM温度、液相率、剩余潜热(MJ)。"""
    cp_kwh_k = max(1e-9, mass*s.pcm_cp/3.6e6)
    latent_kwh = max(1e-9, mass*s.pcm_latent_kJ_kg/3600.0)
    h_start = cp_kwh_k*(s.pcm_t_start-s.ambient)
    h_end_solid = h_start
    h_end_melt = h_start+latent_kwh
    if enthalpy_kwh < h_end_solid:
        temp = s.ambient+enthalpy_kwh/cp_kwh_k
        liquid = 0.0
    elif enthalpy_kwh <= h_end_melt:
        temp = s.pcm_t_start
        liquid = _clip((enthalpy_kwh-h_end_solid)/latent_kwh)
    else:
        temp = s.pcm_t_start+(enthalpy_kwh-h_end_melt)/cp_kwh_k
        liquid = 1.0
    remaining_mj = latent_kwh*(1.0-liquid)*3.6
    return float(temp), float(liquid), float(remaining_mj), latent_kwh

def _fan_power(s, speed):
    return s.fans*_fan_spec(s)["rated_power_W"]*_clip(speed)**3/1000.0

def _ac_power(s, fraction):
    return _ac_spec(s)["rated_power_kW"]*_clip(fraction)**1.5

def _control(mode, s, tmax, spread, pcm_liquid, slope, old_fan, old_ac,
             zone=None, hour=0.0, pcm_temp=None, dt_seconds=300):
    """
    同硬件公平对照：
    固定送风/固定阈值保持基准策略；双预测MPC在候选风机-风阀-空调组合中
    做短时域滚动预测，优先选取满足温度与温差约束的最低瞬时功率组合。
    """
    if mode == "固定送风":
        fan = _clip(s.fixed_speed)
        ac = 0.0 if tmax < 40.0 else _clip(s.fixed_ac_fraction)
        valve = _clip(s.fixed_valve)
    elif mode == "固定阈值":
        fan = 0.0 if tmax < 35.0 else (0.58 if tmax < 38.0 else 0.88)
        ac = 0.0 if tmax < 39.0 else (0.32 if tmax < 41.0 else 0.62)
        valve = 0.68 if spread < s.target_spread else 0.95
    else:
        z = np.asarray(zone if zone is not None else np.full(8, tmax), dtype=float)
        mean_t = float(z.mean())
        hot_idx = np.argsort(z)[-2:]
        mass = max(0.0, s.pcm_volume*s.pcm_density)
        pcm_temp = float(pcm_temp if pcm_temp is not None else s.ambient)
        load = _heat_load(s, hour + max(1, s.prediction_minutes)/120.0)
        heat_pattern = np.array([0.94,0.98,1.02,1.07,1.05,1.00,0.97,0.97])
        cooling_pattern = np.array([1.10,1.06,1.02,0.96,0.92,0.94,0.99,1.01])
        cap_kwh_k = max(0.5, s.pack_heat_capacity_kJ_K/3600.0)
        airflow_full = _fan_spec(s)["rated_airflow_m3h"]*max(1,s.fans)
        ac_capacity = _ac_spec(s)["cooling_capacity_kW"]
        horizon_h = max(10.0, float(s.prediction_minutes)*60.0)/3600.0
        passive = s.passive_loss_kW_K*(mean_t-s.ambient)
        q_pcm = float(np.clip(0.10*(float(z[hot_idx].mean())-pcm_temp),-0.10,0.22))
        if pcm_liquid >= 0.999 and q_pcm > 0: q_pcm *= 0.10
        if pcm_liquid <= 0.001 and q_pcm < 0: q_pcm *= 0.10

        best = None
        # Discrete candidate set makes the MPC auditable and prevents arbitrary AC bias.
        fan_candidates = (0.0,0.22,0.32,0.42,0.52,0.62,0.72,0.82,0.92)
        ac_candidates = (0.0,0.04,0.08,0.12,0.18,0.25,0.35,0.50)
        valve_candidates = (0.60,0.75,0.90,1.00)
        dev = z-z.mean()
        allocation = np.clip(1.0+0.11*dev,0.78,1.22)

        for f in fan_candidates:
            fan_kw = _fan_power(s,f)
            airflow = airflow_full*f
            fan_cooling = min(airflow*0.00012*max(0.0,mean_t-s.ambient+1.0),
                              1.00*load)
            for acv in ac_candidates:
                ac_kw = _ac_power(s,acv)
                ac_cooling = ac_capacity*acv*0.82
                for v in valve_candidates:
                    cooling = (fan_cooling+ac_cooling)*cooling_pattern*allocation*v
                    net = load*heat_pattern-cooling-passive/8.0
                    weights = np.zeros(8); weights[hot_idx]=0.5
                    net -= q_pcm*weights
                    net += 0.18*(z.mean()-z)
                    pred_zone = np.clip(z+net*horizon_h/cap_kwh_k,5.0,65.0)
                    pred_max = float(pred_zone.max())
                    pred_spread = float(pred_zone.max()-pred_zone.min())

                    # Hard-ish thermal constraints, with a strong quadratic penalty
                    # when the current physical conditions make them infeasible.
                    t_violation = max(0.0,pred_max-(s.target_t-0.4))
                    dt_violation = max(0.0,pred_spread-s.target_spread)
                    power = fan_kw+ac_kw
                    switching = 0.025*abs(f-old_fan)+0.035*abs(acv-old_ac)
                    objective = (power + switching
                                 + 18.0*t_violation**2
                                 + 12.0*dt_violation**2)
                    # Prefer fan/valve distribution over AC when costs are comparable.
                    objective += 0.012*acv
                    if best is None or objective < best[0]:
                        best=(objective,f,acv,v,pred_max,pred_spread)

        _,fan,ac,valve,_,_=best
        # Limit abrupt actuator changes without forcing unnecessary AC run-on.
        fan=float(np.clip(fan,old_fan-0.25,old_fan+0.25))
        ac=float(np.clip(ac,old_ac-0.10,old_ac+0.10))
    return _clip(fan), _clip(ac), _clip(valve)

def run_one(s, mode, dt=300):
    dt = max(10, int(dt))
    hours = float(np.clip(s.duration_h, 0.1, 48.0))
    total_seconds = hours * 3600.0
    n_steps = int(np.ceil(total_seconds / dt - 1e-12))
    zone = np.full(8, float(s.initial))
    mass = max(0.0, s.pcm_volume*s.pcm_density)
    pcm_h = max(0.0, (s.initial-s.ambient)*mass*s.pcm_cp/3.6e6)
    fan_old = ac_old = 0.0
    valve_old = float(s.fixed_valve)
    fan_e = ac_e = valve_e = aux_e = 0.0
    fan_h = ac_h = over_h = spread_over_h = 0.0
    prev_tmax = float(zone.max())
    rows = []

    fan_spec, ac_spec = _fan_spec(s), _ac_spec(s)
    airflow_full = fan_spec["rated_airflow_m3h"]*max(1, s.fans)
    ac_capacity = ac_spec["cooling_capacity_kW"]
    heat_pattern = np.array([0.94,0.98,1.02,1.07,1.05,1.00,0.97,0.97])
    cooling_pattern = np.array([1.10,1.06,1.02,0.96,0.92,0.94,0.99,1.01])
    cap_kwh_k = max(0.5, s.pack_heat_capacity_kJ_K/3600.0)
    pcm_cp_kwh_k = max(1e-9, mass*s.pcm_cp/3.6e6)

    for step_index in range(n_steps):
        elapsed_seconds = step_index * dt
        step_seconds = min(float(dt), total_seconds - elapsed_seconds)
        hour = elapsed_seconds / 3600.0
        tmax, tmin = float(zone.max()), float(zone.min())
        spread, mean_t = tmax-tmin, float(zone.mean())
        slope = (tmax-prev_tmax)/max(step_seconds/3600.0, 1e-9)
        pcm_temp, liquid, pcm_remain, latent_kwh = _pcm_state(s, pcm_h, mass)
        fan, ac, valve = _control(mode,s,tmax,spread,liquid,slope,fan_old,ac_old,
                                 zone=zone,hour=hour,pcm_temp=pcm_temp,dt_seconds=step_seconds)

        fan_kw = _fan_power(s,fan)
        ac_kw = _ac_power(s,ac)
        valve_kw = 0.004 if abs(valve-valve_old)>0.025 else 0.0
        aux_kw = 0.018
        airflow = airflow_full*fan*valve
        load = _heat_load(s,hour)
        passive = s.passive_loss_kW_K*(mean_t-s.ambient)

        # 风机强制对流与空调制冷量共同计入热平衡。
        fan_cooling = min(airflow*0.00012*max(0.0,mean_t-s.ambient+1.0),
                          1.00*load)
        ac_cooling = ac_capacity*ac*0.82

        # PCM与电池区之间的有限热耦合。正值表示PCM吸热；负值表示向电池回热。
        pcm_temp, liquid, pcm_remain, latent_kwh = _pcm_state(s,pcm_h,mass)
        # PCM贴近高温电池区域布置：用高温区代表温度估算界面换热，而非舱内平均温度。
        hot_idx = np.argsort(zone)[-2:]
        hot_face_t = float(zone[hot_idx].mean())
        q_pcm = float(np.clip(0.10*(hot_face_t-pcm_temp),-0.10,0.22))
        if liquid >= 0.999 and q_pcm > 0:
            q_pcm *= 0.10
        if liquid <= 0.001 and q_pcm < 0:
            q_pcm *= 0.10

        local_cooling = (fan_cooling+ac_cooling)*cooling_pattern
        local_load = load*heat_pattern
        if mode == "双预测MPC":
            # 分区预测调风：高于均温的区域获得更多冷量，低温区域减少送风。
            dev = zone-zone.mean()
            allocation = np.clip(1.0+0.11*dev,0.78,1.22)
            local_cooling = local_cooling*allocation
        net = local_load-local_cooling-passive/8.0
        # PCM与电池之间的热交换优先作用于接触温度较高的两个区域。
        pcm_weights = np.zeros(8)
        pcm_weights[hot_idx] = 0.5
        net -= q_pcm*pcm_weights
        # 相邻区域热交换，保留热不均匀，同时避免温差数值发散。
        net += 0.18*(zone.mean()-zone)
        zone += net*(step_seconds/3600.0)/cap_kwh_k
        zone = np.clip(zone,5.0,65.0)

        # 更新PCM焓：显热段、相变潜热段、完全熔化后的显热段连续衔接。
        pcm_h += q_pcm*(step_seconds/3600.0)
        pcm_temp, liquid, pcm_remain, latent_kwh = _pcm_state(s,pcm_h,mass)

        fan_e += fan_kw*step_seconds/3600.0
        ac_e += ac_kw*step_seconds/3600.0
        valve_e += valve_kw*step_seconds/3600.0
        aux_e += aux_kw*step_seconds/3600.0
        if fan>0.02: fan_h += step_seconds/3600.0
        if ac>0.02: ac_h += step_seconds/3600.0
        if tmax>s.target_t: over_h += step_seconds/3600.0
        if spread>s.target_spread: spread_over_h += step_seconds/3600.0
        total_e=fan_e+ac_e+valve_e+aux_e

        zmax,zmin=float(zone.max()),float(zone.min())
        row={
            "time_h":(elapsed_seconds+step_seconds)/3600.0,"Tmax_C":zmax,"Tmin_C":zmin,
            "dTmax_C":zmax-zmin,"Tmean_C":float(zone.mean()),
            "Tair_mean_C":float(s.ambient),"PCM_T_C":pcm_temp,
            "PCM_liquid":liquid,"PCM_remaining_MJ":pcm_remain,
            "fan_speed":fan,"fan_power_W":fan_kw*1000,
            "ac_fraction":ac,"ac_power_W":ac_kw*1000,
            "valve_pct":valve*100,"fan_pct":fan*100,"ac_pct":ac*100,
            "fan_energy_kWh":fan_e,"ac_energy_kWh":ac_e,
            "valve_energy_kWh":valve_e,"aux_energy_kWh":aux_e,
            "total_energy_kWh":total_e,"fan_runtime_h":fan_h,"ac_runtime_h":ac_h,
            "airflow_m3h":airflow,"heat_current_kW":load,
            "over_target_h":over_h,"over_spread_h":spread_over_h,
            "fan_energy_kwh":fan_e,"ac_energy_kwh":ac_e,
            "valve_energy_kwh":valve_e,"aux_energy_kwh":aux_e,
            "total_energy_kwh":total_e,"fan_speed_pct":fan*100,
            "ac_load_pct":ac*100,"pcm_liquid_pct":liquid*100,
            "pcm_remaining_mj":pcm_remain,"Tmax":zmax,"Tmean":float(zone.mean()),
            "dTmax":zmax-zmin,"valve_opening_pct":valve*100
        }
        rows.append(row)
        prev_tmax=tmax
        fan_old,ac_old,valve_old=fan,ac,valve
    return pd.DataFrame(rows)

def run_all(s,dt=300):
    return {name:run_one(s,name,dt) for name in ("固定送风","固定阈值","双预测MPC")}

def summarize(df,s=None):
    if df is None or df.empty:return {}
    last=df.iloc[-1]
    return {
        "电池最高温度（℃）":round(float(df["Tmax_C"].max()),2),
        "最大温差（℃）":round(float(df["dTmax_C"].max()),2),
        "平均电池温度（℃）":round(float(df["Tmean_C"].mean()),2),
        "超温累计时间（h）":round(float(last["over_target_h"]),2),
        "温差超限累计时间（h）":round(float(last["over_spread_h"]),2),
        "风机累计能耗（kWh）":round(float(last["fan_energy_kWh"]),3),
        "空调累计能耗（kWh）":round(float(last["ac_energy_kWh"]),3),
        "风阀及辅助能耗（kWh）":round(float(last["valve_energy_kWh"]+last["aux_energy_kWh"]),4),
        "系统总能耗（kWh）":round(float(last["total_energy_kWh"]),3),
        "风机运行时间（h）":round(float(last["fan_runtime_h"]),2),
        "空调运行时间（h）":round(float(last["ac_runtime_h"]),2),
        "PCM末端液相率（%）":round(float(last["PCM_liquid"])*100,1),
        "PCM剩余储热能力（MJ）":round(float(last["PCM_remaining_MJ"]),2)
    }

__all__=["Settings","run_one","run_all","summarize","FAN_DB","AC_DB"]
