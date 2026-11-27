"""
280Ah储能电池舱热管理仿真引擎（兼容版）
输出接口：Settings, run_all, summarize, FAN_DB, AC_DB

用途：三种控制策略在相同工况下的分区等效热模型对比。
注意：属于模型仿真，不是实测数据或CFD验证结果；模型参数需用后续实验/仿真标定。
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
    """可重复的充放电循环等效热负荷，单位kW。"""
    phase = hour % 4.0
    if s.profile == "恒定放电":
        factor = 1.0
    elif s.profile == "间歇运行":
        factor = 0.20 if phase < 1.0 else (0.85 if phase < 3.0 else 0.15)
    else:
        factor = 1.0 if phase < 2.0 else (0.58 if phase < 3.0 else 0.18)
    return max(0.0, s.heat_gain_kW_at_1C * max(0.1, s.rate)**1.65 * factor)

def _pcm(s, tpcm):
    mass = max(0.0, s.pcm_volume * s.pcm_density)
    liquid = _clip((tpcm-s.pcm_t_start)/max(0.1, s.pcm_t_end-s.pcm_t_start))
    remain_mj = mass*s.pcm_latent_kJ_kg*(1-liquid)/1000.0
    return liquid, remain_mj, mass

def _fan_power(s, speed):
    return s.fans*_fan_spec(s)["rated_power_W"]*_clip(speed)**3/1000.0

def _ac_power(s, fraction):
    return _ac_spec(s)["rated_power_kW"]*_clip(fraction)**1.5

def _control(mode, s, tmax, spread, pcm_liquid, slope, old_fan, old_ac):
    # 三组均从初始时刻关闭主动设备；按各自策略决定启动时机。
    if mode == "固定送风":
        # 固定送风对照：风机恒定转速、风阀固定开度；空调仅按固定温控点启停。
        fan = _clip(s.fixed_speed)
        ac = 0.0 if tmax < 40.0 else _clip(s.fixed_ac_fraction)
        valve = _clip(s.fixed_valve)
    elif mode == "固定阈值":
        fan = 0.0 if tmax < 35.0 else (0.58 if tmax < 38.0 else 0.88)
        ac = 0.0 if tmax < 39.0 else (0.38 if tmax < 41.0 else 0.72)
        valve = 0.65 if spread < s.target_spread else 0.95
    else:
        horizon = max(1, s.prediction_minutes)/60.0
        # 用当前温升趋势预测短时最高温度；PCM液相率用于估计被动储热余量。
        predicted = tmax + max(-0.05, slope)*horizon
        pcm_reserve = 1.0-pcm_liquid
        thermal_risk = predicted-(s.target_t-2.5)
        spread_risk = max(0.0, spread-(s.target_spread-0.5))
        if predicted < 34.5 and spread < 2.0:
            fan, ac = 0.0, 0.0
        else:
            fan = _clip(0.18 + 0.14*max(0.0, thermal_risk)
                        + 0.18*spread_risk + 0.10*(1.0-pcm_reserve), 0.0, 1.0)
            ac = _clip((predicted-(s.ac_setpoint+6.0))/9.0, 0.0, 0.82)
            if predicted < s.ac_setpoint+5.5:
                ac = 0.0
            if pcm_liquid > 0.82 and predicted > s.target_t-1.0:
                ac = max(ac, 0.18)
        valve = _clip(0.58+0.16*max(0.0, spread-1.0)
                      +0.10*max(0.0, predicted-s.target_t), 0.45, 1.0)
        # MPC控制量平滑，避免不现实的瞬时跳变。
        fan = float(np.clip(fan, old_fan-0.16, old_fan+0.16))
        ac = float(np.clip(ac, old_ac-0.12, old_ac+0.12))
    return _clip(fan), _clip(ac), _clip(valve)

def run_one(s, mode, dt=60):
    dt = max(10, int(dt))
    hours = float(np.clip(s.duration_h, 0.1, 48.0))
    n = int(hours*3600/dt)+1
    t = np.arange(n)*dt/3600.0

    # 八个等效区域；相邻区域通过混合项交换热量，温差不会被人为归零。
    zone = np.full(8, float(s.initial))
    pcm_temp = float(s.initial)
    fan_old = ac_old = 0.0
    valve_old = float(s.fixed_valve)
    fan_e = ac_e = valve_e = aux_e = 0.0
    fan_h = ac_h = over_h = spread_over_h = 0.0
    prev_tmax = float(zone.max())
    rec = []

    fan_spec, ac_spec = _fan_spec(s), _ac_spec(s)
    airflow_full = fan_spec["rated_airflow_m3h"]*max(1, s.fans)
    ac_capacity = ac_spec["cooling_capacity_kW"]
    # 分区热负荷不均匀，加入有限的横向热扩散。
    heat_pattern = np.array([0.94,0.98,1.02,1.07,1.05,1.00,0.97,0.97])
    cooling_pattern = np.array([1.10,1.06,1.02,0.96,0.92,0.94,0.99,1.01])
    cap_kwh_k = max(0.5, s.pack_heat_capacity_kJ_K/3600.0)

    for i, hour in enumerate(t):
        tmax, tmin = float(zone.max()), float(zone.min())
        spread, mean_t = tmax-tmin, float(zone.mean())
        slope = (tmax-prev_tmax)/max(dt/3600.0, 1e-9)
        liquid, pcm_remaining, pcm_mass = _pcm(s, pcm_temp)
        fan, ac, valve = _control(mode, s, tmax, spread, liquid, slope, fan_old, ac_old)

        fan_kw = _fan_power(s, fan)
        ac_kw = _ac_power(s, ac)
        valve_kw = 0.005 if abs(valve-valve_old) > 0.02 else 0.0
        aux_kw = 0.025
        airflow = airflow_full*fan*valve

        # 空气侧换热能力随风量和电池-环境温差变化；空调按额定制冷量折算。
        fan_cooling = min(airflow*0.000045*max(0.0, mean_t-s.ambient+1.0), 0.65*max(0.1, _heat_load(s,hour)))
        ac_cooling = ac_capacity*ac*0.82
        load = _heat_load(s, hour)
        passive = s.passive_loss_kW_K*(mean_t-s.ambient)

        # PCM相变区吸热；潜热储备耗尽后不再继续提供潜热冷却。
        pcm_liquid, _, _ = _pcm(s, pcm_temp)
        pcm_exchange = min(0.55, max(0.0, (mean_t-pcm_temp)*0.10))
        if pcm_liquid >= 0.995:
            pcm_exchange *= 0.05

        # 温差超限时，分区送风不均衡修正增强；这是风阀分区调节的等效效果。
        local_cooling = (fan_cooling+ac_cooling)*cooling_pattern
        local_load = load*heat_pattern
        net = local_load-local_cooling-passive/8.0-pcm_exchange/8.0
        # 双预测MPC按分区温度偏差分配冷量：高温区增加冷却，低温区减少冷却。
        if mode == "双预测MPC":
            zone_deviation = zone - zone.mean()
            net -= 0.22 * zone_deviation
        # 区域间热扩散，抑制非物理的温差无限累积，但保留真实非均匀性。
        net += 0.22*(zone.mean()-zone)
        zone += net*(dt/3600.0)/cap_kwh_k
        zone = np.clip(zone, 5.0, 65.0)

        # PCM温度动态：吸热升温，向环境/冷却空气缓慢释放热量。
        pcm_temp += (pcm_exchange*dt/3600.0)/max(0.05, pcm_mass*s.pcm_cp/3.6e6)
        pcm_temp -= 0.012*(pcm_temp-s.ambient)*(dt/3600.0)
        pcm_temp = float(np.clip(pcm_temp, s.ambient-3.0, 50.0))

        fan_e += fan_kw*dt/3600.0
        ac_e += ac_kw*dt/3600.0
        valve_e += valve_kw*dt/3600.0
        aux_e += aux_kw*dt/3600.0
        if fan > 0.02: fan_h += dt/3600.0
        if ac > 0.02: ac_h += dt/3600.0
        if tmax > s.target_t: over_h += dt/3600.0
        if spread > s.target_spread: spread_over_h += dt/3600.0

        liquid_now, remain_now, _ = _pcm(s, pcm_temp)
        total_e = fan_e+ac_e+valve_e+aux_e
        row = {
            "time_h": hour, "Tmax_C": float(zone.max()), "Tmin_C": float(zone.min()),
            "dTmax_C": float(zone.max()-zone.min()), "Tmean_C": float(zone.mean()),
            "Tair_mean_C": float(s.ambient), "PCM_T_C": pcm_temp,
            "PCM_liquid": liquid_now, "PCM_remaining_MJ": remain_now,
            "fan_speed": fan, "fan_power_W": fan_kw*1000.0,
            "ac_fraction": ac, "ac_power_W": ac_kw*1000.0,
            "valve_pct": valve*100.0, "fan_pct": fan*100.0, "ac_pct": ac*100.0,
            "fan_energy_kWh": fan_e, "ac_energy_kWh": ac_e,
            "valve_energy_kWh": valve_e, "aux_energy_kWh": aux_e,
            "total_energy_kWh": total_e, "fan_runtime_h": fan_h, "ac_runtime_h": ac_h,
            "airflow_m3h": airflow, "heat_current_kW": load,
            "over_target_h": over_h, "over_spread_h": spread_over_h,
            # 兼容旧版网页中可能引用的字段名
            "fan_energy_kwh": fan_e, "ac_energy_kwh": ac_e,
            "valve_energy_kwh": valve_e, "aux_energy_kwh": aux_e,
            "total_energy_kwh": total_e, "fan_runtime_h": fan_h,
            "ac_runtime_h": ac_h, "fan_speed_pct": fan*100.0,
            "ac_load_pct": ac*100.0, "pcm_liquid_pct": liquid_now*100.0,
            "pcm_remaining_mj": remain_now, "Tmax": float(zone.max()),
            "Tmean": float(zone.mean()), "dTmax": float(zone.max()-zone.min()),
            "valve_opening_pct": valve*100.0
        }
        rec.append(row)
        prev_tmax = tmax
        fan_old, ac_old, valve_old = fan, ac, valve

    return pd.DataFrame(rec)

def run_all(s, dt=60):
    return {
        "固定送风": run_one(s, "固定送风", dt),
        "固定阈值": run_one(s, "固定阈值", dt),
        "双预测MPC": run_one(s, "双预测MPC", dt)
    }

def summarize(df, s=None):
    if df is None or df.empty:
        return {}
    last = df.iloc[-1]
    return {
        "电池最高温度（℃）": round(float(df["Tmax_C"].max()), 2),
        "最大温差（℃）": round(float(df["dTmax_C"].max()), 2),
        "平均电池温度（℃）": round(float(df["Tmean_C"].mean()), 2),
        "超温累计时间（h）": round(float(last["over_target_h"]), 2),
        "温差超限累计时间（h）": round(float(last["over_spread_h"]), 2),
        "风机累计能耗（kWh）": round(float(last["fan_energy_kWh"]), 3),
        "空调累计能耗（kWh）": round(float(last["ac_energy_kWh"]), 3),
        "风阀及辅助能耗（kWh）": round(float(last["valve_energy_kWh"]+last["aux_energy_kWh"]), 4),
        "系统总能耗（kWh）": round(float(last["total_energy_kWh"]), 3),
        "风机运行时间（h）": round(float(last["fan_runtime_h"]), 2),
        "空调运行时间（h）": round(float(last["ac_runtime_h"]), 2),
        "PCM末端液相率（%）": round(float(last["PCM_liquid"])*100, 1),
        "PCM剩余储热能力（MJ）": round(float(last["PCM_remaining_MJ"]), 2)
    }

__all__ = ["Settings","run_one","run_all","summarize","FAN_DB","AC_DB"]
