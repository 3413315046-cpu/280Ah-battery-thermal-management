"""
280Ah储能电池舱热管理仿真引擎
版本：兼容 app.py 的导入接口
导出：Settings, run_all, summarize, FAN_DB, AC_DB

说明：
1. 本模块为集中参数/分区等效热模型，用于网页交互演示和控制策略对比。
2. 输出属于模型计算结果，不等同于实测或经CFD/实验验证的数据。
3. 设备参数采用项目提供的典型规格；未提供的热负荷、传热系数等为可调整的模型假设。
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import math
import numpy as np
import pandas as pd


# ------------------------- 设备参数库 -------------------------

FAN_DB = {
    "EC轴流500mm": {
        "rated_power_W": 200.0,
        "rated_airflow_m3h": 4500.0,
        "rated_pressure_Pa": 350.0,
        "rated_rpm": 1800.0,
        "min_speed": 0.0,
        "max_speed": 1.0,
        "voltage": "48VDC",
        "control": "0-10V / PWM",
        "pq": [(0, 350), (1000, 330), (2000, 290),
               (3000, 230), (4000, 150), (4500, 0)],
    },
    "EC离心355mm": {
        "rated_power_W": 500.0,
        "rated_airflow_m3h": 1800.0,
        "rated_pressure_Pa": 1442.0,
        "rated_rpm": 2600.0,
        "min_speed": 0.0,
        "max_speed": 1.0,
        "voltage": "230VAC",
        "control": "0-10V / RS485",
        "pq": [(0, 1442), (400, 1350), (800, 1200),
               (1200, 1000), (1600, 700), (1800, 0)],
    },
}

AC_DB = {
    "壁挂式7.5kW": {
        "cooling_capacity_kW": 7.5,
        "rated_power_kW": 3.2,
        "rated_current_A": 6.5,
        "airflow_m3h": 2000.0,
        "eer": 2.34,
        "voltage": "380V/50Hz三相",
        "control": "RS485/Modbus RTU",
    },
    "顶装式20kW": {
        "cooling_capacity_kW": 20.0,
        "rated_power_kW": 9.1,
        "rated_current_A": 18.0,
        "airflow_m3h": 6000.0,
        "eer": 2.20,
        "voltage": "380V/50Hz三相",
        "control": "RS485/Modbus RTU",
        "frequency_capacity_curve": [(30, 12), (40, 15), (50, 20), (60, 22)],
    },
}


# ------------------------- 默认工况 -------------------------

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


# ------------------------- 辅助计算 -------------------------

def _clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def _fan_spec(s: Settings):
    return FAN_DB.get(s.fan_type, FAN_DB["EC轴流500mm"])


def _ac_spec(s: Settings):
    return AC_DB.get(s.ac_type, AC_DB["壁挂式7.5kW"])


def _fan_power_kw(s: Settings, speed: float) -> float:
    spec = _fan_spec(s)
    return s.fans * spec["rated_power_W"] * (_clip(speed) ** 3) / 1000.0


def _ac_power_kw(s: Settings, fraction: float) -> float:
    spec = _ac_spec(s)
    # 变频设备采用功率随负荷近似变化；壁挂定频机按启停平均功率计。
    f = _clip(fraction)
    if f <= 0:
        return 0.0
    return spec["rated_power_kW"] * (f ** 1.5)


def _thermal_load(s: Settings, t_h: float) -> float:
    """简化工况热负荷：随倍率和充放电阶段变化，单位kW。"""
    cycle = (t_h % 4.0)
    if s.profile == "恒定放电":
        factor = 1.0
    elif s.profile == "间歇运行":
        factor = 0.25 if cycle < 1.0 else 0.85
    else:
        # 充放电循环：放电段高负荷、充电段较低负荷、静置段低负荷
        factor = 1.0 if cycle < 2.0 else (0.55 if cycle < 3.0 else 0.12)
    return max(0.0, s.heat_gain_kW_at_1C * max(0.1, s.rate) ** 1.65 * factor)


def _pcm_state(s: Settings, pcm_temp: float) -> Tuple[float, float, float]:
    """返回液相率、剩余潜热(MJ)、等效PCM温度(℃)。"""
    mass = max(0.0, s.pcm_volume * s.pcm_density)
    liquid = _clip((pcm_temp - s.pcm_t_start) /
                   max(0.1, s.pcm_t_end - s.pcm_t_start))
    remaining = mass * s.pcm_latent_kJ_kg * (1.0 - liquid) / 1000.0
    return liquid, remaining, pcm_temp


def _control(mode: str, s: Settings, tmax: float, spread: float,
             pcm_liquid: float, trend: float, previous_fan: float,
             previous_ac: float) -> Tuple[float, float, float]:
    """返回风机转速比例、空调负荷比例、风阀开度。"""
    if mode == "固定送风":
        return _clip(s.fixed_speed), 0.0, _clip(s.fixed_valve)

    if mode == "固定阈值":
        # 无预测反馈，仅按当前最高温度和温差触发。
        if tmax < 35.0:
            fan = 0.0
        elif tmax < 39.0:
            fan = 0.55
        else:
            fan = 0.90
        ac = 0.0
        if tmax >= 40.0:
            ac = 0.65
        elif tmax >= 38.0:
            ac = 0.30
        valve = 0.55 if spread < s.target_spread else 0.90
        return fan, ac, valve

    # 双状态预测 + MPC风格滚动决策：
    # 用短时温升趋势预测未来温度，同时考虑PCM剩余潜热和温差。
    horizon_h = max(1, s.prediction_minutes) / 60.0
    predicted = tmax + max(-0.2, trend) * horizon_h
    pcm_buffer = 1.0 - pcm_liquid
    risk = predicted - (s.target_t - 1.0)

    if predicted < 34.0 and spread < s.target_spread:
        fan, ac = 0.0, 0.0
    else:
        fan = _clip(0.12 + 0.13 * max(0.0, risk) +
                    0.12 * max(0.0, spread - 1.5))
        # PCM潜热储备充足时减少主动冷却；储备不足时提前增加冷却。
        fan *= (0.78 + 0.32 * (1.0 - pcm_buffer))
        ac = _clip((predicted - s.ac_setpoint - 5.0) / 10.0)
        if predicted < s.ac_setpoint + 2.0:
            ac = 0.0
        if pcm_liquid > 0.85 and predicted > s.ac_setpoint + 1.0:
            ac = max(ac, 0.25)
    valve = _clip(0.48 + 0.18 * max(0.0, spread - 1.5) +
                  0.12 * max(0.0, predicted - s.target_t), 0.35, 1.0)

    # 限制控制量变化，避免每个计算步骤出现不现实的突变。
    fan = float(np.clip(fan, previous_fan - 0.18, previous_fan + 0.18))
    ac = float(np.clip(ac, previous_ac - 0.15, previous_ac + 0.15))
    return _clip(fan), _clip(ac), valve


# ------------------------- 核心仿真 -------------------------

def run_one(s: Settings, mode: str, dt: int = 60) -> pd.DataFrame:
    """运行单组仿真。dt为秒，默认60秒。"""
    dt = max(10, int(dt))
    duration_h = max(0.1, min(48.0, float(s.duration_h)))
    n = int(duration_h * 3600 / dt) + 1
    times = np.arange(n) * dt / 3600.0

    # 以舱内多区域等效温度表示温度场，避免将单一温度误称为真实CFD场。
    zones = np.full(8, float(s.initial), dtype=float)
    pcm_temp = float(s.initial)
    fan_prev = 0.0
    ac_prev = 0.0
    fan_energy = ac_energy = valve_energy = aux_energy = 0.0
    fan_runtime = ac_runtime = over_target_h = over_spread_h = 0.0
    records = []
    previous_tmax = float(np.max(zones))

    spec_fan = _fan_spec(s)
    spec_ac = _ac_spec(s)
    fan_count = max(1, int(s.fans))
    fan_capacity = spec_fan["rated_airflow_m3h"] * fan_count
    ac_capacity = spec_ac["cooling_capacity_kW"]

    for i, t_h in enumerate(times):
        tmax = float(np.max(zones))
        tmin = float(np.min(zones))
        spread = tmax - tmin
        tmean = float(np.mean(zones))
        trend = (tmax - previous_tmax) / max(dt / 3600.0, 1e-9)

        pcm_liquid, pcm_remaining, pcm_temp_out = _pcm_state(s, pcm_temp)
        fan, ac, valve = _control(
            mode, s, tmax, spread, pcm_liquid, trend, fan_prev, ac_prev
        )

        fan_kw = _fan_power_kw(s, fan)
        ac_kw = _ac_power_kw(s, ac)
        valve_kw = 0.005 if abs(valve - getattr(run_one, "_last_valve", valve)) > 0.01 else 0.0
        # 执行器动作能耗按动作瞬时折算到当前采样周期。
        valve_kw = min(0.005, valve_kw)
        aux_kw = 0.025
        total_kw = fan_kw + ac_kw + valve_kw + aux_kw

        airflow = fan_capacity * fan * valve
        cooling_fan = 0.00125 * airflow / 1000.0 * max(0.0, tmean - s.ambient + 1.5)
        cooling_ac = ac_capacity * ac * 0.72
        load_kw = _thermal_load(s, t_h)

        # PCM吸放热：温度进入相变区后吸收热量，未达到相变区时仅显热交换。
        pcm_mass = max(0.0, s.pcm_volume * s.pcm_density)
        pcm_exchange_kw = 0.0
        if pcm_mass > 0:
            if tmean > pcm_temp:
                pcm_exchange_kw = min(0.75, 0.00055 * pcm_mass * (tmean - pcm_temp))
            else:
                pcm_exchange_kw = -min(0.18, 0.00018 * pcm_mass * (pcm_temp - tmean))
        if pcm_liquid >= 0.995 and pcm_exchange_kw > 0:
            pcm_exchange_kw *= 0.08

        # 舱体与环境的被动换热。
        passive_kw = s.passive_loss_kW_K * (tmean - s.ambient)

        # 分区热负荷略有差异，形成温差；送风与空调共同降低温度。
        zone_factor = np.array([0.96, 0.98, 1.00, 1.03, 1.05, 1.02, 0.99, 0.97])
        local_load = load_kw * zone_factor
        local_cooling = (cooling_fan + cooling_ac) * np.array(
            [1.08, 1.04, 1.00, 0.96, 0.93, 0.97, 1.02, 1.05]
        )
        zone_net_kw = local_load - local_cooling - passive_kw / 8.0 - pcm_exchange_kw / 8.0

        # 等效热容按舱级系统设定；单位kJ/K转换为kW与小时。
        cap_kWh_K = max(0.5, s.pack_heat_capacity_kJ_K / 3600.0)
        dT = zone_net_kw * (dt / 3600.0) / cap_kWh_K
        zones += dT
        zones = np.clip(zones, 5.0, 65.0)

        pcm_net_kJ = pcm_exchange_kw * dt
        if pcm_mass > 0:
            pcm_temp += pcm_net_kJ / max(1.0, pcm_mass * s.pcm_cp / 1000.0) / 3600.0
            pcm_temp = float(np.clip(pcm_temp, s.ambient - 5.0, 55.0))

        dE = total_kw * dt / 3600.0
        fan_energy += fan_kw * dt / 3600.0
        ac_energy += ac_kw * dt / 3600.0
        valve_energy += valve_kw * dt / 3600.0
        aux_energy += aux_kw * dt / 3600.0
        if fan > 0.02:
            fan_runtime += dt / 3600.0
        if ac > 0.02:
            ac_runtime += dt / 3600.0
        if tmax > s.target_t:
            over_target_h += dt / 3600.0
        if spread > s.target_spread:
            over_spread_h += dt / 3600.0

        pcm_liquid_now, pcm_remaining_now, _ = _pcm_state(s, pcm_temp)
        records.append({
            "time_h": round(float(t_h), 6),
            "Tmax_C": float(np.max(zones)),
            "Tmin_C": float(np.min(zones)),
            "dTmax_C": float(np.max(zones) - np.min(zones)),
            "Tmean_C": float(np.mean(zones)),
            "Tair_mean_C": float(s.ambient),
            "PCM_T_C": float(pcm_temp),
            "PCM_liquid": float(pcm_liquid_now),
            "PCM_remaining_MJ": float(pcm_remaining_now),
            "fan_speed": float(fan),
            "fan_power_W": float(fan_kw * 1000.0),
            "ac_fraction": float(ac),
            "ac_power_W": float(ac_kw * 1000.0),
            "fan_energy_kWh": float(fan_energy),
            "ac_energy_kWh": float(ac_energy),
            "valve_energy_kWh": float(valve_energy),
            "aux_energy_kWh": float(aux_energy),
            "total_energy_kWh": float(fan_energy + ac_energy + valve_energy + aux_energy),
            "fan_runtime_h": float(fan_runtime),
            "ac_runtime_h": float(ac_runtime),
            "airflow_m3h": float(airflow),
            "heat_current_kW": float(load_kw),
            "over_target_h": float(over_target_h),
            "over_spread_h": float(over_spread_h),
        })

        previous_tmax = tmax
        fan_prev, ac_prev = fan, ac
        run_one._last_valve = valve

    df = pd.DataFrame.from_records(records)
    df.attrs["mode"] = mode
    df.attrs["final_zone_temperatures"] = zones.copy()
    return df


def run_all(s: Settings, dt: int = 60) -> Dict[str, pd.DataFrame]:
    """返回三组同工况对照结果。"""
    modes = ("固定送风", "固定阈值", "双预测MPC")
    return {name: run_one(s, name, dt) for name in modes}


def summarize(df: pd.DataFrame, s: Settings = None) -> dict:
    """汇总网页评价指标；输入必须是run_one/run_all产生的DataFrame。"""
    if df is None or len(df) == 0:
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
        "风阀及辅助能耗（kWh）": round(float(last["valve_energy_kWh"] + last["aux_energy_kWh"]), 4),
        "系统总能耗（kWh）": round(float(last["total_energy_kWh"]), 3),
        "风机运行时间（h）": round(float(last["fan_runtime_h"]), 2),
        "空调运行时间（h）": round(float(last["ac_runtime_h"]), 2),
        "PCM末端液相率（%）": round(float(last["PCM_liquid"]) * 100.0, 1),
        "PCM剩余储热能力（MJ）": round(float(last["PCM_remaining_MJ"]), 2),
    }


__all__ = ["Settings", "run_one", "run_all", "summarize", "FAN_DB", "AC_DB"]

