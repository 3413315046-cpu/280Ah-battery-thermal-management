import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from thermal_engine import Settings, run_all, summarize, FAN_DB, AC_DB

st.set_page_config(
    page_title="280Ah储能电池舱｜双状态预测与智能热管理",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 高对比度浅色主题：保证标题、图例、坐标轴与正文清晰可读
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: #f3f6fb; color: #172b43; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #dbe4ef; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1600px; }
h1,h2,h3,h4,p,label,span,div { color: inherit; }
.hero {
  background: linear-gradient(120deg,#102b46 0%,#174d62 100%);
  padding: 26px 30px; border-radius: 18px; color: white; margin-bottom: 18px;
  box-shadow: 0 8px 24px rgba(15,45,70,.12);
}
.hero h1,.hero p { color: #ffffff !important; margin: 0; }
.hero p { opacity: .88; margin-top: 8px; }
.kpi {
  background: #ffffff; border: 1px solid #dce6f0; border-radius: 14px;
  padding: 16px 18px; min-height: 108px; box-shadow: 0 2px 8px rgba(20,45,70,.04);
}
.kpi-label { color: #52677d; font-size: 14px; font-weight: 600; }
.kpi-value { color: #122b45; font-size: 29px; font-weight: 750; margin-top: 8px; }
.kpi-sub { color: #557087; font-size: 12px; margin-top: 3px; }
.group-card {
  background: white; border: 1px solid #dce6f0; border-radius: 14px;
  padding: 17px; height: 100%;
}
.group-card h3 { color: #17334f; margin-top: 0; }
.group-card p { color: #536b80; margin-bottom: 0; }
.note {
  background: #fff8e8; border: 1px solid #f0d9a6; color: #6f4e10;
  padding: 12px 15px; border-radius: 10px; font-size: 13px;
}
div[data-testid="stDataFrame"] { background: white; }
</style>
""", unsafe_allow_html=True)

COLORS = {"固定送风": "#8396aa", "固定阈值": "#347fbd", "双预测MPC": "#079b8b"}

def metric_card(label, value, sub=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True
    )

def line_chart(results, column, title, y_title, target=None):
    fig = go.Figure()
    for name, df in results.items():
        fig.add_trace(go.Scatter(
            x=df["time_h"], y=df[column], mode="lines", name=name,
            line=dict(color=COLORS[name], width=3 if name == "双预测MPC" else 2.2),
            hovertemplate="%{x:.1f} h<br>%{y:.2f}<extra>" + name + "</extra>"
        ))
    if target is not None:
        fig.add_hline(y=target, line_dash="dash", line_color="#df8b28",
                      annotation_text=f"控制目标 {target:g}", annotation_position="top right")
    fig.update_layout(
        title=dict(text=title, font=dict(size=19, color="#17334f")),
        height=390, template="plotly_white", hovermode="x unified",
        margin=dict(l=25,r=25,t=65,b=35),
        legend=dict(orientation="h", y=-0.22, font=dict(color="#314b63")),
        xaxis_title="运行时间（h）", yaxis_title=y_title,
        font=dict(color="#243b53"),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e6edf4", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e6edf4", zeroline=False)
    return fig

st.markdown("""
<div class="hero">
  <h1>🔋 280Ah 磷酸铁锂储能电池舱</h1>
  <p>双状态预测 · WT-CPCM被动储热 · 空冷主动排热 · MPC协同调控</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 运行工况")
    ambient = st.slider("环境温度（℃）", 15, 45, 30, 1)
    initial = st.slider("电池初始温度（℃）", 15, 45, 30, 1)
    rate = st.slider("等效热负荷倍率", 0.2, 2.0, 1.0, 0.05)
    duration = st.slider("仿真运行时长（h）", 1, 48, 48, 1)
    profile = st.selectbox("电池负荷时序", ["完整充放电循环", "恒定放电", "间歇运行"])
    st.divider()
    st.subheader("设备配置")
    fan_type = st.selectbox("舱级风机", list(FAN_DB.keys()))
    fans = st.number_input("风机数量", min_value=1, max_value=16, value=4, step=1)
    ac_type = st.selectbox("空调机组", list(AC_DB.keys()))
    fixed_speed = st.slider("固定送风组风机转速比例", 0.2, 1.0, 0.70, 0.05)
    fixed_valve = st.slider("固定送风组风阀开度（%）", 20, 100, 75, 5) / 100
    st.divider()
    st.subheader("控制目标")
    target_t = st.slider("电池最高温度控制目标（℃）", 32, 45, 38, 1)
    target_spread = st.slider("最大温差控制目标（℃）", 1.0, 5.0, 3.0, 0.1)
    ac_setpoint = st.slider("空调参考设定温度（℃）", 20, 32, 27, 1)
    st.caption("设备功率与风量采用当前录入的典型规格；热负荷及舱级等效热参数仍需后续标定。")

settings = Settings(
    ambient=float(ambient), initial=float(initial), rate=float(rate),
    duration_h=int(duration), profile=profile,
    fans=int(fans), fan_type=fan_type, ac_type=ac_type,
    fixed_speed=float(fixed_speed), fixed_valve=float(fixed_valve),
    target_t=float(target_t), target_spread=float(target_spread),
    ac_setpoint=float(ac_setpoint),
)

@st.cache_data(show_spinner=False)
def calculate(settings):
    return run_all(settings, dt=300)

with st.spinner("正在进行三组同工况仿真……"):
    results = calculate(settings)

summaries = {name: summarize(df, settings) for name, df in results.items()}
mpc = summaries["双预测MPC"]
base = summaries["固定送风"]
threshold = summaries["固定阈值"]
energy_saving = (threshold["系统总能耗（kWh）"] - mpc["系统总能耗（kWh）"]) / max(1e-9, threshold["系统总能耗（kWh）"]) * 100

st.subheader("同工况三组对照总览")
k1,k2,k3,k4 = st.columns(4)
with k1: metric_card("双预测MPC最高温度", f'{mpc["电池最高温度（℃）"]:.2f} ℃', "48小时（或所选时长）仿真")
with k2: metric_card("双预测MPC最大温差", f'{mpc["最大温差（℃）"]:.2f} ℃', f'控制目标 ≤ {target_spread:.1f} ℃')
with k3: metric_card("双预测MPC系统总能耗", f'{mpc["系统总能耗（kWh）"]:.2f} kWh', f'相对固定阈值组 {energy_saving:+.1f}%')
with k4: metric_card("双预测MPC超温累计时间", f'{mpc["超温累计时间（h）"]:.2f} h', f'温度阈值 {target_t:.1f} ℃')

st.markdown('<div class="note">说明：以下为当前简化分区热模型的计算结果，属于仿真演示数据，不是电池舱实测或经CFD验证的性能数据。三组采用相同环境温度、初始温度、负荷时序、设备规格和仿真时长；实际性能结论需用后续标定参数复核。</div>', unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown('<div class="group-card"><h3>① 固定送风组</h3><p>风机按固定转速运行、风阀保持固定开度；空调采用固定温控启停，不进行预测优化。</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="group-card"><h3>② 固定阈值组</h3><p>依据当前最高温度和温差触发风机、空调及风阀动作，不使用未来状态预测。</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="group-card"><h3>③ 双预测 MPC 组</h3><p>结合电池温度趋势和PCM储热状态进行滚动决策，协调风机、空调及风阀开度。</p></div>', unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs(["① 对照评价","② 温度与温差","③ PCM储热状态","④ 设备控制与能耗","⑤ 设备参数库"])

with tab1:
    st.subheader("三组评价指标")
    order = ["固定送风","固定阈值","双预测MPC"]
    metric_rows = [
        "电池最高温度（℃）","最大温差（℃）","平均电池温度（℃）",
        "超温累计时间（h）","温差超限累计时间（h）",
        "风机累计能耗（kWh）","空调累计能耗（kWh）",
        "风阀及辅助能耗（kWh）","系统总能耗（kWh）",
        "风机运行时间（h）","空调运行时间（h）"
    ]
    table = pd.DataFrame({name: summaries[name] for name in order}).T[metric_rows].T
    st.dataframe(table, use_container_width=True)
    st.caption("最大温差为各时刻八个等效区域最高温度与最低温度之差的全时段最大值；不会将固定组温差人为置零。")
    st.subheader("系统总能耗构成")
    fig = go.Figure()
    for col,label,color in [
        ("风机累计能耗（kWh）","风机","#347fbd"),
        ("空调累计能耗（kWh）","空调","#e39a36"),
        ("风阀及辅助能耗（kWh）","风阀及辅助","#079b8b")
    ]:
        fig.add_trace(go.Bar(name=label, x=order, y=[summaries[n][col] for n in order]))
    fig.update_layout(barmode="stack", template="plotly_white", height=350,
                      title="三组系统总能耗分项", yaxis_title="能耗（kWh）",
                      font=dict(color="#243b53"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.plotly_chart(line_chart(results,"Tmax_C","电池舱最高温度变化","温度（℃）",target_t), use_container_width=True)
    st.plotly_chart(line_chart(results,"Tmean_C","电池舱平均温度变化","温度（℃）"), use_container_width=True)
    st.plotly_chart(line_chart(results,"dTmax_C","电池组最大温差变化","最大温差（℃）",target_spread), use_container_width=True)

with tab3:
    st.subheader("WT-CPCM储热状态")
    st.plotly_chart(line_chart(results,"PCM_T_C","PCM等效温度变化","温度（℃）"), use_container_width=True)
    st.plotly_chart(line_chart(results,"PCM_liquid","PCM液相率变化","液相率（0–1）"), use_container_width=True)
    st.plotly_chart(line_chart(results,"PCM_remaining_MJ","PCM剩余潜热储能变化","剩余储热能力（MJ）"), use_container_width=True)
    st.caption("PCM指标用于展示被动储热状态并参与MPC决策；其数值依赖PCM体积、密度、潜热和热耦合参数。")

with tab4:
    st.plotly_chart(line_chart(results,"fan_speed","风机转速指令变化","转速比例（0–1）"), use_container_width=True)
    st.plotly_chart(line_chart(results,"ac_fraction","空调负荷指令变化","负荷比例（0–1）"), use_container_width=True)
    st.plotly_chart(line_chart(results,"valve_pct","电动风阀开度变化","开度（%）"), use_container_width=True)
    st.plotly_chart(line_chart(results,"total_energy_kWh","系统累计能耗变化","累计能耗（kWh）"), use_container_width=True)
    st.plotly_chart(line_chart(results,"fan_energy_kWh","风机累计能耗变化","累计能耗（kWh）"), use_container_width=True)
    st.plotly_chart(line_chart(results,"ac_energy_kWh","空调累计能耗变化","累计能耗（kWh）"), use_container_width=True)

with tab5:
    st.subheader("风机参数")
    st.dataframe(pd.DataFrame(FAN_DB).T, use_container_width=True)
    st.subheader("空调参数")
    st.dataframe(pd.DataFrame(AC_DB).T, use_container_width=True)
    st.subheader("固定 WT-CPCM 材料参数")
    st.dataframe(pd.DataFrame([
        ["密度",1100,"kg/m³"],["比热容",1740,"J/(kg·K)"],
        ["导热系数",3.5,"W/(m·K)"],["相变温区","38–43","℃"],
        ["相变潜热",120,"kJ/kg"]
    ],columns=["参数","数值","单位"]),use_container_width=True)

st.divider()
left,right = st.columns(2)
with left:
    st.download_button("下载三组评价指标 CSV",
        pd.DataFrame(summaries).T.to_csv(index=True).encode("utf-8-sig"),
        file_name="thermal_management_summary.csv", mime="text/csv", use_container_width=True)
with right:
    all_df = pd.concat([df.assign(控制组=name) for name,df in results.items()],ignore_index=True)
    st.download_button("下载全时序仿真数据 CSV",
        all_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="thermal_management_timeseries.csv", mime="text/csv", use_container_width=True)
