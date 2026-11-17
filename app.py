import json
from pathlib import Path
import streamlit as st
import pandas as pd

BASE=Path(__file__).parent
def load_json(name):
    return json.loads((BASE/"data"/name).read_text(encoding="utf-8"))

topo=load_json("target_baseline.json")
pcm=load_json("wt_cpcm_reference.json")
cell=load_json("lf280k_reference.json")
sources=load_json("data_sources.json")

st.set_page_config(page_title="280Ah电池柜智能热管理模型",page_icon="🔋",layout="wide")
st.title("🔋 280Ah 磷酸铁锂电池柜智能热管理模型")
st.caption("公开演示平台 · 几何基线、文献参数、数据来源与模型验证状态")
st.info("重要说明：当前网站展示的是研究模型的数据基线与公开资料，不代表已经完成目标电池柜的真实温度预测。由于目标电芯物性、PCM密度/显热比热、界面热导和风机曲线尚未全部闭合，真实工况求解入口暂未开放。")

c1,c2,c3,c4=st.columns(4)
c1.metric("目标电芯数量",topo["cell_count"])
c2.metric("PCM区域",topo["pcm_regions"])
c3.metric("ODE状态目标",topo["ode_states"])
c4.metric("STEP实体数",topo["solid_count"])

tabs=st.tabs(["模型总览","几何与结构","WT-CPCM材料","280Ah电芯数据","数据来源","模型运行条件"])
with tabs[0]:
    st.subheader("模型架构")
    left,right=st.columns(2)
    with left:
        st.markdown("""
        **电池温度预测子模型**
        - 144个电芯热节点
        - 电芯间空气间隙与包间空气间隙
        - 电芯—PCM接触候选热路径
        - 真实热源与空气边界接入后求解
        """)
        st.markdown("""
        **PCM储热状态子模型**
        - 24个PCM区域
        - WT-CPCM分阶段焓参考数据
        - 每区保留历史最高温度状态
        - 论文未给出的物性不自动补造
        """)
    with right:
        st.markdown("""
        **MPC控制目标**
        - 电芯最高温度约束
        - 电芯间温差约束
        - 风机功耗与风量约束
        - 温控效果与能耗多目标权衡
        """)
        st.warning("MPC闭环尚未接入经过验证的192状态真实参数求解器。")
    st.subheader("研究流程")
    st.code("工况输入 → 144芯温度预测 + 24区PCM焓状态预测 → MPC风机控制 → 温度/温差/能耗评价",language="text")
with tabs[1]:
    st.subheader("当前采用的几何基线")
    dims=[
        ("电池舱（长×宽×高）","8200 × 2600 × 2438 mm"),
        ("电池柜（长×宽×高）","1410 × 675 × 1987 mm"),
        ("电芯（长×宽×高）","295 × 248 × 104 mm"),
        ("电芯间距","7 mm"),
        ("电池包间距","29 mm"),
        ("电池包至柜体侧壁间隙","72 mm"),
        ("前后、顶部、底部间隙","各 29 mm"),
        ("每个电池包外侧PCM厚度","7 mm"),
        ("导流板","2400 × 2000 × 3 mm"),
        ("导流板圆孔","直径20 mm，贯穿厚度3 mm"),
    ]
    st.table(pd.DataFrame(dims,columns=["参数","当前基线"]))
    st.caption(f"STEP审计：{topo['solid_count']}个实体；包围盒约 {topo['cabinet_bbox_mm'][0]:.2f} × {topo['cabinet_bbox_mm'][1]:.2f} × {topo['cabinet_bbox_mm'][2]:.2f} mm。")
    st.warning("舱体/间距数据采用用户最新确认值；STEP实体数量和包围盒来自已恢复CAD文件。具体装配方向和各PCM接触面仍以最终装配体映射为准。")
with tabs[2]:
    st.subheader("WT-CPCM文献参考参数")
    st.write(pcm["source"])
    st.write("DOI：",pcm["doi"])
    a,b=st.columns(2)
    a.metric("总吸热焓参考值",f"{pcm['total_endothermic_enthalpy_J_kg']/1000:.2f} kJ/kg")
    a.metric("25℃导热系数",f"{pcm['thermal_conductivity_W_mK']['25C']:.2f} W/(m·K)")
    b.metric("55℃导热系数",f"{pcm['thermal_conductivity_W_mK']['55C']:.2f} W/(m·K)")
    stage=pd.DataFrame([
        {"阶段":"I","温区":f"{pcm['stage_temperature_ranges_C'][0][0]}–{pcm['stage_temperature_ranges_C'][0][1]}℃","吸热焓":f"{pcm['stage_enthalpy_J_kg'][0]/1000:.2f} kJ/kg"},
        {"阶段":"II","温区":f"{pcm['stage_temperature_ranges_C'][1][0]}–{pcm['stage_temperature_ranges_C'][1][1]}℃","吸热焓":f"{pcm['stage_enthalpy_J_kg'][1]/1000:.2f} kJ/kg"},
        {"阶段":"III","温区":f"{pcm['stage_temperature_ranges_C'][2][0]}–{pcm['stage_temperature_ranges_C'][2][1]}℃","吸热焓":f"{pcm['stage_enthalpy_J_kg'][2]/1000:.2f} kJ/kg"},
    ])
    st.dataframe(stage,use_container_width=True,hide_index=True)
    st.write("文献配方质量分数（%）")
    st.dataframe(pd.DataFrame([pcm["composition_wt_percent"]]),use_container_width=True,hide_index=True)
    st.error("文献未提供当前样品批次可直接确认的密度、显热比热及II/III阶段动力学参数。公开页面将其标为待补证据，不据此伪造PCM温度曲线。")
with tabs[3]:
    st.subheader("280Ah电芯公开参考数据")
    st.write("候选规格：",cell["selected_reference"])
    st.metric("标称容量",f"{cell['capacity_Ah']} Ah")
    st.metric("标称电压",f"{cell['nominal_voltage_V']} V")
    st.write("候选质量：",f"{cell['weight_kg_nominal']} ± {cell['weight_kg_tolerance']} kg")
    st.write("候选尺寸：",cell["dimensions_mm"])
    st.write("候选比热范围：",cell["specific_heat_J_kgK"],"J/(kg·K)")
    st.write("候选导热系数：",cell["thermal_conductivity_W_mK"])
    st.error("LF280K只是公开候选型号，尚未确认与你的放大CAD单体对应。其质量、密度、比热不能直接套到295×248×104 mm的CAD单体。")
with tabs[4]:
    st.subheader("数据集与文献")
    for s in sources:
        st.markdown(f"**{s['name']}**  \n{s['url']}  \n用途：{s['role']}  \n适用性：{s['priority']}；本地已恢复：{'是' if s.get('present_locally') else '否'}")
        st.divider()
with tabs[5]:
    st.subheader("公开演示输入面板")
    st.write("以下设置用于定义后续真实求解工况。当前真实求解被物理参数门禁锁定。")
    col1,col2=st.columns(2)
    with col1:
        ambient=st.number_input("环境温度（℃）",value=30.0,step=1.0)
        rate=st.number_input("放电倍率（C）",value=1.0,min_value=0.0,step=0.1)
    with col2:
        fan=st.selectbox("初始风机状态",["关闭","开启"])
        strategy=st.selectbox("控制策略",["MPC预测控制","固定风量基准","自然冷却对照"])
    st.write({"环境温度_C":ambient,"放电倍率_C":rate,"初始风机":fan,"控制策略":strategy})
    st.button("运行真实模型",disabled=True,help="待目标电芯热容、PCM密度/比热、界面热导及风机曲线审核通过后开放。")
    st.warning("当前状态：真实模型运行入口锁定。不会显示虚构的最高温度、最大温差或节能率。")

st.divider()
st.caption("版本：公开资料与几何基线展示版。模型结果须经目标参数辨识及独立实验验证后方可用于工程结论。")
