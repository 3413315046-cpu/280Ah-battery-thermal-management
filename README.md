# 280Ah储能电池舱智能热管理网站 v8.0

## 功能
- 三组同工况对照：固定送风、固定阈值反馈、双预测状态+MPC。
- 风机、空调压缩机和电动风阀协同控制。
- 0–48h全过程逐分钟曲线：最高温度、最大温差、平均温度、等效空气温度、控制动作、设备分项能耗与系统总能耗。
- PCM焓状态保留在MPC内部；PCM液相率和剩余储热能力不列入最终评价表。
- 浅色高对比度界面，支持三组逐分钟CSV导出。

## 运行
```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

## 更新原GitHub仓库
上传并覆盖仓库根目录中的 `app.py`、`thermal_engine.py`、`requirements.txt`、`README.md`；新增/更新 `data/`、`tests/`、`.streamlit/config.toml`。保留目录结构，提交到原来的 `main` 分支，等待Streamlit Cloud重新部署。

## 默认48小时计算结果
见 `results_48h_default_summary.csv`；完整逐分钟曲线见 `results_48h_default_timeseries.csv`；工况、结果和压力测试记录见 `验收报告_48h默认工况.json`。

## 重要模型边界
当前是12个电池等效热区、24个PCM焓节点和12个等效空气节点的降阶模型。空调制冷量作为空气节点热汇，设备容量/输入功率采用用户提供的候选设备参数；风道流量折减、热交换系数、空气热容、电芯内阻/质量/比热仍需实测、CFD或参数辨识校准。模型结果不是现场实测，也不能直接宣称已经通过CFD验证。MPC为离散候选控制量的滚动预测搜索，PCM焓状态进入预测状态。所有三组共用同一热模型和负荷时序，只改变控制策略。

系统总能耗当前按风机、空调压缩机及风阀执行器功耗积分；未单独建模的控制器/辅助设备功耗需后续补充。高温高负荷压力测试中MPC温控优于阈值控制，但总能耗略高，说明算法仍存在工况相关的能耗—温控权衡，不能声称所有工况均同时领先。
