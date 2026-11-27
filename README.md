# 280Ah储能电池舱智能热管理演示平台（兼容修复包）

## 文件
- `app.py`：全新浅色高对比度 Streamlit 网页。
- `thermal_engine.py`：三组分区热模型、控制逻辑、设备参数库及评价指标。
- `requirements.txt`：运行依赖。

## 部署
1. 将三个文件放在 GitHub 仓库根目录。
2. 确认 Streamlit Cloud 的主文件路径为 `app.py`。
3. 提交后等待自动部署。
4. 本地运行：`pip install -r requirements.txt`，然后 `streamlit run app.py`。

## 重要说明
该版本是可交互的简化分区热模型演示。模型结果不是实测数据，也不是CFD验证结果。
模型的热负荷、等效热容、被动散热系数和PCM热耦合系数仍需用真实电池舱仿真/实验进行标定。
三组控制策略采用相同工况与设备参数；不得把当前演示结果直接写成已验证性能。
