# 复现入口与结果索引

对应《有限预算下的机会捕获：利润边界与测量精度》，2026-10-15版本。

- [论文与完整证明](../论文_有限预算下的机会捕获.pdf)
- [完整复现说明](../复现说明.md)
- [一键运行程序](复现.py)
- [依赖清单](requirements.txt)
- [发布前七组运行核验](复现核验.json)
- [图表生成程序](生成图表.py)

## 从哪里开始

解压完整附件，进入本目录，执行：

```text
python -m pip install -r requirements.txt
python 复现.py
```

七组实验的JSON结果更新在`experiments/`；日志写入本目录。若需要重新生成论文两幅图，另运行`python 生成图表.py`，图表写到附件根目录的`图表/`。中文图表默认使用Windows微软雅黑；其他平台应在图表程序中替换中文字体路径。七组数值实验不依赖该字体。

所有数据为固定种子生成的合成数据，无需下载市场数据。归档JSON可直接阅读，不必先运行。以下链接以本目录为基准：

| 论文内容 | 程序 | 归档结果 |
|---|---|---|
| 定理1、表2、表4上端、图1 | [边界与上端](experiments/second_paper_bounded_revision.py) | [JSON](experiments/second_paper_bounded_revision.json) |
| 附录A、B及核不等式 | [独立LP与核检查](experiments/second_paper_bounded_independent.py) | [JSON](experiments/second_paper_bounded_independent.json) |
| 表4行动程序、图2行动曲线 | [行动证书](experiments/second_paper_bounded_deployment.py) | [JSON](experiments/second_paper_bounded_deployment.json) |
| 定理2、3与附录D | [一般读数与费用分区](experiments/second_paper_general_task_separation.py) | [JSON](experiments/second_paper_general_task_separation.json) |
| 表5盒与球比较 | [几何证书](experiments/second_paper_geometric_precision.py) | [JSON](experiments/second_paper_geometric_precision.json) |
| 表5计先导验证 | [先导与新验证](experiments/second_paper_pilot_sphere.py) | [JSON](experiments/second_paper_pilot_sphere.json) |
| 有限上限、定理4/5与附录E | [应用桥梁](experiments/application_contract_bridge.py) | [JSON](experiments/application_contract_bridge.json) |

[图3生成程序](生成应用图表.py)：运行`python 生成应用图表.py`。

核查记录表示程序正常结束；概率定理及连续测度结论仍以论文证明为准。当前提供本地附件，尚无公开GitHub地址或Zenodo DOI。
