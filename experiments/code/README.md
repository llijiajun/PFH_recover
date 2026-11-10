# 实验代码目录说明（experiments/code/）

本目录收录 **Diffusion4Distribution** 论文的全部可复现实验代码。论文把"已知采样率 `p` 的抽样"建模为加噪（二项稀疏 binomial thinning），把"从样本恢复总体频数直方图"建模为去噪（denoising）；核心贡献是 **Gamma–Poisson 共轭下的闭式解析逆**与**校准的不确定性带**。

本公开仓库不含论文正文、LaTeX 源码与论文出图脚本。本目录只放实验复现脚本、数据与归档结果。

## 1. 目录结构

```
experiments/code/
├── d4d_logic.py            # 核心算法：稀疏、后验、plausible set、学习组件
├── d4d_metrics.py          # 指标与单次属性恢复（含 oracle 经验先验）
├── d4d_baselines.py        # 经典基线频谱/NDV 估计（Good–Turing、Chao1、Chebyshev 等）
├── d4d_modern_baselines.py # 现代基线（NPMLE、Smooth Bayes、多项式矩匹配、泛函）
├── eb_prior.py             # 经验贝叶斯 Gamma–Poisson 先验拟合
├── validate_*.py           # 各类验证脚本（基线 / 真实数据 / joint / 拟合先验…）
├── analyze_prior_and_joint.py  # 先验敏感性与联合尺度诊断表
├── validate_downstream_fair_comparison.py  # 下游公平比较与配对检验
├── make_unknown_support_table.py  # 未知支持网格表
├── make_fof_recovery_figure.py  # FoF 复原折线图 + 偏离图（部署态 vs 真值）
├── Diffusion4Distribution_algorithm_tutorial.ipynb  # 算法原理走查（可运行）
└── figures/                # 图，PDF / EPS / SVG 为矢量交付，PNG 仅预览
```

## 2. 核心算法模块

| 文件 | 关键符号 | 对应论文 |
|---|---|---|
| `d4d_logic.py` | `thin` / `analytic_posterior` / `plausible_set` / `Thickener` / `train_thickener` / `chain_thickener` / `w1_distance` / `fofo_spectrum` | 前向加噪（Algorithm 1 noising）；解析逆（Algorithm 1）；90% plausible set；学习端点（Algorithm 2）；W1 指标；频率频谱 $F_j$ |
| `d4d_metrics.py` | `posterior_mean_counts` / `validate_attribute` / `prior_from_truth` | 逐类型后验均值；单次属性完整恢复；oracle 经验先验 |
| `d4d_baselines.py` | `spectrum_good_turing` / `spectrum_chebyshev` / `ndv_chao1` / `evaluate_all` | 经典基线频谱 / NDV（论文用 Chebyshev 矩匹配，不使用未发表的 SVD 基线） |
| `d4d_modern_baselines.py` | `npmle_prior` / `spectrum_npmle` / `smooth_bayes_params` / `spectrum_smooth_bayes` / `downstream_functionals` | NPMLE 先验拟合；闭式后验反卷积；Gamma–Poisson 平滑贝叶斯；下游泛函 |
| `eb_prior.py` | `fitted_prior_eb` | 用单/双例质量分数拟合 $(a_0,b_0)$ 经验贝叶斯先验 |

**依赖关系**：顶层脚本直接 `import d4d_logic` 等扁平模块，**不要移动或重命名这些顶层模块**，否则复现链会断。一次性消融实验已移入 `experiments_archive/`，未收录在本公开仓库中。

## 3. 验证 / 分析脚本（活跃复现入口）

| 脚本 | 产出 | 论文对应 |
|---|---|---|
| `analyze_prior_and_joint.py` | `experiments/results/tables/tab_prior_sensitivity.tex`、`tab_joint_diagnosis.tex` 及 `experiments/results/prior_and_joint_diagnosis/{summary.json,report.md}` | Table III（先验敏感性 root cause）、Table VII（联合尺度诊断） |
| `validate_downstream_fair_comparison.py` | `summary.json` / `report.md` / 三张表格片段 | 下游 self-join / join-cardinality 公平比较与配对检验 |
| `make_unknown_support_table.py` | `experiments/results/tables/tab_unknown_support.tex` | 未知支持网格 |
| `make_fof_recovery_figure.py` | `figures/fof_recovery_comparison.{pdf,eps,svg,png}`、`figures/fof_recovery_deviation.{pdf,eps,svg,png}` 与 `experiments/results/fof_recovery_figure/{summary.json,report.md}` | FoF 复原折线图与偏离图。4 个低采样率（$p \le 0.05$）案例，完整总体 FoF 真值 / 部署态解析逆 / Good–Turing / Naive 1/p / Chebyshev / SB linear 对比；另报告 oracle 信息上限、逐 $j$ 偏差比与 $p$ 扫描；指标为 W1、TV、RMSE 与绝对数量 NDV 误差 |
| `validate_baselines.py` | 基线频谱与 NDV 对比 | 正文 Table IV |
| `validate_real_data.py` | 真实语料结果 | 真实数据实验 |
| `validate_joint_recovery.py` | 联合分布恢复（self-join $\sum f_{vw}^2$） | 联合分布节 |
| `validate_fitted_prior.py` | 拟合先验一致性 | 部署先验 |
| `validate_modern_baselines.py` / `validate_optimizer_readout.py` | 现代基线 / 优化器读出 | 现代基线对比 |
| `verify_diffusion_semigroup.py` / `verify_modern_claims.py` | 理论自检 | 理论节 |
| `significance_tests.py` | 配对显著性 | 统计严谨性 |

> 所有数字必须可回溯到 `experiments/results/` 的 `summary.json` 或日志，禁止凭记忆填数。改数字前先对回对应 `summary.json`。

## 4. 表格与图的输出位置

脚本按仓库根目录解析路径，不依赖任何本机绝对路径。

- 表格片段写入 `experiments/results/tables/`，目录在脚本运行时自动创建。
- `make_hist_figures.py` 把 W1 / TV / RMSE 面板写入 `experiments/results/figures/`。
- `make_fof_recovery_figure.py` 把 FoF 对比图与偏离图写入本目录的 `figures/`，数字写入 `experiments/results/fof_recovery_figure/`。

仓库不含 LaTeX 构建链，也不含论文正文。表格片段只是分析结果的文本导出，可单独查看。

## 5. 关于一次性消融存档

早期消融实验的一次性脚本未收录在本公开仓库中。表格的复现只依赖本目录顶层脚本与 `../results/` 下的归档结果。

## 6. 教学 notebook

`Diffusion4Distribution_algorithm_tutorial.ipynb` 是一份**额外**的、从原理讲清本算法的可运行 notebook，覆盖：前向稀疏、Gamma–Poisson 闭式后验（数值核对与枚举一致）、逐类型恢复、**FoF 复原折线图（4 个低采样率典型案例，多方法与完整总体真值对比）**、90% plausible set 校准、学习组件定位与诚实边界、三条方法边界、代码与术语对照表。把它放在本目录运行即可（`d4d_logic` 会自动加入路径）。所有数值由代码实算，不靠记忆。

notebook 中 FoF 图那一节画的是**绝对类型数 $F_j$**（log 轴），真值定义为完整总体中真实频次等于 $j$ 的 distinct value 数量，用粗黑线加空心黑点**最后绘制**，压在所有估计曲线之上。所有曲线都估计同一个 $F_j$，共享同一 $f$ 轴。**观测样本的 FoF 定义在稀疏后的 $g$ 轴上，不是 $F_j$ 的估计量，与真值不在同一度量上，因此不进图也不进表**。采样率全部取 $p \le 0.05$，只有 5%–19% 的真值类型能留下观测。该节同时打印 W1 / TV / RMSE 与 NDV 相对误差，归一化误差与绝对数量误差两个维度并排。

**先验口径（重要）**。早期版本只画 `prior_from_truth` 构造的 oracle 先验，先验本身来自完整总体真值，所以低采样率下曲线与真值几乎完全重合。那是后验公式的自洽性检验，不是复原。现在主图的解析逆用 `fitted_prior_eb` **只从稀疏样本拟合先验**，是真正的部署态，TV 为 0.37–0.44、NDV 低估 47%–71%。oracle 先验只作为明确标注的信息上限出现在偏离图里，TV 为 0.006–0.026。偏离图还固定一个总体扫描 $p$ 从 0.1 到 0.002，部署态 TV 从 0.29 升到 0.51，量化了「采样率越低越难贴近真值」。四个合成案例比论文真实列实验（$p \ge 0.1$）更稀疏，属于边界研究，不代表论文表格里的池化精度。

## 7. 运行环境

- 解释器：Python 3.11 及以上（需 numpy / scipy / matplotlib / torch）。
- 教学 notebook 只需 numpy / scipy / matplotlib；学习组件（`Thickener` / `train_*`）需要 torch。
- 入口示例：
  ```bash
  cd experiments/code
  python Diffusion4Distribution_algorithm_tutorial.ipynb   # 用 jupyter 打开运行
  python analyze_prior_and_joint.py                        # 复现 Table III / VII
  python make_fof_recovery_figure.py                       # 生成 FoF 对比图 / 偏离图与数字
  python make_hist_figures.py                              # 生成 W1 / TV / RMSE 面板
  ```
