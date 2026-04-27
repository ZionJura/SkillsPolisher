# SkillsPolisher 实验报告

日期：2026-04-23

## 报告说明

本文档统一使用中文记录 `SkillsPolisher` 的研究性实验过程、关键发现与下一步计划。

当前实验默认采用：

- 数据集：`skillsbench`
- 样本数：`10`
- 随机种子：`42`
- 模型：`gpt-4-o`
- 后端：ByteDance Azure
- 结果文件：[compare_result.json](/Users/bytedance/Documents/code/SkillsPolisher/results/gpt-4o-2024-11-20/skillsbench/compare_result.json)
- 统计文件：[results.csv](/Users/bytedance/Documents/code/SkillsPolisher/statistics/results.csv)

标准运行命令：

```bash
python eval_pipeline/run_compare.py --dataset skillsbench --model_name gpt-4-o
```

## 当前评测框架状态

在本轮真实实验之前，已经完成以下框架修复：

1. 修复 `skillsbench` 的评测器问题
- 原先通用 LLM judge 会把长脚本严重截断，导致参考答案都有可能被判错。
- 现在 `skillsbench` 使用专用 evaluator，支持更长上下文，并保留脚本首尾关键信息。

2. 修复 `skillsbench` 的 train/test 泄漏
- 仓库内没有官方 `train split`。
- 现在 `skillsbench` 只保留 `test` split，不再把同一批任务同时当作 train 和 test。

3. 修复 few-shot demo 来源
- 原先 `demo_bank` 被错误映射为 `instruction -> outcome`，其中 `outcome` 基本只是 `"1"`，并不是脚本。
- 现在已经改成 `instruction -> invocation`，few-shot baseline 真正看到的是“题目 -> 可执行脚本”的映射。
- 对 `skillsbench` 而言，few-shot demo pool 现在来自 `demo_bank`，并且会排除与当前测试题同名的 demo，避免同题泄漏。

## 本轮真实实验

### 实验配置

- 日期：2026-04-23
- 模型：`gpt-4-o`
- 后端：ByteDance Azure
- 数据集：`skillsbench`
- 样本数：`10`
- 随机种子：`42`
- Baseline：`zero_shot`、`random_kshot`、`case_bandit`、`bpo_rewrite`
- 运行命令：

```bash
python eval_pipeline/run_compare.py --dataset skillsbench --model_name gpt-4-o --rerun
```

### 本轮覆盖的 10 个任务

1. `citation-check`
2. `econ-detrending-correlation`
3. `edit-pdf`
4. `enterprise-information-search`
5. `flood-risk-analysis`
6. `gravitational-wave-detection`
7. `jax-computing-basics`
8. `shock-analysis-demand`
9. `trend-anomaly-causal-inference`
10. `weighted-gdp-calc`

### 结果总表

| 方法 | 正确率 | 平均得分 | Token 总量 | 调用次数 | 耗时 |
|---|---:|---:|---:|---:|---:|
| `zero_shot` | 10.0% | 0.100 | 69,492 | 20 | 83.0s |
| `random_kshot` | 0.0% | 0.000 | 113,239 | 20 | 105.3s |
| `case_bandit` | 10.0% | 0.100 | 100,553 | 20 | 98.3s |
| `bpo_rewrite` | 0.0% | 0.000 | 61,716 | 20 | 56.0s |

### 做对的任务

1. `zero_shot`
- 做对：`skillsbench_flood-risk-analysis`
- 使用的 skills：`flood-detection`、`nws-flood-thresholds`、`usgs-data-download`
- 说明：这是当前最直接的“模型读到技能文档后独立完成任务”的成功案例。

2. `case_bandit`
- 做对：`skillsbench_gravitational-wave-detection`
- 选到的 demo：`skillsbench_mars-clouds-clustering`、`skillsbench_jpg-ocr-stat`
- 使用的 skills：`custom-distance-metrics`、`image-ocr`
- 说明：虽然选到的 skill 看起来不完全直观，但 bandit 版本确实在本轮中拿到了一题正确。

3. `random_kshot`
- 本轮未做对任何题目。

4. `bpo_rewrite`
- 本轮未做对任何题目。

## 本轮核心结论

### 1. 框架修复后，结果已经明显比之前可信

此前多轮 `0/10` 的结果，混杂了评测器截断、参考答案误判和 train/test 泄漏等框架问题。  
本轮是在修复后的框架上跑出的真实结果，因此可以作为当前阶段的主要依据。

### 2. `zero_shot` 与 `case_bandit` 目前并列最好

从本轮结果看：

- `zero_shot = 1/10`
- `case_bandit = 1/10`
- `random_kshot = 0/10`
- `bpo_rewrite = 0/10`

这意味着：

- 简单 zero-shot 仍然是很强的基线
- bandit 选择在当前设置下至少没有退化
- 随机 few-shot 目前没有带来正收益

### 3. “有 demo pool” 不等于 “few-shot 有帮助”

这次已经确认：

- `skillsbench` 的 few-shot baseline 确实拿到了非泄漏 demo
- demo 也不再是错误的 `"1"`，而是真正的脚本型 `invocation`

但即便如此，`random_kshot` 仍然是 `0/10`，而且 token 消耗最高。  
因此当前问题不在“有没有 demo”，而在“demo 是否匹配当前任务、是否真正有助于推理闭合”。

### 4. `case_bandit` 比 `random_kshot` 更有研究价值

虽然 `case_bandit` 只做到 `1/10`，但它至少显示出：

- 在外部 demo pool 接入后，选择策略是有机会带来增益的
- 相比完全随机抽 demo，bandit 更值得继续优化

当前研究重点应从“继续扩大 demo 数量”转向“提高 demo 选择质量”。

### 5. `bpo_rewrite` 仍然更像低成本弱基线

`bpo_rewrite` 的特点目前很清楚：

- token 更省
- 速度更快
- 准确率没有起色

所以它现在更适合作为“低成本 prompt 改写基线”，而不是主要改进方向。

## 对失败题目的理解

从这 10 个任务看，失败题目仍主要集中在三类：

1. 需要完整文件处理链路的任务
- 如 `citation-check`、`edit-pdf`、`enterprise-information-search`
- 常见问题是脚本不闭合，或缺少最终产物写出逻辑。

2. 需要较强数值与数据处理闭环的任务
- 如 `econ-detrending-correlation`、`shock-analysis-demand`、`weighted-gdp-calc`
- 常见问题是会描述流程，但没有把完整分析落实为最终可执行实现。

3. 需要专业工具链和较长脚本的任务
- 如 `jax-computing-basics`、`trend-anomaly-causal-inference`
- 常见问题是中间步骤合理，但整体交付仍然不完整。

## 下一步实验建议

### 优先级 1：增强脚本级自检与修复

当前最值得做的不是继续加大 demo 数量，而是加一层“脚本是否真的完成任务”的自检。

建议在最终返回前检查：

- 是否是完整 bash 脚本
- 是否包含关键输出文件写出逻辑
- 是否还有说明性文本或未完成片段
- 是否遗漏关键依赖步骤

若不满足要求，则自动做一次 repair。

### 优先级 2：优化 `case_bandit` 的 demo 选择信号

当前 `case_bandit` 有一定希望，但选到的 demo 仍有明显噪声。  
下一步建议：

- 把 skill 名相似度纳入选择信号
- 把 task 类别、tag、difficulty 纳入检索或打分
- 限制 demo 候选范围，减少明显无关的样例进入 bandit 池

### 优先级 3：不要继续盲目增加随机 k-shot

本轮结果已经说明：

- 随机抽 demo 很贵
- 但没有收益

因此下一轮不建议优先扩大 `k`，也不建议把更多预算投到 `random_kshot` 上。

### 优先级 4：继续保留 10-sample 快速迭代流程

当前研究目标是快速定位瓶颈，因此仍建议：

- 每轮默认跑 `10` 条
- 先验证趋势，再决定是否扩大样本数
- 每轮结束后都同步更新本报告

## 当前阶段结论

截至 2026-04-23，基于修复后的 `skillsbench` 评测框架，可以给出当前最稳妥的判断：

- `gpt-4-o` 在 `skillsbench` 上已经不是“完全不会做”
- 但整体成功率仍然很低
- `zero_shot` 和 `case_bandit` 暂时并列最佳
- `random_kshot` 当前无效且成本最高
- 下一阶段最值得投入的是：
  - 脚本自检与修复
  - `case_bandit` 的 demo 选择优化

## 补充实验：`CODEX-GPT-5.4` 子代理 `demo_test`

为避免子代理批测只停留在临时文件中，这里将其登记为一次正式实验。

### 实验定义

- 模型名：`CODEX-GPT-5.4`
- 数据集名：`demo_test`
- baseline：`codex_subagent`
- 题目集合：与当前 `skillsbench` 的同一组 10-sample 完全一致
- 评测方式：使用仓库当前 `SkillsBenchEvaluator` 的 LLM judge 口径

结果文件：

- [compare_result.json](/Users/bytedance/Documents/code/SkillsPolisher/results/CODEX-GPT-5.4/demo_test/compare_result.json)
- [codex_subagent_result.json](/Users/bytedance/Documents/code/SkillsPolisher/results/CODEX-GPT-5.4/demo_test/codex_subagent_result.json)

### 结果总表

| 方法 | 正确率 | 平均得分 | 说明 |
|---|---:|---:|---|
| `codex_subagent` | 70.0% | 0.700 | 子代理逐题构造候选脚本，再由当前 evaluator 判分 |

### 逐题结果

判对：

- `citation-check`
- `econ-detrending-correlation`
- `edit-pdf`
- `gravitational-wave-detection`
- `jax-computing-basics`
- `trend-anomaly-causal-inference`
- `weighted-gdp-calc`

判错：

- `enterprise-information-search`
- `flood-risk-analysis`
- `shock-analysis-demand`

### 这组结果说明什么

1. 当前框架已经能区分模型强弱
- 在同一口径下，`gpt-4-o` 的最好结果是 `1/10`
- `CODEX-GPT-5.4` 子代理可以达到 `7/10`
- 因此当前 benchmark 与 evaluator 已经具备基本辨别力

2. `skillsbench` 不是“天然跑不通”的 benchmark
- 更强的 Codex 型模型在这个框架下确实能做对多数题目
- 所以当前主要矛盾不再是框架是否可信，而是不同模型之间的脚本完成度差距

3. 当前 `gpt-4-o` 的瓶颈更清楚了
- 它不是完全不会
- 但和更强的 Codex 型代理相比，差距主要体现在：
  - 任务闭合能力
  - 长脚本完整性
  - 从技能文档到最终交付脚本的稳定映射能力

4. 当前 judge 仍然是 LLM 口径
- 这组 `7/10` 表示“在当前 evaluator 标准下被判对”
- 并不等价于真实容器执行通过率
- 但已经足以证明当前评测流程有研究价值

### 补充结论

加入这组 `CODEX-GPT-5.4` 的对照后，当前整体判断可以进一步收敛为：

- `skillsbench` 适合考察技能理解、长上下文整合和脚本交付能力
- `gpt-4-o` 当前表现偏弱，主要问题是脚本闭合和完整执行链
- 更强的 Codex 型模型在同一框架下显著更强
- 后续如果要继续研究提升路线，应该优先思考如何缩小 `gpt-4-o` 与强代理之间的行为差距

## 补充实验：`CODEX-GPT-5.4 + CASE`

为了判断 CASE 对强代理是否有帮助，又补做了一组 `CODEX-GPT-5.4 + CASE` 的对照实验。

### 实验定义

- 模型名：`CODEX-GPT-5.4`
- 数据集名：`demo_test_case`
- baseline：`codex_case_subagent`
- 题目集合：与之前 `demo_test` 完全相同
- 选例方式：参考当前仓库 `case_bandit` 的 CASE 风格 demo 选择
- 评测方式：仍使用当前 `SkillsBenchEvaluator`

结果文件：

- [compare_result.json](/Users/bytedance/Documents/code/SkillsPolisher/results/CODEX-GPT-5.4/demo_test_case/compare_result.json)
- [codex_case_subagent_result.json](/Users/bytedance/Documents/code/SkillsPolisher/results/CODEX-GPT-5.4/demo_test_case/codex_case_subagent_result.json)

### 结果总表

| 方法 | 正确率 | 平均得分 | 与无 CASE 对比 |
|---|---:|---:|---|
| `codex_subagent` | 70.0% | 0.700 | 基线 |
| `codex_case_subagent` | 30.0% | 0.300 | 明显下降 |

### 逐题变化

提升：

- `flood-risk-analysis`

持平：

- `econ-detrending-correlation`
- `enterprise-information-search`
- `shock-analysis-demand`
- `weighted-gdp-calc`

退化：

- `citation-check`
- `edit-pdf`
- `gravitational-wave-detection`
- `jax-computing-basics`
- `trend-anomaly-causal-inference`

### 为什么 CASE 会拖后腿

根据 `case_selected_demos.json`，当前问题非常具体：

1. 选到的 demo 经常和目标技能明显不匹配
- `citation-check` 被配到 `image-ocr`、`civ6lib`
- `edit-pdf` 被配到 `audio-extractor`、`image-ocr`
- `trend-anomaly-causal-inference` 被配到 `pdf`、`d3-visualization`、`dc-power-flow`

2. 当前 CASE 的 reward 太弱
- 仓库里的 `_quick_eval()` 本质上只是词重叠加上一点 metadata bonus
- 它没有真正判断“这个 demo 的 skill 结构是否与目标任务匹配”

3. 对强代理而言，错误 demo 比没有 demo 更糟
- `CODEX-GPT-5.4` 无 CASE 时已经能自己组织较完整的脚本
- 加入噪声 demo 后，反而会把它的结构学习带偏

### 当前结论

这组结果说明：

- 现在不是“强模型也需要 CASE”
- 而是“当前 CASE 选例质量不足，正在污染推理”

因此如果继续做 CASE，下一步的重点必须是：

- 提升 demo 候选过滤质量
- 强化 skill 对齐信号
- 先减少明显无关 demo，再讨论 bandit 学习
