# SkillsPolisher IDEA 与研究进度

日期：2026-04-23

## 当前研究目标

当前工作的核心目标不是做一次最终大规模评测，而是先把 `skillsbench` 上的研究闭环跑通，快速判断：

- 当前评测框架是否可信
- 哪类 baseline 在当前设置下更有潜力
- 模型失败的主要原因到底是框架问题、prompt 问题，还是方法本身的问题

因此目前采用的是“小样本、快速迭代”的策略：

- 默认每轮跑 `10` 条样本
- 模型固定为 `gpt-4-o`
- 数据集固定为 `skillsbench`
- 每次改完关键逻辑后立即重跑并记录结论

## 目前已经做了什么

### 1. 统一了实验输出与记录流程

已经完成的基础设施包括：

- 实验结果统一保存到 `./results/{model_name}/{dataset_name}/`
- 对比实验写到 `compare_result.json`
- 各 baseline 单独写到 `{baseline}_result.json`
- 核心统计指标写到 `./statistics/results.csv`
- 研究性总结统一写到 `./report/report.md`

这一步的意义是：

- 后续实验可复现
- 能追踪每次实验配置和结果
- 方便后面整理论文表格和实验对比

### 2. 修复了 `skillsbench` 的评测框架问题

这是当前最重要的一步，因为早期结论被框架错误严重污染。

已经确认并修复的问题有：

1. `skillsbench` evaluator 不适合长脚本
- 旧版通用 judge 会把长答案严重截断
- 导致参考答案自己都可能被判错
- 现在已替换为 `skillsbench` 专用 evaluator

2. `skillsbench` 存在 train/test 泄漏
- 仓库里并没有真正官方 `train split`
- 旧逻辑却把同一批任务同时当作 train 和 test
- 现在已经明确：`skillsbench` 只保留 `test`

3. few-shot demo 来源映射错误
- 旧逻辑把 `demo_bank` 的 `answer` 错映射成 `outcome`
- 实际 `outcome` 往往只是 `"1"`，根本不是可用 demo
- 现在已改成使用 `invocation` 作为 few-shot demo 的答案脚本

当前可以明确判断：

- 之前多轮 `0/10` 不能直接拿来评价模型或 baseline
- 修复后的实验结果才是当前可信依据

### 3. 强化了 `skillsbench` 的生成设置

为了让模型更接近真实任务交付形态，已经做过以下调整：

- 提高 `skillsbench` 的生成预算
- 对脚本任务增加“只返回可执行脚本”的约束
- 对输出做基础格式修正和 repair 尝试

这一步已经确认的结论是：

- 单纯提高 `max_tokens` 并不能自动带来显著准确率提升
- 输出格式更像脚本了，但任务逻辑仍然常常不闭合

### 4. 为 `skillsbench` 建立了真正不泄漏的 demo pool

这一步是为了让 `random_kshot` / `case_bandit` / `bpo_rewrite` 真正具备 few-shot 意义。

当前做法：

- 使用 `demo_bank` 作为外部 demo 来源
- 将每条 demo 映射成 `instruction -> invocation`
- 保留任务 id，用于排除与当前测试题同名的 demo

当前可以确认：

- few-shot baseline 现在确实拿到了脚本型 demo
- 不再是伪 few-shot
- 但“拿到 demo”本身不等于“few-shot 有效果”

### 5. 引入 `CODEX-GPT-5.4` 作为强代理对照

为了确认当前框架能否区分强弱模型，已经补做了一轮正式对照实验：

- 模型名：`CODEX-GPT-5.4`
- 数据集名：`demo_test`
- baseline：`codex_subagent`
- 题目集合：与当前 `skillsbench` 的 10-sample 完全一致

结果：

- `CODEX-GPT-5.4 / demo_test / codex_subagent = 7/10`

这一点很关键，因为它说明：

- 当前 `skillsbench + evaluator` 框架已经不是“天然跑不通”
- 这个 benchmark 确实能区分不同层级模型的能力
- 当前 `gpt-4-o` 表现差，核心不再是框架问题，而更像是模型和策略本身的问题

### 6. 测试 `CODEX-GPT-5.4 + CASE`，确认当前 CASE 方案对强代理无益

为了判断 CASE 是否能帮助强代理，又补做了一轮：

- `CODEX-GPT-5.4 / demo_test_case / codex_case_subagent = 3/10`

和无 CASE 的 `7/10` 相比，明显下降。  
这说明当前 CASE 方案并没有起到“帮助强模型更好利用 demo”的作用，反而会把推理带偏。

## 当前 baseline 的阶段性结论

基于修复后的真实实验：

命令：

```bash
python eval_pipeline/run_compare.py --dataset skillsbench --model_name gpt-4-o --rerun
```

结果：

| baseline | 正确率 | 平均得分 | token | 结论 |
|---|---:|---:|---:|---|
| `zero_shot` | 10.0% | 0.100 | 69,492 | 当前强基线之一 |
| `random_kshot` | 0.0% | 0.000 | 113,239 | 成本最高，当前无收益 |
| `case_bandit` | 10.0% | 0.100 | 100,553 | 有继续优化价值 |
| `bpo_rewrite` | 0.0% | 0.000 | 61,716 | 成本低，但效果弱 |

强代理对照：

| baseline | 正确率 | 平均得分 | 结论 |
|---|---:|---:|---|
| `codex_subagent` | 70.0% | 0.700 | 当前框架下的强代理上限参考 |

### 对各 baseline 的理解

1. `zero_shot`
- 当前依然非常重要。
- 在 few-shot 方案尚不成熟时，zero-shot 是最稳定、最干净的参考基线。
- 当前已经能在 `skillsbench` 上做对部分任务，说明模型并非完全无能力。

2. `random_kshot`
- 当前结论比较明确：不值得优先投入。
- 它已经接上了正确的外部 demo pool，但结果仍然是 `0/10`。
- 说明“随机抽 demo”本身很可能会引入噪声，而不是帮助模型。

3. `case_bandit`
- 当前最值得继续打磨的 baseline。
- 它至少与 `zero_shot` 持平，并且理论上最有可能从 demo pool 中获益。
- 但当前选 demo 的质量还不够稳定，仍有明显噪声。

这里要特别说明 CASE 的方案是什么、设计上想解决什么问题：

- CASE 不是随机 few-shot，而是一个“案例选择”方案。
- 当前仓库中的 `case_bandit` 实现，本质上是：
  1. 从外部 `demo_bank` 中拿到 demo pool
  2. 把每条 demo 当成一个 arm
  3. 用 bandit 风格方法，为当前任务挑出 top-k demo
  4. 把这些 demo 放进 prompt，当作 few-shot 参考

它设计上的作用本来应该是：

- 替代随机抽样
- 提高 demo 和当前任务的相关性
- 帮模型更快学到正确的解题结构
- 降低无关案例带来的 prompt 噪声

但当前实际表现说明：

- 这个目标在当前实现里没有真正达到
- 现阶段它更像“案例污染”，而不是“案例增强”

4. `bpo_rewrite`
- 当前更适合保留为轻量、低成本对照组。
- 它速度快、token 低，但没有展现出明显准确率提升。
- 暂时不应把主要研究资源放在这一条线上。

## 当前已经成立的结论

下面这些判断，当前已经可以视为阶段性成立：

1. `skillsbench` 上最早那批全错结果包含明显框架污染
- 后续分析必须以修复后的实验为准。

2. `gpt-4-o` 不是“完全不会做 `skillsbench`”
- 修复框架后已经能做对部分题。
- 当前问题更像“低成功率”而不是“完全无能力”。

3. few-shot 的关键不在“有没有 demo”，而在“demo 是否匹配”
- `random_kshot` 已经证明：有 demo 但随机选，效果仍可能很差。

4. `case_bandit` 比 `random_kshot` 更值得继续研究
- 当前至少没有退化
- 并且更符合“技能选择/案例选择”这个项目的核心方向

这个判断现在要更精细地理解：

- 对弱模型（如当前 `gpt-4-o`）而言，`case_bandit` 至少没有比随机 few-shot 更差
- 但对强代理（`CODEX-GPT-5.4`）而言，当前 CASE 版本已经证明会明显退化

所以更准确的说法应该是：

- CASE 这条研究路线仍值得研究
- 但当前仓库里的 CASE 实现还远远不够好
- 现在应该优化“CASE 的质量”，而不是默认“用了 CASE 就会更强”

5. 当前主瓶颈仍然是任务闭合能力
- 很多失败并不是方向完全错了
- 而是脚本不完整、产物缺失、执行链没有真正闭环

6. 当前框架已经能够明显区分 `gpt-4-o` 和更强的 Codex 型代理
- `gpt-4-o` 当前最好只有 `1/10`
- `CODEX-GPT-5.4` 子代理能达到 `7/10`
- 因此后续优化的目标非常明确：缩小当前 pipeline 与强代理之间的行为差距

## 为什么当前模型效果不好

这是当前最重要的分析问题。基于现有结果，可以先排除几个错误解释，再聚焦真正的问题。

### 先排除的解释

1. 不是“模型没看到技能”
- 对 `skillsbench` 而言，任务对应的 `SKILL.md` 全文已经进入 `sample.context`
- `zero_shot` 下逐题记录的 `used_skills` 与真实任务技能是对齐的
- 例如：
  - `citation-check` 看到了 `citation-management`
  - `flood-risk-analysis` 看到了 `flood-detection`、`nws-flood-thresholds`、`usgs-data-download`
  - `trend-anomaly-causal-inference` 看到了 4 个相关技能

所以当前不是“技能没给到模型”，而是“给到了，但没稳定转化成完整解法”。

2. 不是“框架天然判不对”
- `CODEX-GPT-5.4` 在同样的 10 题上能做到 `7/10`
- 因此当前 evaluator 至少已经能接受高质量脚本型答案

### 真正的问题更可能是什么

#### 问题 1：`skillsbench` 考的是技能整合 + 脚本闭合，不是普通问答

`skillsbench` 的任务结构决定了它对模型的要求比普通 benchmark 高很多：

- `instruction` 往往很长
- `context` 是多个技能文档全文拼接
- 参考答案本身经常就是长脚本

也就是说，它不是在问“你懂不懂这个概念”，而是在问：

- 你能否从技能文档中抽取出正确操作流程
- 你能否把多步工具链串起来
- 你能否生成一个真正可交付的最终脚本

这对 `gpt-4-o` 来说，难点更像“复杂 agent 交付”，而不是单轮语言理解。

#### 问题 2：任务长度和技能长度本身就很重

当前 10 题里，有几类任务的文本规模非常大：

- `citation-check`: `context_len = 33441`
- `trend-anomaly-causal-inference`: `question_len = 5109`, `context_len = 21370`, `answer_len = 27999`
- `gravitational-wave-detection`: `context_len = 11831`
- `weighted-gdp-calc`: `context_len = 10642`

这意味着模型必须在很长的说明和技能文档里筛出真正关键的部分。  
如果筛选失败，就很容易出现：

- 只抓到高层思路
- 漏掉关键输出步骤
- 没有把多个技能真正串成端到端流水线

#### 问题 3：`gpt-4-o` 当前更像“会解释”，但不够像“会交付”

从结果看，`gpt-4-o` 的典型特征不是完全答非所问，而是：

- 技能方向经常是对的
- 问题也大致理解了
- 但最后交出的脚本经常不闭合

这和 `skillsbench` 的核心要求是直接冲突的。  
因为这个 benchmark 看重的是：

- 输出文件有没有真正写出来
- 数据处理链是否完整
- 工具调用是不是端到端闭环

换句话说，当前 `gpt-4-o` 的失败很多不是“认知失败”，而是“交付失败”。

#### 问题 4：few-shot baseline 目前没有建立起正确帮助

虽然现在已经修正了 demo pool，但现状仍然说明：

- `random_kshot` 的 demo 大量不相关
- `case_bandit` 的选例仍然噪声较大
- `bpo_rewrite` 的改写也没有把脚本闭合问题解决掉

这意味着当前 pipeline 还没能形成一种稳定机制，帮助 `gpt-4-o` 从“知道大概怎么做”提升到“稳定交付完整脚本”。

## `skillsbench` 里提到的内容，与当前问题是什么关系

这个问题的答案很重要：`skillsbench` 提供的技能不是“背景知识”，而是“任务完成规范”。

### 在这个 benchmark 里，`SKILL.md` 的作用不是装饰性的

很多任务会明确依赖某个具体技能，例如：

- `citation-check` -> `citation-management`
- `edit-pdf` -> `pdf-editing`, `text-parser`
- `flood-risk-analysis` -> `flood-detection`, `nws-flood-thresholds`, `usgs-data-download`
- `gravitational-wave-detection` -> `conditioning`, `matched-filtering`
- `trend-anomaly-causal-inference` -> `data_cleaning`, `did_causal_analysis`, `feature_engineering`, `time_series_anomaly_detection`

这说明：

- 技能文档本身就是解题材料的一部分
- 模型必须真的“使用”这些内容，而不是只把它们当成可忽略背景

### 当前 `gpt-4-o` 的问题不是完全没用技能，而是没有把技能转成稳定程序结构

从 `zero_shot` 的 `used_skills` 记录看，它确实已经识别到了正确技能。  
但问题在于：

- 技能没有完全展开为可执行步骤
- 多个技能之间没有形成稳定的调用顺序
- 最后没有落实到完整脚本产物

所以当前问题可以概括为：

> 它知道“应该用哪些 skills”，但还做不到稳定地把这些 skills 编排成一个能交付结果的脚本。

## 为什么当前 CASE 方案会失效

这个问题现在也已经有比较明确的证据了。

### 当前 CASE 方案是什么

当前仓库中的 `case_bandit`，本质上是：

1. 从 `demo_bank` 中构造一个外部 demo pool
2. 把每条 demo 当成一条 bandit arm
3. 对当前任务和 demo 之间做一个非常轻量的相关性打分
4. 选出 top-k demo，拼进 prompt，当作 few-shot 参考

理论上的作用是：

- 用“更相关”的案例替代随机抽样
- 帮模型更快学到正确解题结构
- 减少 irrelevant demo 对 prompt 的污染

### 为什么它现在会失效

从 `CODEX-GPT-5.4 + CASE` 的结果和 `case_selected_demos.json` 看，当前失效原因很具体：

1. 选例信号过弱
- 当前 `_quick_eval()` 主要是词重叠 + 少量 metadata bonus
- 这不足以识别“技能结构是否匹配”

2. 技能对齐没有真正进入核心打分
- 例如 `citation-check` 这种题，本应偏向 `citation-management`
- 结果却选到了 `image-ocr`、`civ6lib`

3. 对强代理来说，错误案例比没有案例更危险
- 强代理本来可以自己组织解题结构
- 错误 demo 进入 prompt 后，会把结构学习带偏

所以当前 CASE 的真实效果不是“帮助推理”，而是“给推理注入噪声”。

### 这也是为什么 `CODEX-GPT-5.4` 对照很关键

强代理在同样任务上能显著更好，说明：

- `skillsbench` 的技能内容本身是足够支撑解题的
- 这些 `SKILL.md` 不是无用信息
- 差距主要在于模型是否能把这些技能真正编排成程序化执行方案

## 当前新的研究判断

在加入 `CODEX-GPT-5.4` 对照后，当前最重要的判断已经更清楚了：

1. `skillsbench` 的核心挑战是“技能到脚本”的映射能力
- 而不是简单语言理解

2. `gpt-4-o` 当前表现差，主要是因为：
- 长上下文技能整合不稳定
- 多步骤任务闭合能力弱
- pipeline 还没有提供足够强的结构化辅助

3. 当前最有价值的后续方向不是继续堆更多 demo
- 而是想办法让模型更像强代理那样，把技能文档转成稳定的程序结构

## 当前不建议优先做的事

为了避免研究发散，下面这些方向当前不建议优先投入：

1. 继续盲目加大 `random_kshot` 的 `k`
- 当前随机 demo 本身没有证明有效
- 继续加大会先增加成本，再放大噪声

2. 直接扩大到全量样本评测
- 当前方法还在快速迭代期
- 先在 `10` 条上把趋势跑清楚更重要

3. 过早下结论说某个 baseline “无效”
- 尤其是 `case_bandit`
- 现在更像“检索/选择信号不够好”，而不是这条路线本身完全没价值

## 当前最值得继续做的方向

### 方向 1：脚本级自检与修复

这是当前最有希望直接提高成功率的方向。

目标：

- 检查输出是不是完整 bash 脚本
- 检查有没有遗漏关键文件写出逻辑
- 检查有没有说明性文本、TODO、半截实现
- 不满足则自动做一轮 repair

理由：

- 当前大量失败都表现为“思路对，但交付不闭合”

### 方向 2：优化 `case_bandit` 的 demo 选择质量

目标：

- 让 bandit 不只是“从全池里探索”
- 而是先用更合理的信号缩小候选范围

可尝试信号：

- skill 名相似度
- task category
- task tags
- difficulty
- 指令文本相似度

理由：

- 当前 `case_bandit` 已经有一点效果
- 如果把 demo 噪声压下去，这条线最可能继续涨

### 方向 3：继续保持 10-sample 快速迭代

当前研究节奏应该维持：

1. 改一项关键逻辑
2. 跑一轮 `10` 条
3. 更新 `report/report.md`
4. 再判断下一步

这比一次性大规模跑实验更适合当前阶段。

## 当前进度总结

如果用一句话概括当前进展：

> 我们已经把 `skillsbench` 的评测框架修正到基本可信，并确认当前最有潜力继续优化的方向是 `case_bandit` 的案例选择，以及脚本级自检/修复，而不是随机 few-shot。

当前项目状态可以概括为：

- 基础设施：已基本稳定
- 评测框架：已修复到可用于研究
- baseline 对比：已有初步可信结果
- 当前最优方向：`zero_shot` 作为强基线，`case_bandit` 作为主要改进线
- 下一阶段重点：提高任务交付闭合率，并重做 CASE 的选例质量，而不是单纯扩大提示或 demo 数量
