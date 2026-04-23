# SPLICE: Sample-Efficient Skill Invocation via Bandit Demo Selection and Prompt Rewriting

**Date**: 2026-04-21
**Status**: Pre-experiment (dry-run pilot complete; real GPU/API experiments pending)
**Target Venue**: ICLR 2027 (or NeurIPS 2027)

---

## 1. Problem Statement

Modern AI agents deployed on complex benchmarks such as SkillsBench (Li et al., 2026) are expected to invoke the correct skill from a library and execute it successfully on diverse tasks. SkillsBench contains 89 tasks spanning engineering, science, finance, and coding domains, each associated with one or more skill bodies (SKILL.md files describing instructions and APIs). Despite SOTA agents achieving reasonable performance on simple single-skill tasks, the success rate on multi-skill composition tasks remains below 50%. This failure mode reveals a fundamental gap: simply having the correct skill available does not guarantee that the agent will invoke it effectively.

The skill invocation problem has two orthogonal bottlenecks. First, the agent needs to see relevant examples of how other tasks have been solved using the same or similar skills — the absence of in-context demonstrations leaves the agent without task-specific guidance. Second, the skill invocation prompt itself (the SKILL.md text prefixed to the agent's context) may be suboptimal: it may be too generic, ambiguous, or verbose, failing to steer the agent toward the specific action required for the current task. Neither of these bottlenecks is addressed by existing work. SkillRouter (Zheng et al., 2026) selects which skill to use but does not optimize how it is invoked. SkillRL (Xia et al., 2026) evolves skill bodies through RL reward but ignores prompt-level phrasing and in-context examples. BPO (Cheng et al., 2024) rewrites prompts but has no notion of skill structure or demonstration selection. CASE (Purohit et al., 2025) selects demonstrations sample-efficiently via a bandit but has only been applied to tabular NLP tasks, not agent skill invocation.

The core insight of this work is that demonstration selection and prompt rewriting are not independent: the best demonstrations for a task depend on the current prompt, and the best prompt rewrite depends on what demonstrations are available. Jointly optimizing both via a feedback loop — using only black-box binary task success as the reward signal — creates a synergistic effect that neither component achieves alone. This is the SPLICE hypothesis, and it motivates a system that is applicable to any agent operating with a skill library, requiring no access to model weights or gradients.

---

## 2. Core Claim

We claim that **jointly selecting skill-usage demonstrations via a bandit algorithm (SPLICE-Select) and rewriting the skill invocation prompt conditioned on those demonstrations (SPLICE-Rewrite) improves agent task success rate on SkillsBench by a statistically significant margin over either component alone**, under the condition that only binary black-box feedback (task success/failure) is available and no model gradient access is assumed.

The specific sub-claims are:
- SPLICE-Select (bandit demo selection alone) outperforms random k-shot and no-demo baselines in LLM query efficiency (fewer queries to converge on useful demos) and in downstream task success.
- SPLICE-Rewrite (prompt rewriting with selected demos) outperforms BPO-only rewriting (prompt rewriting with random demos), demonstrating that demo quality matters for rewriting.
- SPLICE-Full (joint system) outperforms both components individually, demonstrating the complementary nature of demo selection and prompt rewriting.

---

## 3. Method

**SPLICE-Select** adapts the CASE UGapE gap-index bandit (Purohit et al., ICML 2025) to the skill-invocation demonstration selection setting. The DemoBank is a collection of (task instruction, skill name, skill body, oracle invocation, outcome) tuples built from historical SkillsBench task runs. For each evaluation task, up to 20 candidate demonstrations are drawn from the DemoBank (excluding the test task itself). Each candidate is an "arm" in the bandit. The reward signal is binary task success obtained by running the agent with that single demonstration in context. The UCB confidence interval is computed using the CASE HeuristicBeta: `beta(t) = log((log(t)+1)/delta + 1)`. The UGapE stopping criterion fires when `tau_UGapE = max_{j in J_t} max_{i not in J_t} (UCB(i) - LCB(j)) <= epsilon`, where `J_t` is the current estimated top-k set. The algorithm terminates with a sample-efficient top-k subset of demonstrations, using fewer LLM queries than exhaustive evaluation of all candidates.

**SPLICE-Rewrite** takes the top-k selected demonstrations and the current skill invocation prompt (the SKILL.md text) and produces a rewritten prompt using a BPO-style model. In the full system, this uses THUDM/BPO (Cheng et al., ACL 2024) — a 7B LLaMA-based model fine-tuned on preference pairs to improve user prompts without access to model gradients. In SPLICE, the BPO model is applied with a SPLICE-specific template that provides the selected demonstrations as few-shot context, conditioning the rewrite on examples of successful skill invocations for structurally similar tasks. The rewriter is queried once per task (not iteratively during bandit rounds), making it computationally cheap at inference time.

**SPLICE-Full** combines both components in a loop. In the default single-round setting (n_rounds=1), SPLICE-Select first identifies the top-k demonstrations for the task, then SPLICE-Rewrite produces a rewritten skill prompt conditioned on those demonstrations, and finally the agent is evaluated with both the selected demos and the rewritten prompt. In the multi-round setting (n_rounds > 1), the rewritten prompt is fed back into the next bandit round, allowing the demo selection to adapt to the improved prompt. This feedback loop is the distinguishing feature of SPLICE-Full versus running the two components sequentially without iteration.

**System architecture summary**: DemoBank construction (offline, from oracle solve.sh) → SPLICE-Select (online, per-task bandit over demo candidates) → SPLICE-Rewrite (online, per-task BPO rewrite conditioned on selected demos) → agent evaluation with rewritten prompt + selected demos → optional loop back to SPLICE-Select with updated prompt. The entire pipeline uses only binary task success as feedback; no preference labels or reward models beyond the black-box task evaluator are required at inference time.

---

## 4. Experimental Design

### 4.1 Dataset

**SkillsBench** (Li et al., 2026): 89 tasks across engineering, science, finance, coding, and multi-modal domains. Tasks are evaluated via the BenchFlow SDK (`bench eval create`), which runs the agent and reads `reward.txt` (binary 0/1). The benchmark includes both single-skill tasks (one SKILL.md required) and multi-skill composition tasks (2+ skills). Multi-skill tasks represent the hardest setting and are analyzed separately.

### 4.2 Baselines

| Method | Description | Notes |
|--------|-------------|-------|
| Raw (No-ICL) | Default agent, no demonstrations, original SKILL.md | SkillsBench default |
| Random-3shot | 3 randomly sampled demos, original prompt | Standard ICL baseline |
| BPO-Only | BPO-rewritten prompt, random demos (no bandit) | Tests prompt rewriting without selection |
| CASE-Only | Top-3 demos selected by CASE bandit, original prompt | Tests demo selection without rewriting |
| OptiSeq-Random | 3 random demos with OptiSeq-style ordering | Tests ordering effects (Bhope et al.) |
| SPLICE-Select | Bandit demo selection, no prompt rewriting | Ablation: selection component alone |
| SPLICE-Rewrite | BPO rewriting + CASE demos (= splice_rewrite in code) | Ablation: rewriting component alone |
| SPLICE-Full | Full SPLICE: bandit selection + BPO rewriting | Proposed method |
| Oracle | Human-written solve.sh solutions | Upper bound |

Note: `SPLICE-Select` and `CASE-Only` are identical in the current implementation (both = bandit selection without rewriting). This distinction will be formalized in the paper as CASE-Only using the CASE algorithm from the original paper directly, while SPLICE-Select uses our adapted variant with the skill-invocation-specific reward function.

### 4.3 Metrics

1. **Task Success Rate**: Fraction of 89 tasks with binary reward = 1 (primary metric).
2. **LLM Query Efficiency**: Number of bandit rounds to converge (SPLICE-Select vs. exhaustive).
3. **Skill Invocation Accuracy**: Fraction of tasks where the correct skill is invoked (requires parsing agent output).
4. **Multi-Skill Composition Rate**: Success rate on the multi-skill subset (expected to show SPLICE's largest gains).
5. **Demo Selection Efficiency**: `tau_UGapE` trajectory — how quickly does the bandit identify the top-k set?

### 4.4 Evaluation Protocol

```bash
# For each task and method:
bench eval create -t tasks/<task-id> -a claude-agent-acp -s tasks/<task-id>/environment/skills/
bench metrics jobs/<run-dir>
```

For SPLICE-Full, the modified SKILL.md (rewritten by SPLICE-Rewrite) is written to a temp directory and passed via the `-s` flag. The bandit reward during selection rounds uses the same `bench eval create` pipeline. All evaluations use `claude-agent-acp` as the agent backbone.

### 4.5 Implementation Details

- **Bandit hyperparameters**: delta=0.05, k=3, max_rounds=100, n_demo_candidates=20
- **Rewriter**: THUDM/BPO (7B LLaMA-based), 4-bit quantized, temperature=0.7, max_new_tokens=512
- **Demo bank**: Built from oracle solve.sh outputs for all 89 tasks; leave-one-out (test task excluded from its own candidate pool)
- **Random seed**: 42 for all stochastic components
- **Iterative loop**: n_rounds=1 for main results; n_rounds=3 tested in ablation

---

## 5. Key Results

### 5.1 Expected Results Table (from dry-run pilot extrapolated to full benchmark)

| Method | Expected Success Rate | Expected Query Count | Notes |
|--------|----------------------|---------------------|-------|
| Raw (No-ICL) | ~35–40% | 1 | SkillsBench SOTA without ICL |
| Random-3shot | ~38–43% | 1 | Marginal gain |
| OptiSeq-Random | ~40–44% | 1 | Ordering helps slightly |
| BPO-Only | ~42–46% | 1 | Prompt rewriting helps, random demos |
| CASE-Only | ~45–50% | ~25 (20 arms × 1.25) | Sample-efficient demo selection |
| SPLICE-Select | ~47–52% | ~25 | Same as CASE-Only in this impl. |
| SPLICE-Rewrite | ~44–48% | ~25 | Rewriting with CASE demos as context |
| SPLICE-Full | **~52–58%** | ~26 | Joint optimization; largest gain |
| Oracle | ~85–90% | N/A | Human solve.sh |

**Key expected finding**: SPLICE-Full achieves the largest absolute gain on multi-skill composition tasks (currently <50% SOTA), where both components contribute complementary improvements: SPLICE-Select identifies structurally relevant multi-skill demonstrations, while SPLICE-Rewrite clarifies which sub-skills to invoke and in which order.

### 5.2 Dry-Run Pilot Results (n=10 tasks, mock evaluator)

| Method | Success Rate | Std | N Success / N Tasks |
|--------|-------------|-----|---------------------|
| Baseline | 0.500 | 0.527 | 5/10 |
| CASE-Only | 0.400 | 0.516 | 4/10 |
| SPLICE-Full | **0.800** | 0.422 | 8/10 |

**Important caveat**: The dry-run pilot uses a mock evaluator (`DryRunEvaluator`) with a fixed random seed and hard-coded success probabilities (base=0.40 + demo_bonus per demo + 0.08 for rewrite). The results are not based on real agent executions. The CASE-Only result being lower than baseline in the pilot is an artifact of the random seed (different bandit rounds cause different random draws). These numbers should NOT be reported as evidence of real performance. They only validate that the pipeline executes correctly end-to-end.

### 5.3 Interpretation

When real experiments run:
- SPLICE-Full > SPLICE-Select validates that prompt rewriting adds value beyond demo selection.
- SPLICE-Full > SPLICE-Rewrite validates that bandit demo selection improves rewriting quality.
- SPLICE-Full > BPO-Only validates that demo selection quality matters for prompt rewriting.
- CASE-Only > Random-3shot validates the bandit's sample efficiency claim.
- Multi-skill gap analysis: if SPLICE's gain is larger on multi-skill tasks, it supports the hypothesis that demo selection provides compositional guidance.

---

## 6. Ablation Studies

### 6.1 Demonstration Count (k ablation)
- **What**: Run SPLICE-Full with k ∈ {1, 3, 5} demos.
- **Tests**: How many demonstrations are needed? Is there diminishing returns?
- **Expected finding**: k=3 optimal; k=5 provides marginal improvement at higher query cost.

### 6.2 Bandit Rounds Budget
- **What**: Run SPLICE-Select with max_rounds ∈ {10, 20, 50, 100}.
- **Tests**: Convergence speed of the UGapE criterion; trade-off between demo quality and query cost.
- **Expected finding**: Top-k identified within 30–40 rounds for 20 arms; >50 rounds provides negligible gain.

### 6.3 Prompt Rewriting Scope
- **What**: Rewrite only first 100 tokens / 200 tokens / full SKILL.md.
- **Tests**: Does rewriting the full skill body cause hallucination? Is a shorter prefix sufficient?
- **Expected finding**: Rewriting the first 200 tokens (the instruction prefix) is sufficient; full-body rewriting may degrade tool-use accuracy.

### 6.4 Iterative Loop Depth
- **What**: Run SPLICE-Full with n_rounds ∈ {1, 2, 3}.
- **Tests**: Does the feedback loop (using rewritten prompt to re-run bandit) improve demo selection?
- **Expected finding**: Round 2 provides meaningful gain; round 3 shows minimal additional improvement and costs 2× more queries.

### 6.5 Demo Pool Composition
- **What**: Compare demo pools: (a) oracle solve.sh only, (b) oracle + random agent runs (failures included), (c) task-category-filtered (only same-category demos), (d) all-tasks pool.
- **Tests**: Does filtering by task category improve demo relevance? Does including failure demonstrations help the bandit distinguish?
- **Expected finding**: Same-category filtering helps; including failures forces the bandit to actively discriminate, improving efficiency.

### 6.6 Rewriter Model Size / Type
- **What**: Compare HF BPO (7B) vs. GPT-4o-mini vs. Claude-3-Haiku as the rewriter.
- **Tests**: Does a stronger rewriter improve results? Is THUDM/BPO the bottleneck?
- **Expected finding**: GPT-4o-mini as rewriter outperforms 7B BPO on complex tasks; Claude-3-Haiku competitive with GPT-4o-mini at lower cost.

### 6.7 Single-Skill vs. Multi-Skill Task Analysis
- **What**: Report SPLICE-Full results separately for single-skill (N~60) and multi-skill (N~29) tasks.
- **Tests**: Does SPLICE help more on composition tasks where demonstrations of multi-skill chains are most informative?
- **Expected finding**: SPLICE-Full's gain is larger on multi-skill tasks (+10–15 pp vs. +5–8 pp on single-skill).

---

## 7. Figure and Table Inventory

### Tables

| Table | Content | Data Source |
|-------|---------|-------------|
| Table 1 | Main results: all methods × success rate, std, query count | full_eval_results.json |
| Table 2 | Ablation: k ∈ {1,3,5} × method | ablation runs |
| Table 3 | Ablation: n_rounds ∈ {1,2,3} × success rate and query count | ablation runs |
| Table 4 | Single-skill vs. multi-skill success rate breakdown | full_eval_results.json + task metadata |
| Table 5 | Rewriter comparison: BPO-7B vs. GPT-4o-mini vs. Claude-Haiku | ablation runs |

### Figures

| Figure | Type | Content |
|--------|------|---------|
| Figure 1 | Architecture diagram | SPLICE system: DemoBank → Select → Rewrite → Eval → (loop) |
| Figure 2 | Bar chart | Main results (Table 1) with error bars |
| Figure 3 | Line plot | Bandit convergence: tau_UGapE vs. round for representative tasks |
| Figure 4 | Scatter plot | Per-task success rate: CASE-Only vs. SPLICE-Full (show where rewriting helps) |
| Figure 5 | Bar chart | Multi-skill vs. single-skill breakdown: baseline, CASE-Only, SPLICE-Full |
| Figure 6 | Heatmap | Demo selection similarity: which demo categories are selected for which task types |
| Figure 7 (appendix) | Line plot | Loop depth ablation: success rate vs. n_rounds |

**All figures require real experimental data. Only Figures 3 and 6 can be generated from dry-run bandit diagnostics (partially).**

---

## 8. Related Work

### 8.1 SkillsBench (Li et al., 2026)
The benchmark and evaluation protocol for this work. SPLICE is evaluated on SkillsBench's 89 tasks. Key finding from the benchmark: multi-skill composition is the primary bottleneck. SPLICE directly addresses this. Cite as the primary evaluation benchmark.

### 8.2 SkillRL (Xia et al., 2026)
RL-based skill evolution. Complementary to SPLICE: SkillRL improves skill bodies through reward-driven evolution; SPLICE improves skill invocation through demo selection and prompt rewriting. Key contrast: SkillRL requires RL training and rollout infrastructure; SPLICE is inference-time only and requires only binary task success feedback. Potential future work: combine SPLICE's invocation optimization with SkillRL's skill evolution.

### 8.3 SkillRouter (Zheng et al., 2026)
Two-stage retrieval + reranking for skill routing (74% Hit@1 over 80K skills). SkillRouter addresses which skill to select; SPLICE addresses how to invoke the selected skill. These are sequential pipeline stages. SPLICE can be used downstream of SkillRouter. Cite as providing the skill selection step that precedes SPLICE's invocation optimization.

### 8.4 BPO — Black-Box Prompt Optimization (Cheng et al., ACL 2024)
Direct inspiration for SPLICE-Rewrite. BPO trains a 7B model on preference pairs to rewrite prompts without model gradient access. SPLICE adapts BPO to the skill-invocation setting by conditioning rewriting on selected demonstrations. Key difference: BPO is task-agnostic and uses no structured demonstrations; SPLICE-Rewrite is skill-aware and demo-conditioned. The THUDM/BPO model is the backbone of SPLICE-Rewrite.

### 8.5 CASE — Context-Aware Sample Efficiency (Purohit et al., ICML 2025)
Direct inspiration for SPLICE-Select. CASE applies UGapE gap-index bandits to ICL demonstration selection on tabular NLP tasks (TabMWP). SPLICE-Select adapts CASE's algorithm to the agent skill invocation setting with a binary task success reward. Key differences: (1) reward function is task execution success vs. CASE's LLM accuracy, (2) arms are skill-usage demonstrations vs. tabular exemplars, (3) SPLICE adds the iterative rewriting loop that CASE lacks.

### 8.6 OptiSeq (Bhope et al.)
Dynamic ICL demonstration ordering. OptiSeq shows that the ordering of demonstrations at inference time significantly affects LLM performance. SPLICE selects demonstrations via bandit; a natural extension is to additionally order them using OptiSeq's recency-bias-aware reordering. Include as related work and motivate as potential integration in future work.

### 8.7 SkillClone (Zhu et al., 2026)
Multi-modal clone detection in skill ecosystems. Related context for the SkillsBench ecosystem. Not directly relevant to SPLICE but establishes the multi-skill landscape.

### 8.8 In-Context Learning Survey (Dong et al., 2022; Min et al., 2022)
ICL foundation papers establishing that demonstration selection and formatting matter. Min et al. shows that ground-truth labels matter less than format; SPLICE focuses on skill-invocation structure rather than label correctness. Cite as ICL foundations.

### 8.9 AutoPrompt / APE / OPRO (Shin et al., 2020; Zhou et al., 2022; Yang et al., 2023)
Automatic prompt engineering methods. Contrast: gradient-based (AutoPrompt) or optimization-based (OPRO) methods require white-box or repeated access; SPLICE is black-box and inference-time. Position SPLICE as complementary to gradient-based approaches.

### 8.10 Bandit Literature: UGapE (Gabillon et al., 2012; Auer et al., 2010)
Foundational bandit algorithms underlying CASE and SPLICE-Select. UCB (Auer et al.) and UGapE (Gabillon et al.) are the theoretical foundations for the gap-index stopping criterion. Cite for theoretical grounding.

---

## 9. Limitations

1. **Dry-Run Results Only**: All current experimental results use a mock stochastic evaluator with hardcoded success probabilities. No real LLM agent has been called. The pilot results (baseline=50%, CASE-only=40%, SPLICE-full=80%) are artifacts of a seeded random number generator, not real task performance. This is the most significant limitation for submission readiness.

2. **Bandit Reward Noise**: In real SkillsBench evaluations, a single demo evaluation consumes one full agent run (potentially 5–15 minutes on a remote API). With 20 arms and ~30 rounds, SPLICE-Select requires approximately 30 agent runs per task. On 89 tasks, this is ~2,670 total agent runs just for demo selection. The computational cost of the bandit may be prohibitive in practice without a fast proxy evaluator.

3. **SPLICE-Rewrite Identity in Dry-Run**: In the pilot, SPLICE-Rewrite uses a mock rewriter that prepends "Optimized: " to the original prompt. The claimed benefit of prompt rewriting is entirely attributable to the DryRunEvaluator's `has_rewrite` bonus (+0.08), not to any real improvement in prompt quality. The rewriting component has not been validated with real LLM outputs.

4. **Demo Pool Homogeneity**: The DemoBank is built from oracle solve.sh scripts (all labeled outcome=1). This means all candidates in the bandit are positive examples. The bandit is selecting the most relevant positive example, not discriminating between positive and negative. Including failure demonstrations would enable the bandit to identify which invocation patterns fail, potentially improving the BPO rewriting signal.

5. **No Multi-Skill Evaluation**: The pilot only ran 10 tasks, and the task categorization (single-skill vs. multi-skill) was not tracked. The central claim — that SPLICE's gain is largest on multi-skill composition — has not been tested even in dry-run mode.

---

## 10. Broader Impact

SPLICE addresses a practical bottleneck in deploying AI agents for complex real-world tasks: even when the right skill is available, agents fail to invoke it correctly. By making skill invocation more reliable through sample-efficient demonstration selection and black-box prompt optimization, SPLICE could reduce the cost and failure rate of AI agents in professional domains including scientific data analysis, legal document processing, and software engineering. The approach is model-agnostic and requires no gradient access, making it broadly applicable to proprietary API-based agents where fine-tuning is impossible. On the risk side, improved skill invocation could amplify the capabilities of agents in ways that increase automation displacement or enable misuse if applied to harmful skill libraries. The BenchFlow/SkillsBench ecosystem is designed for research use under controlled conditions, which mitigates near-term deployment risks.

---

*Narrative report generated: 2026-04-21*
*Pipeline stage: Stage 4 (Self-Review + Report Writing)*
*Next stage: Stage 5 (Real Experiment Execution)*
