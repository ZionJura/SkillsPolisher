# Skills Learning Research — Stage 1: Idea Discovery Report

**Date**: 2026-04-21  
**Domain**: Skill Improvement + Context/In-Context Learning + Prompt Learning/Optimization  
**Target Benchmark**: SkillsBench (89 tasks, BenchFlow SDK)

---

## Part 1: Literature Survey Summary

### 1.1 Surveyed Papers and Systems

| Paper | Venue | Core Contribution |
|-------|-------|-------------------|
| SkillsBench (Li et al. 2026) | — | First benchmark for evaluating AI agent skill usage; 89 tasks, gym-style evaluation, BenchFlow SDK |
| SkillRL (Xia et al. 2026) | — | Recursive skill-augmented RL; agents evolve skills hierarchically through RL feedback |
| SkillRouter (Zheng et al. 2026) | — | Two-stage retrieval (embedding + reranker, 0.6B each) over ~80K skills; 74% Hit@1 |
| SkillClone (Zhu et al. 2026) | — | Multi-modal clone detection in the agent skill ecosystem |
| BPO (Cheng et al. 2024) | ACL 2024 | Black-box prompt optimization via a small model trained on pairwise preference feedback |
| CASE (Purohit et al. 2025) | ICML 2025 | Gap-index bandit algorithm for sample-efficient ICL demonstration selection |
| OptiSeq (Bhope et al.) | — | Dynamic ordering of ICL demonstrations on-the-fly; ordering significantly impacts performance |
| BoT (Chen et al. 2025) | ICLR 2024 | Boosting of Thoughts: trial-and-error iterative reasoning with LLMs |
| llmpebase (Chen & Li) | — | Unified platform for LLM reasoning experiments (CoT, BoT, TR, chain/tree/graph) |

### 1.2 Key Findings from Survey

**SkillsBench** establishes that agents fail most on tasks requiring skill composition (≥2 skills). Performance is below 50% for SOTA models on these tasks. Skill bodies (full text including instructions + scripts) matter more than metadata alone.

**SkillRouter** showed that routing to the right skill requires reading the full skill body, not just name/description. A 2-stage retrieve-then-rerank pipeline achieves 74% Hit@1 over 80K skills. However, it does not adapt skills or optimize their prompts — it only selects which skill to use.

**SkillRL** demonstrates that skills can evolve through RL feedback. Tasks are recursively decomposed, and sub-skills are refined over multiple episodes. However, it does not leverage ICL demonstrations or prompt optimization — skill evolution is purely reward-driven.

**BPO** optimizes prompts in a black-box manner without access to model gradients. A small (7B) LLaMA-based rewriter is trained on preference pairs to rewrite user prompts. However, BPO is task-agnostic and not aware of skill structure or ICL demonstrations.

**CASE** selects demonstration examples for ICL using a bandit framework that minimizes LLM queries while maximizing coverage. Evaluated on TabMWP and similar benchmarks — not on skill-based agent tasks.

**OptiSeq** shows that the ordering of ICL examples matters as much as their selection. Dynamic ordering (computed at inference time) outperforms static orderings. Not applied to agent skills.

### 1.3 Identified Research Gaps

**Gap 1**: No existing work combines skill-body-aware prompt optimization with sample-efficient ICL demonstration selection. BPO optimizes prompts but ignores skill structure; CASE selects demonstrations but not for skill-invocation contexts.

**Gap 2**: ICL demonstrations for skill-based agents have not been studied. When an agent must invoke a skill, which examples of skill usage most effectively guide the invocation? Ordering of these skill-usage examples is also unexplored (OptiSeq gap).

**Gap 3**: Skill evolution via RL (SkillRL) is disconnected from prompt-level optimization. After a skill improves, its invocation prompt may no longer be optimal — there is no feedback loop between skill body changes and prompt adaptation.

**Gap 4**: Multi-skill composition fails at <50% for SOTA models. No work specifically addresses prompt or demonstration strategies for skill composition vs single-skill tasks.

**Gap 5**: SkillRouter performs routing but the routing signal itself (skill body text) is static. Routing quality could improve if skill bodies were iteratively rewritten to be more discriminative (routing-aware skill authoring).

---

## Part 2: Generated Research Ideas

---

### Idea 1: SPLICE — Skill-Prompt Learning via In-Context Example Bandit

**Title**: SPLICE: Sample-Efficient Skill Invocation via Bandit-Selected ICL Demonstrations and Adaptive Prompt Rewriting

**Core Hypothesis**: The quality of skill invocation by an LLM agent can be significantly improved by (a) selecting a small but highly informative set of skill-usage demonstration examples via a bandit algorithm (analogous to CASE), and (b) rewriting the skill invocation prompt based on preference feedback from those demonstrations (analogous to BPO). The joint optimization of demonstration selection and prompt rewriting creates a feedback loop that outperforms either method alone.

**Why Novel vs. Existing Work**:
- CASE selects demonstrations but for standard NLP tasks, not skill-invocation contexts, and does not optimize the prompt text.
- BPO optimizes prompts but has no notion of demonstrations or skill structure.
- SkillRL evolves skills through reward signals but does not use ICL or prompt optimization.
- No prior work combines bandit-based demonstration selection with black-box prompt rewriting specifically for agent skill invocation.

**Proposed Method**:
1. **Demonstration Bank Construction**: For each skill, collect a bank of (task, skill-invocation, outcome) triples from historical agent runs on SkillsBench.
2. **Bandit Demonstration Selection (SPLICE-Select)**: Adapt CASE's gap-index bandit to the skill-invocation setting. Each "arm" is a candidate demonstration. The reward signal is binary task success on SkillsBench. Select a top-k subset of demonstrations that maximizes success rate while minimizing LLM calls.
3. **Prompt Rewriting (SPLICE-Rewrite)**: Given selected demonstrations, use a BPO-style small model (7B rewriter) trained on preference pairs (selected demos + original prompt vs. selected demos + rewritten prompt) to rewrite the skill invocation prefix. The rewriter is trained offline on SkillsBench tasks; at inference, it is applied zero-shot to unseen tasks.
4. **Iterative Loop**: The updated prompt affects future skill invocations, generating new (task, invocation, outcome) triples that update the demonstration bank and refine the bandit model.

**Experimental Design**:
- **Datasets/Benchmarks**: SkillsBench (89 tasks), with focus on the multi-skill composition subset
- **Baselines**:
  - Raw skill invocation (no demos, no rewriting) — SkillsBench default
  - BPO alone (prompt rewriting without demonstration selection)
  - CASE alone (demonstration selection without prompt rewriting)
  - Random k-shot demonstrations
  - OptiSeq ordering applied to random demos
- **Metrics**: Task success rate (reward.txt from SkillsBench), number of LLM queries (efficiency), skill invocation accuracy (does the agent invoke the right skill?)
- **Ablation**: SPLICE-Select only vs. SPLICE-Rewrite only vs. SPLICE-Full

**Pilot Experiment Design** (<1 hour on single GPU):
- Pick 10 SkillsBench tasks, all single-skill
- Build a 20-example demonstration bank from oracle solutions (solve.sh outputs)
- Run CASE bandit for 5 rounds to select top-3 demonstrations per task
- Rewrite the skill SKILL.md prefix using a pre-trained BPO model (THUDM/BPO on HuggingFace)
- Compare success rate: baseline (no demos) vs. CASE demos vs. CASE demos + BPO rewrite
- Expected runtime: ~30 minutes on a single A100 with 4-bit quantized BPO model

**Estimated Novelty**: HIGH  
**Estimated Feasibility**: HIGH (all components — CASE code, BPO model, SkillsBench SDK — are available)

---

### Idea 2: SkillCompose-ICL — Compositional Skill Invocation via Ordered Demonstration Chains

**Title**: SkillCompose-ICL: Dynamic Demonstration Ordering for Multi-Skill Composition Tasks

**Core Hypothesis**: For tasks requiring 2+ skills (the hardest SkillsBench setting), the ordering and selection of ICL demonstrations depicting successful skill-composition chains is the primary bottleneck to agent success. An OptiSeq-style dynamic ordering algorithm adapted to skill-composition graphs will significantly improve performance on multi-skill tasks.

**Why Novel vs. Existing Work**:
- OptiSeq orders ICL examples for standard tasks (classification, QA); it has no concept of skill composition order or skill dependency graphs.
- SkillsBench evaluates multi-skill tasks but no prior work specifically designs ICL strategies for them.
- Skill composition requires correct ordering of skill invocations (skill A must precede skill B); demonstration ordering must reflect this dependency.

**Proposed Method**:
1. **Skill Dependency Graph Extraction**: For each multi-skill SkillsBench task, extract a skill dependency graph from oracle solve.sh: which skills are invoked in which order and with which intermediate outputs.
2. **Demonstration Ordering via Graph Similarity**: Given a new test task, compute similarity between its dependency graph topology and candidate demonstration graphs. Order demonstrations so that structurally similar examples (most similar graph structure) appear closest to the query (consistent with OptiSeq's recency bias finding).
3. **Chain Demonstration Construction**: Each demonstration is a chain of (skill-A invocation, intermediate result, skill-B invocation, final result). The agent sees k such chains ordered by graph similarity before attempting the test task.
4. **Adaptive Reordering**: After each failed attempt, use the failure mode (which skill failed?) to reorder demonstrations emphasizing the failing skill's usage.

**Experimental Design**:
- **Datasets**: SkillsBench multi-skill composition subset
- **Baselines**: No demos; random k-shot; OptiSeq (flat ordering without graph structure); BM25 similarity ordering
- **Metrics**: Task success rate, correct skill invocation rate (each skill separately), partial credit (fraction of required skills correctly invoked)
- **Ablation**: Graph-similarity ordering vs. flat ordering; with vs. without adaptive reordering

**Pilot Experiment Design**:
- Select 5 two-skill SkillsBench tasks
- Extract dependency chains from solve.sh for 15 examples each (using oracle)
- Implement graph similarity as graph edit distance over skill-invocation sequences
- Compare flat random ordering vs. graph-similarity ordering on GPT-4o-mini via BenchFlow SDK
- Expected time: ~45 minutes

**Estimated Novelty**: MEDIUM-HIGH  
**Estimated Feasibility**: HIGH

---

### Idea 3: EvolvSkill-BPO — Iterative Skill Body Evolution via Black-Box Prompt Optimization

**Title**: EvolvSkill-BPO: Closing the Loop Between Skill Execution Feedback and Skill Body Rewriting

**Core Hypothesis**: Current skill evolution approaches (SkillRL) update skills only through RL reward signals. A more sample-efficient approach is to treat the skill body (the SKILL.md text + associated scripts) as a "prompt" and apply BPO-style black-box preference optimization to iteratively rewrite skill bodies based on task success/failure feedback. This closes the loop between skill execution outcomes and skill content.

**Why Novel vs. Existing Work**:
- SkillRL evolves skills through RL reward but does not operate at the prompt/text level — it generates new sub-skills but does not refine the wording of existing ones.
- BPO rewrites user prompts but has no notion of skill bodies or iterative execution feedback.
- SkillRouter uses static skill bodies for routing; if skill bodies improve, routing quality should also improve.

**Proposed Method**:
1. **Skill Execution Log Collection**: Run a base agent on SkillsBench; log (task, skill-body-used, outcome) triples. Partition into successes and failures.
2. **Preference Pair Construction**: For each failed task, the original skill body is the "rejected" prompt; the oracle-derived skill (extracted from solve.sh) is the "chosen" prompt. Construct (chosen, rejected) pairs as BPO training data.
3. **Skill Body Rewriter Training**: Fine-tune a small model (7B LLaMA) on these preference pairs using DPO or RLHF to produce a skill-body rewriter that improves skill bodies based on failure modes.
4. **Iterative Evolution Loop**: Apply rewriter to skill bodies; re-run agent; collect new logs; repeat for N rounds. Evaluate improvement trajectory.
5. **Routing Validation**: After evolution, measure SkillRouter Hit@1 on evolved vs. original skill bodies to show joint improvement.

**Experimental Design**:
- **Datasets**: SkillsBench (all 89 tasks); SkillRouter evaluation set (75 queries)
- **Baselines**: Original skill bodies; SkillRL (if replicable); random skill body perturbation; human-edited skills
- **Metrics**: SkillsBench task success rate over evolution rounds; SkillRouter Hit@1 before/after evolution
- **Ablation**: Number of evolution rounds; preference pair quality (oracle vs. GPT-4 generated)

**Pilot Experiment Design**:
- Select 10 SkillsBench tasks where the base agent fails
- Manually construct 20 (chosen, rejected) skill body pairs using oracle solve.sh as chosen
- Fine-tune THUDM/BPO (already trained for prompt rewriting) with 5 LoRA steps on these pairs
- Apply to 3 skill bodies; re-run agent; check if success rate improves
- Expected time: ~50 minutes (LoRA fine-tuning on single GPU)

**Estimated Novelty**: HIGH  
**Estimated Feasibility**: MEDIUM (requires LoRA fine-tuning; preference pair construction needs careful design)

---

### Idea 4: SkillICL-Router — Demonstration-Conditioned Skill Routing

**Title**: SkillICL-Router: Improving Skill Routing via In-Context Demonstration Conditioning

**Core Hypothesis**: SkillRouter routes queries to skills based on query-skill similarity alone. Conditioning the routing query on a small set of relevant ICL demonstrations (analogous to CASE-selected examples) will improve routing precision, especially for ambiguous queries where multiple skills overlap.

**Why Novel vs. Existing Work**:
- SkillRouter performs retrieval + reranking but treats each query independently.
- No prior work uses ICL demonstrations to augment the routing query itself.
- CASE selects demonstrations for task-solving; applying it to query augmentation for routing is novel.

**Proposed Method**:
1. **Demonstration-Augmented Query Construction**: Given a routing query q, use CASE bandit to select k demonstration (query, correct-skill) pairs from a curated pool.
2. **Augmented Query**: Construct q' = [demo_1 query → skill_name; ...; demo_k query → skill_name; q → ?] and pass q' through SkillRouter-Embedding for retrieval.
3. **Bandit Query Budget**: CASE's gap-index criterion determines how many demonstrations are needed per query; for easy queries (large margin between top-1 and top-2 skills), fewer demos are used.
4. **Reranker Conditioning**: Pass the augmented representation to SkillRouter-Reranker with the demonstrations providing context for the final ranking.

**Experimental Design**:
- **Datasets**: SkillRouter evaluation set (75 queries over 80K skills); SkillsBench (routing as a sub-task)
- **Baselines**: SkillRouter (original, no demos); BM25 routing; SkillRouter with random demos; SkillRouter with top-1 similar demo
- **Metrics**: Hit@1, Hit@5, MRR; number of demonstrations needed per query (efficiency)
- **Ablation**: k=1,2,3,5 demonstrations; CASE selection vs. random selection

**Pilot Experiment Design**:
- Use SkillRouter's 75-query evaluation set
- Construct a 100-example (query, skill) demonstration pool from training split
- Compare: SkillRouter (no demos) vs. SkillRouter + 3 CASE-selected demos (using accuracy reward)
- Expected time: ~20 minutes (inference only, no training)

**Estimated Novelty**: MEDIUM  
**Estimated Feasibility**: HIGH (pure inference-time change, no training required)

---

### Idea 5: AdaptiSkill — Context-Adaptive Skill Prompt Compression for Efficient Composition

**Title**: AdaptiSkill: Compressing Skill Prompts Adaptively for Multi-Skill Composition via Learned ICL Selectors

**Core Hypothesis**: In multi-skill composition tasks, the agent's context window is consumed by multiple full skill bodies. Compressing each skill body to a task-relevant summary—selected adaptively via ICL example similarity—preserves performance while enabling more skills to fit in context. A CASE-style bandit selects which sections of each skill body to include based on the current task.

**Why Novel vs. Existing Work**:
- SkillsBench uses full skill bodies; no work compresses skills for context efficiency.
- CASE selects examples but not sub-sections of structured documents.
- BPO rewrites prompts but not by selective compression.
- This directly addresses the multi-skill composition failure mode in SkillsBench.

**Proposed Method**:
1. **Skill Section Decomposition**: Parse each skill body into sections (description, usage, examples, scripts). Each section is an "arm" in a bandit.
2. **Task-Adaptive Section Selection**: Given a test task, use a CASE-style bandit to select which sections of each required skill are most relevant. Reward = task success.
3. **Context Budget Management**: Given a fixed context window, allocate token budget across skills proportional to their estimated importance (estimated via embedding similarity to task).
4. **Compressed Skill Prompt**: Concatenate selected sections across skills to form a compressed multi-skill context.
5. **ICL Calibration**: Use k successful task demonstrations to calibrate section importance weights per skill.

**Experimental Design**:
- **Datasets**: SkillsBench multi-skill composition tasks
- **Baselines**: Full skill bodies (no compression); random section selection; BM25 section selection; GPT-4 summarization of skill bodies
- **Metrics**: Task success rate; context tokens used (efficiency); token reduction ratio
- **Ablation**: Budget constraints (25%, 50%, 75% of full body); number of calibration demonstrations

**Pilot Experiment Design**:
- Select 5 two-skill SkillsBench tasks
- Manually segment 2 skill bodies into 4-6 sections each
- Implement greedy section selection by embedding similarity (cosine) to task instruction
- Compare full body vs. top-50% sections by similarity on GPT-4o-mini
- Expected time: ~25 minutes

**Estimated Novelty**: MEDIUM-HIGH  
**Estimated Feasibility**: HIGH

---

## Part 3: Idea Ranking

| Rank | Idea | Novelty | Feasibility | Score (N×F) | Justification |
|------|------|---------|-------------|-------------|---------------|
| 1 | SPLICE (Idea 1) | HIGH (3) | HIGH (3) | 9 | Directly combines CASE + BPO in a novel skill-invocation loop; all components available; evaluable on SkillsBench |
| 2 | EvolvSkill-BPO (Idea 3) | HIGH (3) | MEDIUM (2) | 6 | Novel skill-body evolution via preference feedback; slightly harder to implement (LoRA fine-tuning) |
| 3 | SkillCompose-ICL (Idea 2) | MEDIUM-HIGH (2.5) | HIGH (3) | 7.5 | Strong practical impact on multi-skill composition; graph similarity is novel for ICL ordering |
| 4 | AdaptiSkill (Idea 5) | MEDIUM-HIGH (2.5) | HIGH (3) | 7.5 | Practical compression idea; addresses real SkillsBench bottleneck; lower novelty vs. SPLICE |
| 5 | SkillICL-Router (Idea 4) | MEDIUM (2) | HIGH (3) | 6 | Useful but closer to an incremental improvement to SkillRouter |

**Final ranking**: SPLICE > SkillCompose-ICL ≈ AdaptiSkill > EvolvSkill-BPO > SkillICL-Router

**Selected Top Idea**: **SPLICE — Skill-Prompt Learning via In-Context Example Bandit**

---

## Part 4: Top Idea — Detailed Experimental Plan

### 4.1 Full Research Plan for SPLICE

**Full Title**: SPLICE: Sample-Efficient Skill Invocation Improvement via Bandit-Selected ICL Demonstrations and Black-Box Prompt Rewriting

**Research Question**: Can we jointly optimize (1) which skill-usage demonstrations to show an LLM agent and (2) how to phrase the skill invocation prompt, using only black-box feedback from task execution, to improve agent performance on SkillsBench?

---

### 4.2 System Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │               SPLICE System                  │
                        │                                              │
  Task (SkillsBench) ──►│  1. SPLICE-Select (Bandit Demo Selection)   │
                        │     - Arms: candidate skill-usage demos      │
                        │     - Reward: task success (binary)          │
                        │     - Algorithm: CASE gap-index bandit        │
                        │     - Output: top-k demos per task           │
                        │                                              │
                        │  2. SPLICE-Rewrite (BPO Prompt Rewriting)   │
                        │     - Input: selected demos + current skill  │
                        │       invocation prompt (SKILL.md prefix)    │
                        │     - Model: 7B LLaMA fine-tuned on          │
                        │       skill-usage preference pairs           │
                        │     - Output: rewritten skill invocation     │
                        │       prompt                                 │
                        │                                              │
                        │  3. Execution & Feedback Loop               │
                        │     - Run agent with new demos + new prompt  │
                        │     - Collect (task, invocation, outcome)    │
                        │     - Update demo bank + preference pairs    │
                        └─────────────────────────────────────────────┘
```

---

### 4.3 Detailed Experimental Plan

#### Phase 1: Data Collection (Week 1-2)

**Goal**: Build demonstration bank and preference pairs for SPLICE training.

**Steps**:
1. Run oracle agent (using `solve.sh`) on all 89 SkillsBench tasks.
2. For each task, extract skill invocation sequences: (task instruction, selected skill, invocation prompt used, outcome).
3. Partition by outcome: successes → "chosen" demonstrations; failures → "rejected" demonstrations.
4. Construct demonstration bank: 20 examples per skill (where available), balanced between task types.
5. Construct BPO preference pairs: for each failed task where oracle succeeds, pair (oracle invocation prompt = chosen, original agent invocation prompt = rejected).

**Expected output**: ~89 (task, demo-bank, preference-pair) records; ~300-500 preference pairs total.

#### Phase 2: SPLICE-Select Implementation (Week 2-3)

**Goal**: Adapt CASE gap-index bandit for skill-invocation demo selection.

**Key changes from CASE**:
- Arms are skill-usage demonstrations (not tabular exemplars)
- Reward function: `reward = bench_eval_task(task, selected_demos + skill_prompt)` — binary 0/1
- Feature representation: embedding of (task instruction, demo similarity) via sentence-transformers
- Gap-index criterion: `gap_i = mu_hat[i] - mu_hat[i+1]` where mu_hat are estimated demo utilities
- Stop criterion: delta = 0.05 (same as CASE default)

**Implementation**: Extend `CASE/Code/LLM_experiments/` with `splice_select.py`.

#### Phase 3: SPLICE-Rewrite Implementation (Week 3-4)

**Goal**: Train BPO-style skill-body rewriter.

**Training**:
- Base model: `THUDM/BPO` (already fine-tuned LLaMA-2 7B for prompt rewriting)
- Fine-tune with LoRA (rank=16, alpha=32) on skill-invocation preference pairs from Phase 1
- Training data format: `[INST] Improve this skill invocation prompt to maximize task success: {original_prompt} [/INST] {improved_prompt}`
- Training: 3 epochs, batch size 4, lr 1e-4, A100 40GB
- Expected training time: ~2 hours

**Inference**: For each test task, rewrite the skill's invocation prefix (the first 200 tokens of SKILL.md) conditioned on the SPLICE-Select top-k demonstrations.

#### Phase 4: Evaluation on SkillsBench (Week 4-5)

**Baselines**:

| Method | Description |
|--------|-------------|
| Raw (No-ICL) | Default agent, no demonstrations, original SKILL.md |
| BPO-Only | BPO-rewritten skill prompt, no demonstrations |
| CASE-Only | Top-3 demonstrations selected by CASE bandit, original prompt |
| OptiSeq-Random | 3 random demonstrations, OptiSeq-ordered |
| Random-3shot | 3 random demonstrations, original prompt |
| Oracle | Human-written solve.sh solution |
| SPLICE-Select | Bandit demo selection, no prompt rewriting |
| SPLICE-Rewrite | Prompt rewriting, random demonstrations |
| SPLICE-Full | Full SPLICE system |

**Metrics**:
1. **Task Success Rate**: Primary metric — fraction of 89 tasks with `reward.txt` = 1
2. **LLM Query Efficiency**: Number of LLM calls to reach final answer (lower is better)
3. **Skill Invocation Accuracy**: Fraction of tasks where the correct skill is invoked
4. **Multi-Skill Composition Rate**: Success rate specifically on tasks requiring ≥2 skills
5. **Demo Selection Efficiency**: Number of bandit rounds needed to converge on top-k demos

**Evaluation Protocol**:
```bash
# For each method:
bench eval create -t tasks/<task-id> -a claude-agent-acp -s tasks/<task-id>/environment/skills/
bench metrics jobs/<run-dir>
# Aggregate across all 89 tasks
```

#### Phase 5: Analysis (Week 5-6)

**Ablation studies**:
1. k ∈ {1, 3, 5} demonstrations — how many are needed?
2. Rewriting only the first 100 / 200 / full SKILL.md
3. SPLICE without the iterative feedback loop (single round vs. 3 rounds)
4. Effect of preference pair quality: oracle pairs vs. GPT-4-judged pairs

**Error analysis**:
- Which task categories most benefit from SPLICE?
- Does SPLICE help more on single-skill or multi-skill tasks?
- Where does SPLICE fail — demo selection failure vs. prompt rewriting failure?

---

### 4.4 Baseline Comparison — Expected Results

| Method | Expected Success Rate | Reasoning |
|--------|----------------------|-----------|
| Raw (No-ICL) | ~35-40% | SkillsBench SOTA baseline without demos |
| Random-3shot | ~38-42% | Marginal improvement from random demos |
| OptiSeq-Random | ~40-44% | Ordering helps slightly |
| BPO-Only | ~42-46% | Prompt rewriting helps, but no task-specific demos |
| CASE-Only | ~45-50% | Task-relevant demos improve invocation significantly |
| SPLICE-Select | ~47-52% | Bandit selection more efficient than random CASE |
| SPLICE-Rewrite | ~44-48% | Rewriting with CASE demos as context |
| SPLICE-Full | **~52-58%** | Joint optimization beats individual components |
| Oracle | ~85-90% | Human-written solutions |

**Key expected finding**: SPLICE-Full should achieve the largest gain on multi-skill composition tasks (currently <50% for SOTA), where both the demonstration selection AND prompt rewriting components contribute complementary improvements.

---

## Part 5: Pilot Code Sketch

```python
"""
SPLICE Pilot Experiment
Runs in <1 hour on single GPU (A100 40GB)
Evaluates: baseline vs CASE-Select vs CASE-Select+BPO-Rewrite
on 10 SkillsBench single-skill tasks
"""

import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ── CONFIG ──────────────────────────────────────────────────────────────────
SKILLSBENCH_ROOT = Path("/mnt/d/Code/AI4R/Skills-Learning2/related-works/skillsbench")
PILOT_TASKS = [
    # Select 10 single-skill tasks from SkillsBench
    "tasks/task-001", "tasks/task-002", "tasks/task-003", "tasks/task-004",
    "tasks/task-005", "tasks/task-006", "tasks/task-007", "tasks/task-008",
    "tasks/task-009", "tasks/task-010",
]
N_DEMO_CANDIDATES = 20   # Arms in the bandit
K_SELECTED = 3           # Top-k demos to select
DELTA = 0.05             # Confidence parameter (CASE default)
BPO_MODEL_ID = "THUDM/BPO"

# ── STEP 1: BUILD DEMONSTRATION BANK ────────────────────────────────────────

def build_demo_bank(task_ids: list[str]) -> dict[str, list[dict]]:
    """
    For each task, read the oracle solution (solve.sh output) and
    construct candidate demonstrations.
    Returns: {task_id: [{"instruction": str, "skill_invocation": str, "outcome": int}]}
    """
    bank = {}
    for task_id in task_ids:
        task_path = SKILLSBENCH_ROOT / task_id
        instruction = (task_path / "instruction.md").read_text()
        # Read skill body from environment/skills/
        skills_dir = task_path / "environment" / "skills"
        skill_bodies = []
        for skill_file in skills_dir.glob("*/SKILL.md"):
            skill_bodies.append(skill_file.read_text())
        # Oracle invocation: treat solve.sh as the gold invocation
        oracle_invocation = (task_path / "solution" / "solve.sh").read_text()
        bank[task_id] = [
            {"instruction": instruction,
             "skill_body": skill_bodies[0] if skill_bodies else "",
             "invocation": oracle_invocation,
             "outcome": 1}  # oracle always succeeds
        ]
    return bank


# ── STEP 2: CASE BANDIT DEMO SELECTION ──────────────────────────────────────

class CASEBandit:
    """
    Simplified gap-index bandit for demonstration selection.
    Adapts CASE (Purohit et al. 2025) to skill-invocation setting.
    Each arm is a candidate demonstration.
    """
    def __init__(self, n_arms: int, k: int, delta: float = 0.05):
        self.n_arms = n_arms
        self.k = k
        self.delta = delta
        self.counts = np.zeros(n_arms)       # pulls per arm
        self.rewards = np.zeros(n_arms)      # cumulative reward per arm
        self.t = 0                            # total rounds

    def ucb_score(self) -> np.ndarray:
        """Upper confidence bound for each arm."""
        mu_hat = np.where(self.counts > 0, self.rewards / self.counts, 1.0)
        beta = np.sqrt(np.log(self.n_arms * self.t**2 / self.delta + 1) /
                       (2 * np.maximum(self.counts, 1)))
        return mu_hat + beta

    def gap_index(self) -> float:
        """
        Gap index: difference between k-th and (k+1)-th best UCB scores.
        Stop when gap is large (high confidence in top-k set).
        """
        scores = self.ucb_score()
        sorted_scores = np.sort(scores)[::-1]
        if len(sorted_scores) <= self.k:
            return float('inf')
        return sorted_scores[self.k - 1] - sorted_scores[self.k]

    def select_arm(self) -> int:
        """Select arm with highest UCB score that hasn't converged."""
        return int(np.argmax(self.ucb_score()))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.t += 1

    def top_k_arms(self) -> list[int]:
        """Return indices of top-k arms by estimated mean reward."""
        mu_hat = np.where(self.counts > 0, self.rewards / self.counts, 0.0)
        return list(np.argsort(mu_hat)[::-1][:self.k])

    def converged(self, threshold: float = 0.5) -> bool:
        """Stop if gap index exceeds threshold or min pulls reached."""
        return self.gap_index() > threshold and self.t >= 2 * self.n_arms


def evaluate_demos_on_task(task_id: str, selected_demo_indices: list[int],
                            demo_candidates: list[dict],
                            skill_prompt: str,
                            llm_fn) -> float:
    """
    Run agent with selected demonstrations and skill prompt on task.
    Returns 1.0 on success, 0.0 on failure.
    In pilot: call BenchFlow SDK or a local mock.
    """
    demos = [demo_candidates[i] for i in selected_demo_indices]
    demo_text = "\n\n".join([
        f"Example {j+1}:\nTask: {d['instruction']}\nInvocation: {d['invocation']}"
        for j, d in enumerate(demos)
    ])
    prompt = f"{demo_text}\n\n---\nCurrent Task:\n{skill_prompt}"
    response = llm_fn(prompt)
    # In real eval: run bench eval create and parse reward.txt
    # In pilot: use simple heuristic
    return 1.0 if "success" in response.lower() else 0.0


def splice_select(task_id: str, demo_candidates: list[dict],
                  skill_prompt: str, llm_fn,
                  k: int = K_SELECTED, delta: float = DELTA) -> list[int]:
    """
    Run CASE bandit to select top-k demonstrations for a task.
    Returns indices into demo_candidates.
    """
    bandit = CASEBandit(n_arms=len(demo_candidates), k=k, delta=delta)
    max_rounds = 50  # safety limit for pilot

    for _ in range(max_rounds):
        if bandit.converged():
            break
        arm = bandit.select_arm()
        reward = evaluate_demos_on_task(task_id, [arm], demo_candidates,
                                        skill_prompt, llm_fn)
        bandit.update(arm, reward)

    return bandit.top_k_arms()


# ── STEP 3: BPO SKILL PROMPT REWRITING ──────────────────────────────────────

class BPOSkillRewriter:
    """
    Wraps THUDM/BPO model for skill invocation prompt rewriting.
    """
    def __init__(self, model_id: str = BPO_MODEL_ID, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            load_in_4bit=True,  # 4-bit quantization for pilot efficiency
            device_map=device
        )
        self.template = (
            "[INST] You are an expert at writing skill invocation prompts for AI agents. "
            "Given the following examples of successful skill invocations:\n{demos}\n\n"
            "Please rewrite the following skill instruction to be more effective:\n{skill_prompt} [/INST]"
        )

    def rewrite(self, skill_prompt: str, selected_demos: list[dict]) -> str:
        demo_text = "\n".join([
            f"- Task: {d['instruction'][:100]}... → Invocation: {d['invocation'][:100]}..."
            for d in selected_demos[:3]
        ])
        input_text = self.template.format(demos=demo_text, skill_prompt=skill_prompt[:500])
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        return response.strip()


# ── STEP 4: PILOT EVALUATION ─────────────────────────────────────────────────

def run_pilot():
    """
    Pilot experiment comparing:
    1. Baseline: no demos, original skill prompt
    2. CASE-only: CASE-selected demos, original skill prompt
    3. SPLICE-full: CASE-selected demos + BPO-rewritten skill prompt
    """
    results = {"baseline": [], "case_only": [], "splice_full": []}

    # Load BPO model
    print("Loading BPO model (4-bit quantized)...")
    rewriter = BPOSkillRewriter()

    # Mock LLM function for pilot (replace with real BenchFlow eval)
    def mock_llm(prompt: str) -> str:
        # In real experiment: call claude-agent-acp via BenchFlow SDK
        # Here: simulate with GPT-4o-mini API call
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content

    # Build demo bank from available tasks
    demo_bank = build_demo_bank(PILOT_TASKS)

    for task_id in PILOT_TASKS:
        print(f"\nEvaluating task: {task_id}")
        task_path = SKILLSBENCH_ROOT / task_id

        # Load skill prompt
        skills_dir = task_path / "environment" / "skills"
        skill_files = list(skills_dir.glob("*/SKILL.md"))
        if not skill_files:
            print(f"  No SKILL.md found, skipping.")
            continue
        skill_prompt = skill_files[0].read_text()[:500]  # First 500 tokens

        # Build demo candidates (use other tasks' oracle solutions as demos)
        demo_candidates = []
        for other_task_id, demos in demo_bank.items():
            if other_task_id != task_id:
                demo_candidates.extend(demos)
        if not demo_candidates:
            print(f"  No demo candidates available, skipping.")
            continue

        # 1. Baseline: no demos
        baseline_score = evaluate_demos_on_task(
            task_id, [], demo_candidates, skill_prompt, mock_llm)
        results["baseline"].append(baseline_score)
        print(f"  Baseline: {baseline_score:.2f}")

        # 2. CASE-only: bandit-selected demos, original prompt
        top_k_indices = splice_select(
            task_id, demo_candidates[:N_DEMO_CANDIDATES], skill_prompt, mock_llm)
        case_score = evaluate_demos_on_task(
            task_id, top_k_indices, demo_candidates, skill_prompt, mock_llm)
        results["case_only"].append(case_score)
        print(f"  CASE-Only: {case_score:.2f} (selected demos: {top_k_indices})")

        # 3. SPLICE-full: CASE demos + BPO-rewritten prompt
        selected_demos = [demo_candidates[i] for i in top_k_indices]
        rewritten_prompt = rewriter.rewrite(skill_prompt, selected_demos)
        splice_score = evaluate_demos_on_task(
            task_id, top_k_indices, demo_candidates, rewritten_prompt, mock_llm)
        results["splice_full"].append(splice_score)
        print(f"  SPLICE-Full: {splice_score:.2f}")

    # Summary
    print("\n" + "="*50)
    print("PILOT RESULTS SUMMARY")
    print("="*50)
    for method, scores in results.items():
        if scores:
            print(f"{method:20s}: {np.mean(scores):.3f} ± {np.std(scores):.3f} "
                  f"(n={len(scores)})")

    # Save results
    with open("/mnt/d/Code/AI4R/Skills-Learning2/idea-stage/pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to idea-stage/pilot_results.json")


if __name__ == "__main__":
    run_pilot()
```

---

### 5.1 Full Implementation Plan for SPLICE (Post-Pilot)

**Repository structure**:
```
splice/
├── splice_select.py          # CASE bandit demo selection (extends CASE codebase)
├── splice_rewrite.py         # BPO-style skill prompt rewriter
├── splice_loop.py            # Main iterative improvement loop
├── demo_bank.py              # Demonstration bank construction and management
├── eval_runner.py            # BenchFlow SDK integration for SkillsBench eval
├── configs/
│   └── splice_default.json   # Default hyperparameters
├── data/
│   └── preference_pairs/     # Skill-invocation preference pairs (generated)
└── results/
    └── pilot_results.json    # Pilot experiment outputs
```

**Key hyperparameters**:
```json
{
  "bandit": {
    "delta": 0.05,
    "k_selected": 3,
    "max_rounds": 100,
    "n_demo_candidates": 20
  },
  "rewriter": {
    "model_id": "THUDM/BPO",
    "lora_rank": 16,
    "lora_alpha": 32,
    "max_new_tokens": 300,
    "temperature": 0.7,
    "load_in_4bit": true
  },
  "loop": {
    "n_rounds": 3,
    "tasks": "all_89",
    "eval_model": "claude-agent-acp"
  }
}
```

---

## Part 6: Connections to Existing Codebases

| SPLICE Component | Built On | File to Extend |
|-----------------|----------|----------------|
| Bandit demo selection | CASE | `CASE/Code/LLM_experiments/CASE_tabmwp_selection.py` |
| Gap-index criterion | CASE | `CASE/bandits.py` |
| Preference pairs | BPO | `BPO/src/data_construction/` |
| Skill prompt rewriting | BPO | `BPO/src/inference/` |
| Evaluation | SkillsBench | `bench eval create -t ... -a claude-agent-acp -s ...` |
| Reasoning backbone | llmpebase | `llmpebase/models/` (for CoT + skill invocation) |
| Skill routing (validation) | SkillRouter | `SkillRouter/scripts/evaluate_open_models.sh` |

---

*Report generated: 2026-04-21*  
*Pipeline: Stage 1 — Idea Discovery*  
*Next stage: Stage 2 — Pilot Experiment Execution*
