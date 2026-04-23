"""
splice_select.py — CASE-style Gap-Index Bandit for Demo Selection

Adapts the CASE (Purohit et al. 2025) gap-index bandit algorithm to the
skill-invocation demonstration selection setting for SPLICE.

Each "arm" is a candidate demonstration from the DemoBank.
The reward signal is binary task success (0/1).
The algorithm uses UCB + gap-index stopping criterion.

Usage:
    from splice_select import CASEBandit
    bandit = CASEBandit(n_arms=20, k=3, delta=0.05)
    arm = bandit.select_arm()
    bandit.update(arm, reward=1.0)
    if bandit.converged():
        top_k = bandit.top_k_arms()
"""

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── CASE Gap-Index Bandit ────────────────────────────────────────────────────

class CASEBandit:
    """
    Gap-index Top-m bandit for demonstration selection.

    Directly adapts the CASE (Purohit et al., ICML 2025) UGapE-style
    gap-index criterion to the skill-invocation demo selection task.

    Arms correspond to candidate demonstrations from the DemoBank.
    Reward is binary: 1 if the task succeeds with that demo, 0 otherwise.

    Key CASE concepts adapted here:
    - UCB confidence intervals: mu_hat ± beta(t, delta)
    - Gap index B_ij: upper bound on the gap between arm i and arm j
    - UGapE stopping: stop when max gap index among top-k arms is <= epsilon
    - Recommendation: return top-k arms by empirical mean

    Args:
        n_arms: Number of candidate demonstrations (arms).
        k: Number of demonstrations to select (top-k).
        delta: Confidence parameter (CASE default: 0.05).
        epsilon: Stopping tolerance (CASE default: 0.0).
        sigma: Assumed noise std for Gaussian UCB (default: 0.5).
        max_rounds: Hard maximum number of bandit rounds.
    """

    def __init__(
        self,
        n_arms: int,
        k: int,
        delta: float = 0.05,
        epsilon: float = 0.0,
        sigma: float = 0.5,
        max_rounds: int = 200,
    ):
        assert n_arms > 0, "n_arms must be > 0"
        assert 0 < k <= n_arms, f"k must be in (0, {n_arms}]"
        assert 0 < delta < 1, "delta must be in (0, 1)"
        assert sigma > 0, "sigma must be > 0"

        self.n_arms = n_arms
        self.k = k
        self.delta = delta
        self.epsilon = epsilon
        self.sigma = sigma
        self.max_rounds = max_rounds

        # Per-arm statistics
        self.counts = np.zeros(n_arms, dtype=float)    # number of pulls
        self.cum_rewards = np.zeros(n_arms, dtype=float)  # cumulative reward
        self.t = 0  # total rounds elapsed

        # Track history for diagnostics
        self.history: List[Dict[str, Any]] = []

    # ── UCB utilities (CASE HeuristicBeta) ───────────────────────────────────

    def _beta(self, t: int) -> float:
        """
        CASE HeuristicBeta: beta(t) = log((log(t)+1) / delta).
        Matches the CASE paper's choice of exploration bonus function.
        t is 1-indexed; we use max(t, 1) to avoid log(0).
        """
        t_safe = max(t, 1)
        return math.log((math.log(t_safe) + 1.0) / self.delta + 1.0)

    def _ucb(self, arm: int) -> float:
        """
        UCB index for arm i: mu_hat[i] + CI_radius[i].
        CI_radius = sigma * sqrt(2 * beta(t) / na[i]).
        """
        t = max(self.t, 1)
        na = self.counts[arm]
        mu_hat = self.cum_rewards[arm] / na if na > 0 else 1.0
        if na == 0:
            return float("inf")
        radius = self.sigma * math.sqrt(2.0 * self._beta(t) / na)
        return mu_hat + radius

    def _lcb(self, arm: int) -> float:
        """Lower confidence bound for arm i."""
        t = max(self.t, 1)
        na = self.counts[arm]
        mu_hat = self.cum_rewards[arm] / na if na > 0 else 0.0
        if na == 0:
            return -float("inf")
        radius = self.sigma * math.sqrt(2.0 * self._beta(t) / na)
        return mu_hat - radius

    def _empirical_mean(self, arm: int) -> float:
        """Empirical mean reward for arm i."""
        if self.counts[arm] == 0:
            return 0.0
        return float(self.cum_rewards[arm] / self.counts[arm])

    def empirical_means(self) -> np.ndarray:
        """Return array of all empirical mean rewards."""
        return np.array([self._empirical_mean(i) for i in range(self.n_arms)])

    # ── Gap-index (CASE B_ij) ─────────────────────────────────────────────────

    def _B_ij(self, i: int, j: int) -> float:
        """
        Gap index B(i, j) from CASE / UGapE.
        B(i,j) = UCB(i) - LCB(j) if i != j, else UCB(i).
        Represents the uncertainty in whether arm i is better than arm j.
        The stopping criterion fires when this becomes small.
        """
        if i == j:
            return self._ucb(i)
        return self._ucb(i) - self._lcb(j)

    def _compute_Jt(self) -> List[int]:
        """
        Compute current top-k set J_t (arms with highest empirical means).
        Ties broken by UCB value. Returns plain Python ints.
        """
        means = self.empirical_means()
        return [int(i) for i in np.argsort(means)[::-1][:self.k]]

    def _compute_index_for_arm(self, j: int, J: List[int]) -> float:
        """
        UGapE arm index for arm j in J:
        I(j) = max_{i not in J} B(j, i) where B(j,i) = UCB(i) - LCB(j).
        Represents: how confident are we that arm j belongs in the top-k?
        """
        not_J = [a for a in range(self.n_arms) if a not in J]
        if not not_J:
            return 0.0
        return float(max(self._B_ij(i, j) for i in not_J))

    def _tau_ugape(self) -> float:
        """
        UGapE stopping quantity: max_{j in J_t} I(j).
        Stop when tau <= epsilon.
        Directly mirrors CASE's tauUGapE().
        """
        J = self._compute_Jt()
        indices = [self._compute_index_for_arm(j, J) for j in J]
        return float(max(indices)) if indices else 0.0

    def gap_index(self) -> float:
        """
        Simplified gap index: difference between k-th and (k+1)-th UCB score.
        This is the "gap between the boundary arms" — converges to large values
        when the top-k set is clearly identified.
        Used as a secondary convergence signal alongside tau_ugape.
        """
        if self.n_arms <= self.k:
            return float("inf")
        ucbs = [self._ucb(i) for i in range(self.n_arms)]
        sorted_ucbs = sorted(ucbs, reverse=True)
        gap = sorted_ucbs[self.k - 1] - sorted_ucbs[self.k]
        return max(gap, 0.0)

    # ── Core bandit interface ─────────────────────────────────────────────────

    def select_arm(self) -> int:
        """
        CASE sampling rule: select the arm that maximizes uncertainty.

        During initialization (before all arms pulled once): round-robin.
        After initialization: pull the arm with the highest UCB that is most
        uncertain relative to the current top-k boundary.
        """
        # Initialization phase: pull each arm at least once
        unpulled = [i for i in range(self.n_arms) if self.counts[i] == 0]
        if unpulled:
            return random.choice(unpulled)

        # Main phase: pull arm with highest UCB (LUCB-style sampling)
        J = self._compute_Jt()
        not_J = [a for a in range(self.n_arms) if a not in J]

        # Best arm in J (by LCB — most uncertain top-k arm)
        if J:
            lcbs_J = [self._lcb(j) for j in J]
            b_t = int(J[int(np.argmin(lcbs_J))])
        else:
            b_t = int(np.argmax([self._ucb(i) for i in range(self.n_arms)]))

        # Challenger arm not in J (highest B_ij(challenger, b_t))
        if not_J:
            b_ij_scores = [self._B_ij(s, b_t) for s in not_J]
            challenger = int(not_J[int(np.argmax(b_ij_scores))])
        else:
            challenger = b_t

        # Pull the arm with larger variance between b_t and challenger
        var_bt = self.sigma * math.sqrt(2.0 * self._beta(max(self.t, 1)) / max(self.counts[b_t], 1))
        var_ch = self.sigma * math.sqrt(2.0 * self._beta(max(self.t, 1)) / max(self.counts[challenger], 1))
        return int(b_t if var_bt >= var_ch else challenger)

    def update(self, arm: int, reward: float) -> None:
        """
        Update bandit statistics after pulling arm and observing reward.

        Args:
            arm: The arm that was pulled (0-indexed).
            reward: Observed reward (typically 0.0 or 1.0 for binary).
        """
        assert 0 <= arm < self.n_arms, f"arm {arm} out of range [0, {self.n_arms})"
        self.counts[arm] += 1
        self.cum_rewards[arm] += reward
        self.t += 1
        self.history.append({"t": self.t, "arm": arm, "reward": reward})

    def converged(self, gap_threshold: float = 0.3) -> bool:
        """
        CASE stopping criterion: stop if UGapE tau <= epsilon AND
        a minimum number of rounds have elapsed.

        Also accepts a gap_threshold shortcut when n_arms is small.

        Args:
            gap_threshold: Secondary gap threshold for early stopping.

        Returns:
            True if the bandit has identified the top-k with high confidence.
        """
        # Must have pulled all arms at least once
        if any(self.counts[i] == 0 for i in range(self.n_arms)):
            return False

        # Hard max rounds
        if self.t >= self.max_rounds:
            return True

        # Primary: UGapE tau <= epsilon
        tau = self._tau_ugape()
        if tau <= self.epsilon:
            return True

        # Secondary: gap_index large (clear separation)
        if self.t >= 2 * self.n_arms and self.gap_index() > gap_threshold:
            return True

        return False

    def top_k_arms(self) -> List[int]:
        """
        Return the indices of the top-k arms by empirical mean reward.
        This is the SPLICE-Select output: indices into demo_candidates.
        """
        means = self.empirical_means()
        # Sort descending, take top-k; convert to plain Python int for JSON
        return [int(i) for i in np.argsort(means)[::-1][:self.k]]

    def diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic info about the bandit state (all JSON-serializable)."""
        means = self.empirical_means()
        J = self._compute_Jt()
        return {
            "t": int(self.t),
            "n_arms": int(self.n_arms),
            "k": int(self.k),
            "empirical_means": [float(x) for x in means],
            "counts": [float(x) for x in self.counts],
            "current_J": [int(x) for x in J],
            "tau_ugape": float(self._tau_ugape()),
            "gap_index": float(self.gap_index()),
            "converged": bool(self.converged()),
            "top_k_arms": self.top_k_arms(),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.counts = np.zeros(self.n_arms, dtype=float)
        self.cum_rewards = np.zeros(self.n_arms, dtype=float)
        self.t = 0
        self.history = []


# ── High-level SPLICE-Select function ────────────────────────────────────────

def splice_select(
    demo_candidates: List[Dict[str, Any]],
    eval_fn: Callable[[int, List[Dict[str, Any]]], float],
    k: int = 3,
    delta: float = 0.05,
    epsilon: float = 0.0,
    sigma: float = 0.5,
    max_rounds: int = 100,
    verbose: bool = False,
) -> Tuple[List[int], CASEBandit]:
    """
    Run SPLICE-Select: CASE bandit to choose top-k demonstrations.

    Args:
        demo_candidates: List of demo dicts from DemoBank.
        eval_fn: Callable(arm_idx, all_candidates) -> float reward.
                 The eval function takes a single arm index and the full
                 candidates list, evaluates the demo, and returns 0/1 reward.
        k: Number of demonstrations to select.
        delta: CASE confidence parameter.
        epsilon: CASE stopping tolerance.
        sigma: Assumed noise std.
        max_rounds: Maximum bandit rounds.
        verbose: Print round-by-round info.

    Returns:
        (top_k_indices, bandit): selected demo indices and fitted bandit.
    """
    n_arms = len(demo_candidates)
    if n_arms == 0:
        return [], CASEBandit(n_arms=1, k=1)

    k = min(k, n_arms)
    bandit = CASEBandit(
        n_arms=n_arms,
        k=k,
        delta=delta,
        epsilon=epsilon,
        sigma=sigma,
        max_rounds=max_rounds,
    )

    for rnd in range(max_rounds):
        if bandit.converged():
            if verbose:
                print(f"  CASE converged at round {rnd} (tau={bandit._tau_ugape():.4f})")
            break

        arm = bandit.select_arm()
        reward = eval_fn(arm, demo_candidates)
        bandit.update(arm, reward)

        if verbose:
            print(
                f"  Round {rnd+1:3d}: arm={arm}, reward={reward:.2f}, "
                f"tau={bandit._tau_ugape():.4f}, gap={bandit.gap_index():.4f}"
            )

    top_k = bandit.top_k_arms()
    if verbose:
        means = bandit.empirical_means()
        print(f"  Selected top-{k}: {top_k} (means: {[round(means[i], 3) for i in top_k]})")

    return top_k, bandit


# ── Saving/loading bandit state ───────────────────────────────────────────────

def save_bandit_state(bandit: CASEBandit, path: str) -> None:
    """Save bandit statistics to JSON."""
    state = {
        "n_arms": bandit.n_arms,
        "k": bandit.k,
        "delta": bandit.delta,
        "epsilon": bandit.epsilon,
        "sigma": bandit.sigma,
        "max_rounds": bandit.max_rounds,
        "counts": bandit.counts.tolist(),
        "cum_rewards": bandit.cum_rewards.tolist(),
        "t": bandit.t,
        "history": bandit.history[-100:],  # last 100 steps
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_bandit_state(path: str) -> CASEBandit:
    """Load bandit statistics from JSON."""
    with open(path, "r") as f:
        state = json.load(f)
    bandit = CASEBandit(
        n_arms=state["n_arms"],
        k=state["k"],
        delta=state["delta"],
        epsilon=state["epsilon"],
        sigma=state["sigma"],
        max_rounds=state["max_rounds"],
    )
    bandit.counts = np.array(state["counts"], dtype=float)
    bandit.cum_rewards = np.array(state["cum_rewards"], dtype=float)
    bandit.t = state["t"]
    bandit.history = state.get("history", [])
    return bandit


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    """
    Demo: simulate SPLICE-Select with random rewards to test the bandit.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test SPLICE-Select bandit")
    parser.add_argument("--n_arms", type=int, default=15)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--max_rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Simulate true arm means (unknown to bandit)
    true_means = np.random.uniform(0.0, 1.0, args.n_arms)
    print(f"True arm means: {true_means.round(3).tolist()}")
    true_top_k = set(np.argsort(true_means)[::-1][: args.k])
    print(f"True top-{args.k}: {sorted(true_top_k)}")

    # Dummy demo candidates
    demo_candidates = [{"task_id": f"task_{i}", "instruction": f"Demo {i}"} for i in range(args.n_arms)]

    # Eval function: Bernoulli reward with true mean
    def eval_fn(arm_idx: int, candidates: List[Dict]) -> float:
        return float(np.random.random() < true_means[arm_idx])

    top_k, bandit = splice_select(
        demo_candidates=demo_candidates,
        eval_fn=eval_fn,
        k=args.k,
        delta=args.delta,
        max_rounds=args.max_rounds,
        verbose=True,
    )

    print(f"\nSPLICE-Select result: top-{args.k} = {top_k}")
    print(f"Correct: {set(top_k) == true_top_k} ({set(top_k)} vs {true_top_k})")
    diag = bandit.diagnostics()
    print(f"Rounds used: {diag['t']}, tau: {diag['tau_ugape']:.4f}")


if __name__ == "__main__":
    main()
