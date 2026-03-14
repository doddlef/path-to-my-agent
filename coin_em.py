"""Simple EM implementation for the two-coin problem.

Problem setup:
- We have two coins, A and B, with unknown head probabilities theta_A and theta_B.
- Each experiment chooses one coin (hidden), flips it several times, and records only
  the number of heads and tails.
- EM estimates both coin biases from the observed counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Trial:
    heads: int
    tails: int

    @property
    def flips(self) -> int:
        return self.heads + self.tails


@dataclass
class EMResult:
    theta_a: float
    theta_b: float
    iterations: int
    history: List[Tuple[int, float, float]]


def likelihood(theta: float, heads: int, tails: int) -> float:
    """Bernoulli likelihood for aggregated flips, ignoring the binomial coefficient.

    For EM responsibility calculations, the shared binomial coefficient cancels out,
    so theta^heads * (1-theta)^tails is enough.
    """
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must be between 0 and 1 (exclusive)")
    return (theta ** heads) * ((1.0 - theta) ** tails)


def em_two_coins(
    trials: Sequence[Trial],
    theta_a: float = 0.6,
    theta_b: float = 0.5,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-8,
    verbose: bool = True,
) -> EMResult:
    """Run EM for the two-coin problem.

    Args:
        trials: Observed experiments as (heads, tails) counts.
        theta_a: Initial guess for coin A head probability.
        theta_b: Initial guess for coin B head probability.
        prior_a: Prior probability of choosing coin A.
        prior_b: Prior probability of choosing coin B.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance on parameter change.
        verbose: Whether to print per-iteration details.

    Returns:
        EMResult containing final parameters and history.
    """
    if not trials:
        raise ValueError("trials must not be empty")
    if not isclose(prior_a + prior_b, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("prior_a + prior_b must equal 1")

    history: List[Tuple[int, float, float]] = []

    for iteration in range(1, max_iter + 1):
        # E-step: compute responsibilities.
        responsibilities = []
        expected_heads_a = 0.0
        expected_tails_a = 0.0
        expected_heads_b = 0.0
        expected_tails_b = 0.0

        for trial in trials:
            prob_a = likelihood(theta_a, trial.heads, trial.tails) * prior_a
            prob_b = likelihood(theta_b, trial.heads, trial.tails) * prior_b
            total = prob_a + prob_b
            if total == 0.0:
                raise ZeroDivisionError("Both probabilities are zero; try different initialization")

            r_a = prob_a / total
            r_b = prob_b / total
            responsibilities.append((r_a, r_b))

            expected_heads_a += r_a * trial.heads
            expected_tails_a += r_a * trial.tails
            expected_heads_b += r_b * trial.heads
            expected_tails_b += r_b * trial.tails

        # M-step: update parameters from expected counts.
        new_theta_a = expected_heads_a / (expected_heads_a + expected_tails_a)
        new_theta_b = expected_heads_b / (expected_heads_b + expected_tails_b)

        history.append((iteration, new_theta_a, new_theta_b))

        if verbose:
            print(f"Iteration {iteration}")
            for i, trial in enumerate(trials, start=1):
                r_a, r_b = responsibilities[i - 1]
                print(
                    f"  Trial {i}: heads={trial.heads}, tails={trial.tails}, "
                    f"P(A|x)={r_a:.4f}, P(B|x)={r_b:.4f}"
                )
            print(
                f"  Expected counts A: heads={expected_heads_a:.4f}, tails={expected_tails_a:.4f}"
            )
            print(
                f"  Expected counts B: heads={expected_heads_b:.4f}, tails={expected_tails_b:.4f}"
            )
            print(f"  Updated theta_A={new_theta_a:.6f}, theta_B={new_theta_b:.6f}")
            print()

        if max(abs(new_theta_a - theta_a), abs(new_theta_b - theta_b)) < tol:
            theta_a, theta_b = new_theta_a, new_theta_b
            return EMResult(theta_a=theta_a, theta_b=theta_b, iterations=iteration, history=history)

        theta_a, theta_b = new_theta_a, new_theta_b

    return EMResult(theta_a=theta_a, theta_b=theta_b, iterations=max_iter, history=history)


def main() -> None:
    # Classic example dataset: five experiments, each with 10 flips.
    # Observed only as head/tail counts.
    raw_trials = [(5, 5), (9, 1), (8, 2), (4, 6), (7, 3)]
    trials = [Trial(heads=h, tails=t) for h, t in raw_trials]

    result = em_two_coins(
        trials,
        theta_a=0.6,
        theta_b=0.5,
        prior_a=0.5,
        prior_b=0.5,
        max_iter=50,
        tol=1e-10,
        verbose=True,
    )

    print("Final result")
    print(f"  iterations: {result.iterations}")
    print(f"  theta_A:    {result.theta_a:.6f}")
    print(f"  theta_B:    {result.theta_b:.6f}")


if __name__ == "__main__":
    main()
