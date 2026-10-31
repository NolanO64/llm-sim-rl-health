"""The existing growing-batch Q refit, shared by LLM and statistical worlds."""
import numpy as np


GAMMA = 0.9
EPISODES_PER_ROUND = 40
Q_REFIT_PASSES = 60
Q_ALPHA = 0.1


def refit_q(buffer, passes=Q_REFIT_PASSES, alpha=Q_ALPHA):
    Q = np.zeros((2, 4, 4))
    rng = np.random.default_rng(0)
    order = list(range(len(buffer)))
    for _ in range(passes):
        rng.shuffle(order)
        for j in order:
            c, b, a, r, next_c, next_b, done = buffer[j]
            target = r if done else r + GAMMA * Q[next_c, next_b].max()
            Q[c, b, a] += alpha * (target - Q[c, b, a])
    return Q
