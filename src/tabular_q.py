"""Tabular Q-learning for StepCountJITAI.

Two training regimes are used in the thesis:

* offline -- batch Q-learning over a fixed set of transitions extracted from an
  LLM-world corpus (the deployable learner never touches the real environment);
* online -- epsilon-greedy Q-learning inside the real environment, used both as a
  trained-on-reality reference and to obtain the privileged full-state ceiling.

The deployable state is (inferred context, message-streak bucket). The full-state
reference additionally uses binned habituation and disengagement, which are only
available to an agent allowed to see the environment's hidden variables.
"""
import numpy as np

from .stepcount import MAX_EPISODE_LENGTH, make_env, message_bucket

GAMMA = 0.9
CONTEXT_INDEX = {"A": 0, "B": 1}


def corpus_to_transitions(corpus):
    """Flatten an LLM-world corpus into ``(c, bucket, a, r, c', bucket', done)`` tuples."""
    transitions = []
    for patient in corpus:
        steps = []
        streak = 0
        for record in patient.get("trajectory", []):
            context = CONTEXT_INDEX.get(str(record.get("context", "A")).upper()[:1], 0)
            action = int(record["action"])
            reward = float(record["outcome"])
            steps.append((context, message_bucket(streak), action, reward))
            streak = 0 if action == 0 else streak + 1
            if record.get("quit"):
                break
        for i, (c, b, a, r) in enumerate(steps):
            if i + 1 < len(steps):
                next_c, next_b = steps[i + 1][0], steps[i + 1][1]
                transitions.append((c, b, a, r, next_c, next_b, False))
            else:
                transitions.append((c, b, a, r, 0, 0, True))
    return transitions


def train_offline_q(transitions, seed, passes=120, alpha=0.1, gamma=GAMMA):
    """Batch Q-learning: repeated shuffled sweeps over a fixed transition set."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((2, 4, 4))
    order = list(range(len(transitions)))
    for _ in range(passes):
        rng.shuffle(order)
        for j in order:
            c, b, a, r, next_c, next_b, done = transitions[j]
            target = r if done else r + gamma * Q[next_c, next_b].max()
            Q[c, b, a] += alpha * (target - Q[c, b, a])
    return Q


def train_online_q(config, seed, n_episodes=25000, alpha=0.1, gamma=GAMMA):
    """Epsilon-greedy Q-learning in the real environment on the observable state."""
    env = make_env(seed, config)
    rng = np.random.default_rng(seed)
    Q = np.zeros((2, 4, 4))
    for episode in range(n_episodes):
        env.reset()
        streak = 0
        eps = max(0.05, 1.0 - episode / (0.7 * n_episodes))
        for _ in range(MAX_EPISODE_LENGTH):
            obs = env.unpack_obs_array(env.current_state)
            c = int(obs["c_infer"])
            b = message_bucket(streak)
            if rng.random() < eps:
                action = int(rng.integers(4))
            else:
                action = int(Q[c, b].argmax())
            next_streak = 0 if action == 0 else streak + 1
            _, reward, terminated, truncated, _ = env.step(action)
            next_obs = env.unpack_obs_array(env.current_state)
            next_c = int(next_obs["c_infer"])
            next_b = message_bucket(next_streak)
            done = terminated or truncated
            target = float(reward) if done else float(reward) + gamma * Q[next_c, next_b].max()
            Q[c, b, action] += alpha * (target - Q[c, b, action])
            streak = next_streak
            if done:
                break
    return Q


def make_state_bucketers(config, seed=123, n_episodes=120):
    """Quantile bin edges for habituation and disengagement, from random rollouts.

    These are needed only by the privileged full-state reference; the deployable
    policies never see these variables.
    """
    env = make_env(seed, config)
    rng = np.random.default_rng(0)
    h_values, d_values = [], []
    for _ in range(n_episodes):
        env.reset()
        for _ in range(MAX_EPISODE_LENGTH):
            obs = env.unpack_obs_array(env.current_state)
            h_values.append(float(obs["h"]))
            d_values.append(float(obs["d"]))
            _, _, terminated, truncated, _ = env.step(int(rng.integers(4)))
            if terminated or truncated:
                break
    h_edges = np.quantile(h_values, [0.2, 0.4, 0.6, 0.8])
    d_edges = np.quantile(d_values, [0.33, 0.66])
    h_bucket = lambda h: int(np.digitize(h, h_edges))
    d_bucket = lambda d: int(np.digitize(d, d_edges))
    return h_bucket, d_bucket


def train_fullstate_q(config, h_bucket, d_bucket, seed, n_episodes=12000, alpha=0.1, gamma=GAMMA):
    """Privileged Q-learning on (true context, habituation bucket, disengagement bucket)."""
    env = make_env(seed, config)
    rng = np.random.default_rng(seed)
    Q = np.zeros((2, 5, 3, 4))
    for episode in range(n_episodes):
        env.reset()
        eps = max(0.05, 1.0 - episode / (0.7 * n_episodes))
        for _ in range(MAX_EPISODE_LENGTH):
            obs = env.unpack_obs_array(env.current_state)
            ct, h, d = int(obs["c_true"]), h_bucket(obs["h"]), d_bucket(obs["d"])
            if rng.random() < eps:
                action = int(rng.integers(4))
            else:
                action = int(Q[ct, h, d].argmax())
            _, reward, terminated, truncated, _ = env.step(action)
            next_obs = env.unpack_obs_array(env.current_state)
            next_ct = int(next_obs["c_true"])
            next_h, next_d = h_bucket(next_obs["h"]), d_bucket(next_obs["d"])
            done = terminated or truncated
            target = float(reward) if done else float(reward) + gamma * Q[next_ct, next_h, next_d].max()
            Q[ct, h, d, action] += alpha * (target - Q[ct, h, d, action])
            if done:
                break
    return Q


def greedy_policy(Q):
    """Deterministic greedy policy on the observable state from a Q-table."""
    Q = np.asarray(Q)

    def policy(obs, streak, rng):
        return int(Q[int(obs["c_infer"]), message_bucket(streak)].argmax())

    return policy


def epsilon_blend_policy(Q, eps):
    """Greedy with probability ``eps``, otherwise no action.

    Used to manufacture a spread of intermediate-quality policies from a single
    trained Q-table, so the benchmark sees more than a handful of distinct returns.
    """
    Q = np.asarray(Q)

    def policy(obs, streak, rng):
        if rng.random() < eps:
            return int(Q[int(obs["c_infer"]), message_bucket(streak)].argmax())
        return 0

    return policy


def greedy_fullstate_policy(Q, h_bucket, d_bucket):
    """Deterministic greedy policy on the privileged full state."""
    Q = np.asarray(Q)

    def policy(obs, streak, rng):
        return int(Q[int(obs["c_true"]), h_bucket(obs["h"]), d_bucket(obs["d"])].argmax())

    return policy
