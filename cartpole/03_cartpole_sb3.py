"""
CartPole — Step 3: PPO with Stable-Baselines3
===============================================
Now we use SB3 — a professional RL library — to solve the same problem.
Compare with 02_cartpole_ppo.py to see what a production library adds:
  - Vectorised environments (run N envs in parallel)
  - Automatic observation normalisation
  - TensorBoard logging
  - Evaluation callbacks
  - One line to switch algorithms (PPO → SAC → A2C → ...)

Key lesson: the algorithm is the same PPO — but the engineering around it
matters a lot for reliability and speed.
"""

import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np


# ─── Train with PPO ──────────────────────────────────────────────────────────

def train_ppo():
    # vectorised env: run 4 envs in parallel → 4× faster data collection
    env = make_vec_env("CartPole-v1", n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate   = 3e-4,
        n_steps         = 512,      # steps per env before update
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.0,
        verbose         = 1,
    )

    # stop early when mean reward >= 490
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=490, verbose=1)
    eval_cb = EvalCallback(
        make_vec_env("CartPole-v1", n_envs=1),
        callback_on_new_best = stop_cb,
        eval_freq  = 2_000,
        best_model_save_path = "./cartpole_best/",
        verbose    = 1,
    )

    print("Training CartPole with SB3 PPO...")
    model.learn(total_timesteps=50_000, callback=eval_cb)
    model.save("cartpole_sb3_ppo")
    print("Saved cartpole_sb3_ppo.zip")
    return model


# ─── Compare algorithms ───────────────────────────────────────────────────────

def compare_algorithms(n_steps=30_000):
    """
    Train multiple algorithms on CartPole and compare.
    This shows students that PPO is not the only option.
    """
    algorithms = {
        "PPO": PPO("MlpPolicy", make_vec_env("CartPole-v1", n_envs=4),
                   verbose=0, n_steps=512),
        "A2C": A2C("MlpPolicy", make_vec_env("CartPole-v1", n_envs=4),
                   verbose=0, n_steps=128),
    }

    results = {}
    for name, model in algorithms.items():
        print(f"Training {name}...")
        rewards_per_eval = []

        class RewardTracker:
            def __init__(self):
                self.rewards = []
            def __call__(self, locals_, globals_):
                if locals_.get("iteration", 0) % 5 == 0:
                    mean_r, _ = evaluate_policy(model,
                                    make_vec_env("CartPole-v1"),
                                    n_eval_episodes=10, warn=False)
                    self.rewards.append(mean_r)
                return True

        tracker = RewardTracker()
        model.learn(total_timesteps=n_steps, callback=tracker)
        results[name] = tracker.rewards
        mean, std = evaluate_policy(model,
                        make_vec_env("CartPole-v1"),
                        n_eval_episodes=20, warn=False)
        print(f"  {name} final: {mean:.1f} ± {std:.1f}")

    # plot comparison
    plt.figure(figsize=(10, 4))
    for name, rewards in results.items():
        plt.plot(rewards, label=name, linewidth=2)
    plt.axhline(500, color="green", ls="--", alpha=0.5, label="Max (500)")
    plt.axhline(195, color="red",   ls="--", alpha=0.5, label="Threshold (195)")
    plt.xlabel("Evaluation"); plt.ylabel("Mean Return")
    plt.title("Algorithm Comparison — CartPole")
    plt.legend(); plt.tight_layout()
    plt.savefig("03_algorithm_comparison.png", dpi=150)
    plt.show()
    print("Saved 03_algorithm_comparison.png")
    return results


# ─── Watch trained agent ──────────────────────────────────────────────────────

def watch(model_path="cartpole_best/best_model", episodes=5):
    model = PPO.load(model_path)
    env   = gym.make("CartPole-v1", render_mode="human")

    print(f"Watching {episodes} episodes...")
    for ep in range(episodes):
        obs, _ = env.reset()
        done   = False
        total  = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            total += r
        print(f"Episode {ep+1}: {total:.0f} steps")

    env.close()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch()
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_algorithms()
    else:
        train_ppo()
        print("\nCommands:")
        print("  python 03_cartpole_sb3.py watch    → watch trained agent")
        print("  python 03_cartpole_sb3.py compare  → compare PPO vs A2C")
