"""
BipedalWalker-v3 using Stable-Baselines3 PPO.
Includes early stopping when performance degrades — no more 4-hour wasted runs.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ─── Early Stop Callback ─────────────────────────────────────────────────────

class EarlyStopCallback(BaseCallback):
    """
    Stops training if mean reward drops more than `threshold` below
    the best reward seen so far, for `patience` consecutive evaluations.

    Example: best=250, threshold=50 → stops if reward drops below 200
    for 3 evals in a row.
    """
    def __init__(self, threshold=50, patience=3, verbose=1):
        super().__init__(verbose)
        self.threshold  = threshold
        self.patience   = patience
        self.best       = -np.inf
        self.bad_evals  = 0

    def _on_step(self):
        # triggered by EvalCallback via on_event
        return True

    def on_event(self, reward):
        if reward > self.best:
            self.best      = reward
            self.bad_evals = 0
            if self.verbose:
                print(f"  [EarlyStop] New best: {self.best:.1f}")
        elif reward < self.best - self.threshold:
            self.bad_evals += 1
            if self.verbose:
                print(f"  [EarlyStop] Degraded eval {self.bad_evals}/{self.patience} "
                      f"(best={self.best:.1f}, now={reward:.1f})")
            if self.bad_evals >= self.patience:
                print(f"\n  [EarlyStop] Stopping — reward dropped {self.threshold} "
                      f"below best for {self.patience} evals in a row.")
                return False   # stop training
        else:
            self.bad_evals = 0
        return True


class EvalWithEarlyStop(EvalCallback):
    """EvalCallback that also triggers EarlyStopCallback."""

    def __init__(self, *args, early_stop_cb=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop_cb = early_stop_cb

    def _on_step(self):
        result = super()._on_step()
        if self.early_stop_cb and hasattr(self, "last_mean_reward"):
            keep_going = self.early_stop_cb.on_event(self.last_mean_reward)
            if not keep_going:
                return False
        return result


# ─── Train ───────────────────────────────────────────────────────────────────

def train():
    env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy", env,
        learning_rate = 1e-4,   # slower lr → more stable late training
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.1,    # tighter clip → smaller policy steps
        ent_coef      = 0.0,    # no entropy — was causing instability
        max_grad_norm = 0.3,    # stricter gradient clipping
        verbose       = 1,
        tensorboard_log = "./tb_logs/",
        policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    )

    eval_env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3")])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False)

    early_stop = EarlyStopCallback(threshold=50, patience=3, verbose=1)

    eval_cb = EvalWithEarlyStop(
        eval_env,
        early_stop_cb        = early_stop,
        best_model_save_path = "./best_model/",
        log_path             = "./logs/",
        eval_freq            = 10_000,
        n_eval_episodes      = 10,      # more episodes → more reliable estimate
        deterministic        = True,
        verbose              = 1,
    )

    print("Training BipedalWalker-v3 with PPO + Early Stopping")
    print("Will stop automatically if performance degrades.\n")
    model.learn(total_timesteps=3_000_000, callback=eval_cb)

    model.save("bipedal_walker_sb3")
    env.save("vec_normalize.pkl")
    print("\nFinal model saved. Best model is in best_model/best_model.zip")


# ─── Watch ───────────────────────────────────────────────────────────────────

def watch():
    import time
    env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3", render_mode="human")])
    env = VecNormalize.load("vec_normalize.pkl", env)
    env.training   = False
    env.norm_reward= False
    model = PPO.load("best_model/best_model", env=env)

    print("Watching best saved model...\n")
    for ep in range(5):
        obs = env.reset(); total = 0.0; done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += reward[0]
            time.sleep(0.01)
        print(f"Episode {ep+1}: return = {total:.1f}")
    env.close()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch()
    else:
        train()
