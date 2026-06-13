"""
BipedalWalker-v3 using Stable-Baselines3 PPO.
This is a proven implementation that successfully learns to walk.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os


def train():
    env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    eval_env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3")])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    print("Training BipedalWalker-v3 with SB3 PPO...")
    print("This typically solves (~300 reward) in about 1-2M steps.\n")
    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    model.save("bipedal_walker_sb3")
    env.save("vec_normalize.pkl")
    print("\nModel saved to bipedal_walker_sb3.zip")


def watch():
    from stable_baselines3.common.vec_env import VecNormalize
    import time

    env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3", render_mode="human")])
    env = VecNormalize.load("vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False

    model = PPO.load("best_model/best_model", env=env)

    for ep in range(5):
        obs = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            time.sleep(0.01)
        print(f"Episode {ep+1}: return = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch()
    else:
        train()
