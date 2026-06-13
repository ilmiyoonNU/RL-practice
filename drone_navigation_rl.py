"""
2D Drone Navigation RL — PPO from scratch
Environment: a drone must fly from a random start position to a goal while avoiding obstacles.
State: [dx, dy, vx, vy, ax, ay, dist_to_goal, angle_to_goal]  (relative to goal)
Actions: [thrust_x, thrust_y] continuous in [-1, 1]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ─── Environment ────────────────────────────────────────────────────────────

class DroneNav2D:
    """
    2D drone navigation with wind disturbance and optional obstacles.
    The drone must reach the goal within max_steps without crashing.
    """

    dt = 0.05
    max_thrust = 5.0       # m/s²
    drag = 0.3             # linear drag coefficient
    world_size = 10.0      # metres, square world [-5, 5]
    goal_radius = 0.3
    crash_radius = 0.3
    max_steps = 500
    wind_std = 0.1         # stochastic wind noise

    def __init__(self, n_obstacles=3):
        self.n_obstacles = n_obstacles
        self.obs_dim = 8 + n_obstacles * 3   # drone state + obstacle (dx, dy, dist) each
        self.act_dim = 2

    def reset(self):
        half = self.world_size / 2
        self.pos = np.random.uniform(-half * 0.8, half * 0.8, 2)
        self.vel = np.zeros(2)

        # goal on opposite side of world
        self.goal = -self.pos + np.random.uniform(-1, 1, 2)
        self.goal = np.clip(self.goal, -half * 0.9, half * 0.9)

        # place obstacles not too close to start or goal
        self.obstacles = []
        for _ in range(self.n_obstacles):
            for _ in range(50):
                c = np.random.uniform(-half * 0.7, half * 0.7, 2)
                r = np.random.uniform(0.3, 0.7)
                if (np.linalg.norm(c - self.pos) > r + 0.5 and
                        np.linalg.norm(c - self.goal) > r + 0.5):
                    self.obstacles.append((c, r))
                    break

        self.step_count = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        return self._obs()

    def _obs(self):
        rel = self.goal - self.pos
        dist = np.linalg.norm(rel)
        angle = np.arctan2(rel[1], rel[0])
        state = np.array([rel[0], rel[1], self.vel[0], self.vel[1],
                          dist, np.cos(angle), np.sin(angle),
                          self.step_count / self.max_steps], dtype=np.float32)
        obs_parts = [state]
        for obs_c, obs_r in self.obstacles:
            d = self.pos - obs_c
            obs_parts.append(np.array([d[0], d[1], np.linalg.norm(d) - obs_r], dtype=np.float32))
        # pad if fewer obstacles than expected
        while len(obs_parts) < 1 + self.n_obstacles:
            obs_parts.append(np.array([99.0, 99.0, 99.0], dtype=np.float32))
        return np.concatenate(obs_parts)

    def step(self, action):
        thrust = np.clip(action, -1.0, 1.0) * self.max_thrust
        wind = np.random.normal(0, self.wind_std, 2)
        accel = thrust + wind - self.drag * self.vel

        self.vel += accel * self.dt
        self.pos += self.vel * self.dt
        self.pos = np.clip(self.pos, -self.world_size / 2, self.world_size / 2)
        self.step_count += 1

        dist = np.linalg.norm(self.goal - self.pos)

        # reward shaping
        r_progress = (self.prev_dist - dist) * 10.0
        r_goal = 200.0 if dist < self.goal_radius else 0.0
        r_energy = -0.001 * np.sum(thrust ** 2)
        r_boundary = -5.0 if np.any(np.abs(self.pos) >= self.world_size / 2 - 0.1) else 0.0
        reward = r_progress + r_goal + r_energy + r_boundary

        # obstacle collision
        crashed = False
        for obs_c, obs_r in self.obstacles:
            if np.linalg.norm(self.pos - obs_c) < obs_r + self.crash_radius:
                reward -= 100.0
                crashed = True
                break

        self.prev_dist = dist
        reached = dist < self.goal_radius
        done = reached or crashed or self.step_count >= self.max_steps

        return self._obs(), reward, done, {
            "reached": reached, "crashed": crashed, "dist": dist
        }

    def render_episode(self, policy_net, save_path="drone_episode.png"):
        """Run one episode and plot the trajectory."""
        obs = self.reset()
        trajectory = [self.pos.copy()]
        done = False
        while not done:
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist_obj, _ = policy_net(o)
                action = dist_obj.mean.squeeze(0).numpy()
            obs, _, done, _ = self.step(action)
            trajectory.append(self.pos.copy())

        traj = np.array(trajectory)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_title("Drone Navigation Episode")

        for obs_c, obs_r in self.obstacles:
            ax.add_patch(patches.Circle(obs_c, obs_r, color="red", alpha=0.5))

        ax.add_patch(patches.Circle(self.goal, self.goal_radius, color="green", alpha=0.7, label="Goal"))
        ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.5, label="Trajectory")
        ax.plot(traj[0, 0], traj[0, 1], "bs", markersize=8, label="Start")
        ax.plot(traj[-1, 0], traj[-1, 1], "b^", markersize=8, label="End")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"Episode trajectory saved to {save_path}")


# ─── PPO Actor-Critic ────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = torch.tanh(self.actor_mean(h))
        std = self.actor_log_std.exp().clamp(1e-3, 0.8)
        return Normal(mean, std), self.critic(h).squeeze(-1)

    def act(self, obs):
        dist, value = self(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value


# ─── PPO Trainer ─────────────────────────────────────────────────────────────

class PPO:
    def __init__(self, obs_dim, act_dim,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=10, batch_size=64):
        self.net = ActorCritic(obs_dim, act_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma; self.lam = lam
        self.clip_eps = clip_eps; self.epochs = epochs; self.batch_size = batch_size

    def collect_rollout(self, env, steps=2048):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        obs = env.reset()
        for _ in range(steps):
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act, logp, val = self.net.act(o)
            a = act.squeeze(0).numpy()
            next_obs, rew, done, _ = env.step(a)
            obs_buf.append(obs); act_buf.append(a)
            logp_buf.append(logp.item()); rew_buf.append(rew)
            val_buf.append(val.item()); done_buf.append(done)
            obs = env.reset() if done else next_obs
        return (np.array(obs_buf, np.float32), np.array(act_buf, np.float32),
                np.array(logp_buf, np.float32), np.array(rew_buf, np.float32),
                np.array(val_buf, np.float32), np.array(done_buf, np.float32))

    def compute_gae(self, rewards, values, dones, last_val=0.0):
        advantages, gae = np.zeros_like(rewards), 0.0
        for t in reversed(range(len(rewards))):
            nv = last_val if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages, advantages + values

    def update(self, obs, acts, logps_old, advs, rets):
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        obs_t = torch.tensor(obs); acts_t = torch.tensor(acts)
        logps_t = torch.tensor(logps_old); advs_t = torch.tensor(advs); rets_t = torch.tensor(rets)
        pls, vls = [], []
        for _ in range(self.epochs):
            idx = np.random.permutation(len(obs))
            for s in range(0, len(obs), self.batch_size):
                b = idx[s:s + self.batch_size]
                d, vals = self.net(obs_t[b])
                ratio = (d.log_prob(acts_t[b]).sum(-1) - logps_t[b]).exp()
                s1 = ratio * advs_t[b]
                s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advs_t[b]
                pl = -torch.min(s1, s2).mean()
                vl = (vals - rets_t[b]).pow(2).mean()
                loss = pl + 0.5 * vl - 0.01 * d.entropy().sum(-1).mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                pls.append(pl.item()); vls.append(vl.item())
        return np.mean(pls), np.mean(vls)

    def train(self, env, total_steps=300_000, rollout_steps=2048):
        returns, success_rates = [], []
        steps_done = 0

        print("Training 2D Drone Navigator with PPO")
        print(f"{'Step':>10}  {'Return':>10}  {'SuccessRate':>12}  {'PLoss':>10}  {'VLoss':>8}")

        while steps_done < total_steps:
            obs_b, act_b, logp_b, rew_b, val_b, done_b = self.collect_rollout(env, rollout_steps)
            adv_b, ret_b = self.compute_gae(rew_b, val_b, done_b)
            pl, vl = self.update(obs_b, act_b, logp_b, adv_b, ret_b)

            # estimate success rate from this rollout
            ep_ret, successes, episodes = 0.0, 0, 0
            ep_ret = rew_b.sum()
            returns.append(ep_ret)
            sr = successes / max(episodes, 1)
            success_rates.append(sr)
            steps_done += rollout_steps
            print(f"{steps_done:>10}  {ep_ret:>10.1f}  {sr:>12.2%}  {pl:>10.4f}  {vl:>8.4f}")

        return returns, success_rates

    def save(self, path="drone_navigator.pth"):
        torch.save(self.net.state_dict(), path)
        print(f"Model saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def plot_training(returns, success_rates):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    w = max(1, len(returns) // 20)
    sm_ret = np.convolve(returns, np.ones(w) / w, mode="valid")
    sm_sr = np.convolve(success_rates, np.ones(w) / w, mode="valid")

    ax1.plot(returns, alpha=0.3); ax1.plot(sm_ret)
    ax1.set_title("Return per Rollout"); ax1.set_xlabel("Rollout"); ax1.set_ylabel("Return")

    ax2.plot(success_rates, alpha=0.3); ax2.plot(sm_sr)
    ax2.set_title("Success Rate"); ax2.set_xlabel("Rollout"); ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("drone_training.png", dpi=150)
    plt.show()
    print("Training curves saved to drone_training.png")


if __name__ == "__main__":
    env = DroneNav2D(n_obstacles=3)
    agent = PPO(env.obs_dim, env.act_dim)
    returns, success_rates = agent.train(env, total_steps=300_000)
    agent.save()
    plot_training(returns, success_rates)
    env.render_episode(agent.net)   # visualize one episode
