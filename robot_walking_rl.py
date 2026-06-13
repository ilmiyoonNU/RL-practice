"""
2D Robot Walking RL — PPO from scratch
Environment: a bipedal robot must move forward by applying torques to hip and knee joints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation


# ─── Environment ────────────────────────────────────────────────────────────

class BipedalWalker2D:
    """
    Simplified 2D bipedal robot.
    State: [torso_angle, torso_vel, hip1, knee1, hip2, knee2, foot1_contact, foot2_contact]
    Actions: torques on [hip1, knee1, hip2, knee2] in [-1, 1]
    """

    dt = 0.02
    gravity = 9.8
    max_torque = 150.0
    max_steps = 1000

    def __init__(self):
        self.obs_dim = 8
        self.act_dim = 4

    def reset(self):
        self.step_count = 0
        self.x_pos = 0.0
        self.torso_angle = 0.0
        self.torso_vel = 0.0
        self.joint_angles = np.zeros(4)   # hip1, knee1, hip2, knee2
        self.joint_vels = np.zeros(4)
        self.foot_contact = np.array([1.0, 0.0])
        return self._obs()

    def _obs(self):
        return np.concatenate([
            [self.torso_angle, self.torso_vel],
            self.joint_angles,
            self.foot_contact,
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        torques = action * self.max_torque

        # simplified physics: joints accelerate proportional to torque
        inertia = np.array([5.0, 2.0, 5.0, 2.0])
        self.joint_vels += (torques / inertia - 0.1 * self.joint_vels) * self.dt
        self.joint_angles += self.joint_vels * self.dt
        self.joint_angles = np.clip(self.joint_angles, -np.pi / 2, np.pi / 2)

        # torso stability influenced by leg asymmetry
        hip_diff = self.joint_angles[0] - self.joint_angles[2]
        self.torso_vel += (-self.gravity * np.sin(self.torso_angle) * 0.1
                           + hip_diff * 0.3) * self.dt
        self.torso_angle += self.torso_vel * self.dt
        self.torso_angle = np.clip(self.torso_angle, -np.pi / 3, np.pi / 3)

        # forward progress from leg swing
        forward_vel = (np.cos(self.joint_angles[0]) - np.cos(self.joint_angles[2])) * 0.5
        self.x_pos += forward_vel * self.dt

        # alternating foot contact
        self.foot_contact = np.array([
            float(self.joint_angles[1] < -0.1),
            float(self.joint_angles[3] < -0.1),
        ])

        # rewards
        r_forward = forward_vel * 10.0
        r_stability = -abs(self.torso_angle) * 2.0
        r_torque = -0.001 * np.sum(torques ** 2)
        reward = r_forward + r_stability + r_torque

        self.step_count += 1
        fallen = abs(self.torso_angle) > np.pi / 3
        done = fallen or self.step_count >= self.max_steps

        if fallen:
            reward -= 100.0

        return self._obs(), reward, done, {"x_pos": self.x_pos}

    def get_body_parts(self):
        """Return (x, y) positions of body parts for rendering."""
        # torso center at (x_pos, 1.0), tilted by torso_angle
        tx, ty = self.x_pos, 1.0
        torso_len = 0.5
        tx2 = tx + torso_len * np.sin(self.torso_angle)
        ty2 = ty + torso_len * np.cos(self.torso_angle)

        # hip joints at bottom of torso
        h1x = tx - 0.2 * np.cos(self.torso_angle)
        h1y = ty - 0.2 * np.sin(self.torso_angle) - 0.3
        h2x = tx + 0.2 * np.cos(self.torso_angle)
        h2y = ty + 0.2 * np.sin(self.torso_angle) - 0.3

        thigh_len, shin_len = 0.4, 0.35

        # leg 1
        k1x = h1x + thigh_len * np.sin(self.joint_angles[0])
        k1y = h1y - thigh_len * np.cos(self.joint_angles[0])
        f1x = k1x + shin_len * np.sin(self.joint_angles[0] + self.joint_angles[1])
        f1y = k1y - shin_len * np.cos(self.joint_angles[0] + self.joint_angles[1])

        # leg 2
        k2x = h2x + thigh_len * np.sin(self.joint_angles[2])
        k2y = h2y - thigh_len * np.cos(self.joint_angles[2])
        f2x = k2x + shin_len * np.sin(self.joint_angles[2] + self.joint_angles[3])
        f2y = k2y - shin_len * np.cos(self.joint_angles[2] + self.joint_angles[3])

        return {
            "torso": ([tx, tx2], [ty, ty2]),
            "head": (tx2, ty2 + 0.15),
            "leg1": ([h1x, k1x, f1x], [h1y, k1y, f1y]),
            "leg2": ([h2x, k2x, f2x], [h2y, k2y, f2y]),
            "foot1_contact": self.foot_contact[0],
            "foot2_contact": self.foot_contact[1],
        }


# ─── PPO Actor-Critic ────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = torch.tanh(self.actor_mean(h))
        std = self.actor_log_std.exp().clamp(1e-3, 1.0)
        return Normal(mean, std), self.critic(h).squeeze(-1)

    def act(self, obs):
        dist, value = self(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value


# ─── PPO Trainer ─────────────────────────────────────────────────────────────

class PPO:
    def __init__(self, obs_dim, act_dim,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, epochs=10, batch_size=64):
        self.net = ActorCritic(obs_dim, act_dim)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

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
        return (np.array(obs_buf, dtype=np.float32), np.array(act_buf, dtype=np.float32),
                np.array(logp_buf, dtype=np.float32), np.array(rew_buf, dtype=np.float32),
                np.array(val_buf, dtype=np.float32), np.array(done_buf, dtype=np.float32))

    def compute_gae(self, rewards, values, dones, last_val=0.0):
        advantages, gae = np.zeros_like(rewards), 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_val if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, obs, acts, logps_old, advs, rets):
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        obs_t = torch.tensor(obs); acts_t = torch.tensor(acts)
        logps_t = torch.tensor(logps_old); advs_t = torch.tensor(advs); rets_t = torch.tensor(rets)

        policy_losses, value_losses = [], []
        for _ in range(self.epochs):
            idx = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                b = idx[start:start + self.batch_size]
                dist, vals = self.net(obs_t[b])
                logps_new = dist.log_prob(acts_t[b]).sum(-1)
                ratio = (logps_new - logps_t[b]).exp()
                surr1 = ratio * advs_t[b]
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advs_t[b]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = (vals - rets_t[b]).pow(2).mean()
                loss = p_loss + 0.5 * v_loss - 0.01 * dist.entropy().sum(-1).mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                policy_losses.append(p_loss.item()); value_losses.append(v_loss.item())
        return np.mean(policy_losses), np.mean(value_losses)

    def train(self, env, total_steps=200_000, rollout_steps=2048):
        ep_returns, steps_done = [], 0
        ep_ret, obs = 0.0, env.reset()

        print("Training 2D Robot Walker with PPO")
        print(f"{'Step':>10}  {'MeanReturn':>12}  {'PolicyLoss':>12}  {'ValueLoss':>10}")

        while steps_done < total_steps:
            obs_b, act_b, logp_b, rew_b, val_b, done_b = self.collect_rollout(env, rollout_steps)
            adv_b, ret_b = self.compute_gae(rew_b, val_b, done_b)
            pl, vl = self.update(obs_b, act_b, logp_b, adv_b, ret_b)

            ep_ret = rew_b.sum()
            ep_returns.append(ep_ret)
            steps_done += rollout_steps
            print(f"{steps_done:>10}  {ep_ret:>12.1f}  {pl:>12.4f}  {vl:>10.4f}")

        return ep_returns

    def save(self, path="robot_walker.pth"):
        torch.save(self.net.state_dict(), path)
        print(f"Model saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def watch_robot(env, net, episodes=3):
    """Render the robot walking live using matplotlib."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_ylim(-0.2, 2.5)
    ax.set_title("Robot Walking — Live View")
    ax.set_xlabel("Position (m)")
    ax.axhline(0, color="brown", linewidth=3, label="Ground")

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, _ = net(o)
                action = dist.mean.squeeze(0).numpy()
            obs, reward, done, info = env.step(action)
            total_reward += reward

            ax.clear()
            ax.set_ylim(-0.2, 2.5)
            ax.set_title(f"Robot Walking — Episode {ep+1}  |  x={info['x_pos']:.2f}m  |  return={total_reward:.1f}")
            ax.set_xlabel("Position (m)")
            ax.axhline(0, color="brown", linewidth=3)

            # draw robot
            bp = env.get_body_parts()
            ax.plot(*bp["torso"], "b-", linewidth=6)          # torso
            ax.plot(bp["head"][0], bp["head"][1], "bo",
                    markersize=14)                              # head
            c1 = "green" if bp["foot1_contact"] else "orange"
            c2 = "green" if bp["foot2_contact"] else "orange"
            ax.plot(*bp["leg1"], color=c1, linewidth=4)       # leg 1
            ax.plot(*bp["leg2"], color=c2, linewidth=4)       # leg 2

            # ground markers
            ax.set_xlim(info["x_pos"] - 2, info["x_pos"] + 2)
            legend = [mpatches.Patch(color="green", label="Foot contact"),
                      mpatches.Patch(color="orange", label="Foot in air")]
            ax.legend(handles=legend, loc="upper right")
            plt.pause(0.02)

    plt.ioff()
    plt.show()


def plot_returns(returns, title="Robot Walking — PPO Training"):
    window = max(1, len(returns) // 20)
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(10, 4))
    plt.plot(returns, alpha=0.3, label="Raw return")
    plt.plot(smoothed, label=f"Smoothed (w={window})")
    plt.xlabel("Rollout"); plt.ylabel("Return"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig("robot_walking_training.png", dpi=150)
    plt.show()
    print("Training curve saved to robot_walking_training.png")


if __name__ == "__main__":
    import sys
    env = BipedalWalker2D()

    # python robot_walking_rl.py watch  → load saved model and watch it walk
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        net = ActorCritic(env.obs_dim, env.act_dim)
        net.load_state_dict(torch.load("robot_walker.pth"))
        net.eval()
        watch_robot(env, net, episodes=5)
    else:
        agent = PPO(env.obs_dim, env.act_dim)
        returns = agent.train(env, total_steps=200_000)
        agent.save()
        plot_returns(returns)
        print("\nTo watch the trained robot walk, run:")
        print("  python robot_walking_rl.py watch")
