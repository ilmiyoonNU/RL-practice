"""
CartPole — Step 2: PPO from Scratch
=====================================
We implement PPO step by step. Every component is explained.
By the end, the agent scores 500/500 — perfectly balancing the pole.

PPO in a nutshell:
  1. Collect experience using the current policy (rollout)
  2. Compute how good each action was (advantage estimation via GAE)
  3. Update the policy — but not too aggressively (clipped objective)
  4. Repeat

Why "Proximal"? We constrain how much the policy can change per update.
Too large a step → policy collapses. PPO's clip prevents this.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


# ─── CartPole Environment ────────────────────────────────────────────────────
# (same as 01_cartpole_random.py — no gym needed)

class CartPole:
    gravity = 9.8;  mass_cart = 1.0;  mass_pole = 0.1
    pole_half = 0.5; force_mag = 10.0; dt = 0.02; max_steps = 500
    angle_limit = 12 * np.pi / 180;   pos_limit = 2.4

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.state.astype(np.float32)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        total_mass = self.mass_cart + self.mass_pole
        pml = self.mass_pole * self.pole_half
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        temp     = (force + pml * theta_dot**2 * sin_t) / total_mass
        theta_acc= (self.gravity*sin_t - cos_t*temp) / (
                    self.pole_half*(4/3 - self.mass_pole*cos_t**2/total_mass))
        x_acc    = temp - pml * theta_acc * cos_t / total_mass
        x        += self.dt*x_dot;     x_dot     += self.dt*x_acc
        theta    += self.dt*theta_dot; theta_dot += self.dt*theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        done = (abs(x)>self.pos_limit or abs(theta)>self.angle_limit
                or self.steps>=self.max_steps)
        return self.state.astype(np.float32), 1.0, done

    def render_state(self):
        x, _, theta, _ = self.state
        return x, 0.0, x+2*self.pole_half*np.sin(theta), 2*self.pole_half*np.cos(theta)


# ─── COMPONENT 1: Actor-Critic Network ───────────────────────────────────────
#
# One network, two heads:
#   Actor  → outputs action probabilities  (WHAT to do)
#   Critic → outputs state value V(s)      (HOW GOOD is this state)
#
# Why share layers? The lower layers learn useful features for BOTH.

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()

        # shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )

        # actor head: outputs logits for each action
        self.actor  = nn.Linear(hidden, act_dim)

        # critic head: outputs a single scalar V(s)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h     = self.shared(x)
        logits= self.actor(h)
        value = self.critic(h).squeeze(-1)
        dist  = Categorical(logits=logits)   # discrete action distribution
        return dist, value

    def act(self, obs):
        """Sample an action and return (action, log_prob, value)."""
        dist, value = self(obs)
        action  = dist.sample()
        log_prob= dist.log_prob(action)
        return action, log_prob, value


# ─── COMPONENT 2: Rollout Buffer ─────────────────────────────────────────────
#
# We collect N steps of experience BEFORE updating.
# Why not update after every step? High variance — one bad step
# could destroy the policy. Batching smooths this out.

def collect_rollout(env, net, n_steps=512):
    """
    Run the policy for n_steps and store everything we need for the update.
    Returns arrays of: observations, actions, log_probs, rewards, values, dones
    """
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, done_buf = [], [], []

    obs  = env.reset()
    done = False

    for _ in range(n_steps):
        o = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, logp, value = net.act(o)

        next_obs, reward, done = env.step(action.item())

        obs_buf.append(obs);          act_buf.append(action.item())
        logp_buf.append(logp.item()); rew_buf.append(reward)
        val_buf.append(value.item()); done_buf.append(float(done))

        obs = env.reset() if done else next_obs

    return (np.array(obs_buf,  dtype=np.float32),
            np.array(act_buf,  dtype=np.int64),
            np.array(logp_buf, dtype=np.float32),
            np.array(rew_buf,  dtype=np.float32),
            np.array(val_buf,  dtype=np.float32),
            np.array(done_buf, dtype=np.float32))


# ─── COMPONENT 3: GAE — Generalised Advantage Estimation ─────────────────────
#
# Advantage A(s,a) = "how much better was this action than average?"
# A > 0 → action was better than expected → increase its probability
# A < 0 → action was worse than expected  → decrease its probability
#
# GAE balances bias vs variance using λ:
#   λ=0 → low variance, high bias  (like TD learning)
#   λ=1 → high variance, low bias  (like Monte Carlo)
#   λ=0.95 → sweet spot used in practice

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        # if episode ended, next value = 0
        next_value = 0.0 if t == len(rewards)-1 else values[t+1]
        next_value *= (1 - dones[t])

        # TD error: how wrong was our value estimate?
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE: exponentially weighted sum of TD errors
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values   # target for value function
    return advantages, returns


# ─── COMPONENT 4: PPO Update ─────────────────────────────────────────────────
#
# The core PPO idea: limit how much we change the policy.
#
# Naive policy gradient: maximize E[log π(a|s) * A(s,a)]
# Problem: a large gradient step can destroy the policy.
#
# PPO fix: clip the probability ratio r = π_new/π_old
#   L_clip = E[min(r*A, clip(r, 1-ε, 1+ε)*A)]
#
# This means: if the new policy is very different from the old one,
# stop updating in that direction.

def ppo_update(net, optimizer, obs, acts, logps_old, advantages, returns,
               clip_eps=0.2, epochs=4, batch_size=64):
    """
    Run multiple epochs of PPO updates on the collected data.
    Multiple epochs = better sample efficiency (reuse the same data).
    """
    # normalise advantages: zero mean, unit variance
    # this stabilises training — without this, large rewards dominate
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_t  = torch.tensor(obs)
    acts_t = torch.tensor(acts)
    logp_t = torch.tensor(logps_old)
    adv_t  = torch.tensor(advantages, dtype=torch.float32)
    ret_t  = torch.tensor(returns,    dtype=torch.float32)

    policy_losses, value_losses, entropies = [], [], []

    for epoch in range(epochs):
        # shuffle data each epoch
        idx = np.random.permutation(len(obs))

        for start in range(0, len(obs), batch_size):
            b = idx[start:start+batch_size]

            # re-evaluate actions under the NEW policy
            dist, values = net(obs_t[b])
            new_logps    = dist.log_prob(acts_t[b])

            # probability ratio: π_new(a|s) / π_old(a|s)
            # using log: exp(log π_new - log π_old)
            ratio = (new_logps - logp_t[b]).exp()

            # PPO clipped objective
            surr1 = ratio * adv_t[b]
            surr2 = ratio.clamp(1-clip_eps, 1+clip_eps) * adv_t[b]
            policy_loss = -torch.min(surr1, surr2).mean()

            # value function loss: MSE between predicted and actual returns
            value_loss = (values - ret_t[b]).pow(2).mean()

            # entropy bonus: encourages exploration
            # without this, policy collapses to always picking one action
            entropy = dist.entropy().mean()

            # total loss
            loss = policy_loss + 0.5*value_loss - 0.05*entropy

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping: prevent exploding gradients
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)


# ─── Training Loop ───────────────────────────────────────────────────────────

def train(total_steps=300_000, rollout_steps=512):
    env = CartPole()
    net = ActorCritic()
    opt = optim.Adam(net.parameters(), lr=3e-4)

    all_returns = []
    steps_done  = 0
    ep_rewards  = []
    current_ep_reward = 0.0

    print("Training CartPole with PPO from Scratch")
    print("=" * 55)
    print(f"{'Step':>10}  {'MeanReturn':>12}  {'MaxReturn':>10}  "
          f"{'PLoss':>8}  {'VLoss':>8}")

    obs  = env.reset()
    done = False

    while steps_done < total_steps:
        # Step 1: collect rollout
        O, A, L, R, V, D = collect_rollout(env, net, rollout_steps)

        # Step 2: compute advantages
        adv, ret = compute_gae(R, V, D)

        # Step 3: PPO update
        pl, vl, ent = ppo_update(net, opt, O, A, L, adv, ret)

        steps_done += rollout_steps

        # track episode returns from this rollout
        ep_ret = 0.0
        for r, d in zip(R, D):
            ep_ret += r
            if d:
                ep_rewards.append(ep_ret)
                ep_ret = 0.0

        if ep_rewards:
            mean_ret = np.mean(ep_rewards[-20:])
            max_ret  = np.max(ep_rewards[-20:])
            all_returns.extend(ep_rewards[-5:])
            print(f"{steps_done:>10}  {mean_ret:>12.1f}  {max_ret:>10.1f}  "
                  f"{pl:>8.4f}  {vl:>8.4f}")

            # solved when consistently scoring 495+
            if mean_ret >= 495 and len(ep_rewards) >= 20:
                print(f"\nSolved at step {steps_done}! Mean return = {mean_ret:.1f}")
                break

    torch.save(net.state_dict(), "cartpole_ppo.pth")
    print("Model saved to cartpole_ppo.pth")
    return net, all_returns


# ─── Watch Trained Agent ──────────────────────────────────────────────────────

def watch(net, episodes=3):
    env = CartPole()
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4))

    for ep in range(episodes):
        obs  = env.reset()
        done = False
        total_r = 0.0; step = 0

        while not done:
            o = torch.tensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist, _ = net(o)
                action  = dist.probs.argmax().item()   # deterministic
            obs, r, done = env.step(action)
            total_r += r; step += 1

            x, cy, px2, py2 = env.render_state()
            ax.clear()
            ax.set_xlim(-3, 3); ax.set_ylim(-0.3, 1.5)
            ax.set_title(f"PPO Agent — Episode {ep+1}  |  "
                         f"Step {step}  |  Return {total_r:.0f}")
            ax.axhline(0, color="brown", lw=4)
            ax.fill_between([-3,3],[-0.3,-0.3],[0,0],color="peru",alpha=0.3)
            ax.add_patch(plt.Rectangle((x-0.2,-0.1),0.4,0.2,color="steelblue"))
            ax.plot([x,px2],[cy,py2],"r-",lw=6,solid_capstyle="round")
            ax.plot(x, cy, "ko", ms=8)
            ax.set_xlabel("Cart position")
            plt.pause(0.02)

        print(f"Episode {ep+1}: {total_r:.0f} steps")

    plt.ioff(); plt.show()


# ─── Plot Training Curve ──────────────────────────────────────────────────────

def plot_returns(returns):
    w  = max(1, len(returns)//20)
    sm = np.convolve(returns, np.ones(w)/w, mode="valid")
    plt.figure(figsize=(10,4))
    plt.plot(returns, alpha=0.3, color="steelblue", label="Episode return")
    plt.plot(sm, color="orange", lw=2, label=f"Smoothed (w={w})")
    plt.axhline(500, color="green", ls="--", label="Solved (500)")
    plt.axhline(195, color="red",   ls="--", label="OpenAI threshold (195)")
    plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title("PPO from Scratch — CartPole Training")
    plt.legend(); plt.tight_layout()
    plt.savefig("02_ppo_returns.png", dpi=150)
    plt.show()
    print("Saved 02_ppo_returns.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    env = CartPole()
    net = ActorCritic()

    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        net.load_state_dict(torch.load("cartpole_ppo.pth", weights_only=True))
        net.eval()
        watch(net, episodes=5)
    else:
        net, returns = train(total_steps=100_000)
        plot_returns(returns)
        print("\nTo watch the trained agent:")
        print("  python 02_cartpole_ppo.py watch")
