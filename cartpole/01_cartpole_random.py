"""
CartPole — Step 1: Random Policy Baseline
==========================================
Before training anything, we establish a baseline by taking random actions.
This shows students WHY we need RL — a random policy fails completely.

Environment:
  - State:   [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
  - Actions: 0 = push left, 1 = push right
  - Reward:  +1 for every timestep the pole stays upright
  - Done:    pole falls beyond ±12°, cart moves beyond ±2.4 units, or 500 steps

Key question to ask students:
  "What average score do you expect from a random policy?"
  Answer: ~8-10 steps (out of 500 max). Pure chance can't balance a pole.
"""

import numpy as np
import matplotlib.pyplot as plt


# ─── Minimal CartPole Environment (no gym needed) ────────────────────────────

class CartPole:
    """
    CartPole physics from scratch.
    Based on Barto, Sutton & Anderson (1983).
    """

    gravity    = 9.8
    mass_cart  = 1.0
    mass_pole  = 0.1
    pole_half  = 0.5       # half the pole length
    force_mag  = 10.0
    dt         = 0.02
    max_steps  = 500

    angle_limit = 12 * np.pi / 180   # 12 degrees in radians
    pos_limit   = 2.4

    def reset(self):
        # small random starting state
        self.state = np.random.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        total_mass = self.mass_cart + self.mass_pole
        pole_mass_length = self.mass_pole * self.pole_half

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # physics equations
        temp = (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_half * (4/3 - self.mass_pole * cos_theta**2 / total_mass)
        )
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

        # euler integration
        x         += self.dt * x_dot
        x_dot     += self.dt * x_acc
        theta     += self.dt * theta_dot
        theta_dot += self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = (abs(x)     > self.pos_limit or
                abs(theta) > self.angle_limit or
                self.steps >= self.max_steps)

        reward = 1.0 if not done or self.steps >= self.max_steps else 0.0
        return self.state.copy(), reward, done

    def render_state(self):
        """Return positions for visualization."""
        x, _, theta, _ = self.state
        cart_y  = 0.0
        pole_x2 = x + 2 * self.pole_half * np.sin(theta)
        pole_y2 = cart_y + 2 * self.pole_half * np.cos(theta)
        return x, cart_y, pole_x2, pole_y2


# ─── Random Policy ───────────────────────────────────────────────────────────

def random_policy(state):
    """Takes a completely random action — ignores the state entirely."""
    return np.random.randint(2)


# ─── Run Episodes ─────────────────────────────────────────────────────────────

def run_episodes(n_episodes=200, render_every=50):
    env     = CartPole()
    returns = []

    print("Random Policy Baseline")
    print("=" * 40)
    print(f"Running {n_episodes} episodes with random actions...\n")

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = random_policy(obs)
            obs, reward, done = env.step(action)
            total_reward += reward

        returns.append(total_reward)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1:>4}  |  Return: {total_reward:>6.1f}  |  "
                  f"Running avg: {np.mean(returns[-50:]):.1f}")

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean return : {np.mean(returns):.1f}")
    print(f"  Max return  : {np.max(returns):.1f}")
    print(f"  Min return  : {np.min(returns):.1f}")
    print(f"\n  (Maximum possible return = 500)")
    print(f"  Random policy achieves {np.mean(returns)/500*100:.1f}% of maximum")
    return returns


# ─── Visualise one episode ────────────────────────────────────────────────────

def visualise_episode():
    """Show the pole falling in real time."""
    env = CartPole()
    obs = env.reset()
    done = False

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))

    step = 0
    while not done:
        action = random_policy(obs)
        obs, _, done = env.step(action)
        x, cy, px2, py2 = env.render_state()

        ax.clear()
        ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 1.5)
        ax.set_title(f"Random Policy — Step {step+1}  |  "
                     f"Angle: {np.degrees(obs[2]):.1f}°")
        ax.axhline(0, color="brown", linewidth=4)
        ax.fill_between([-3, 3], [-0.5, -0.5], [0, 0],
                        color="peru", alpha=0.3)

        # cart
        cart = plt.Rectangle((x-0.2, -0.1), 0.4, 0.2,
                              color="steelblue")
        ax.add_patch(cart)
        # pole
        ax.plot([x, px2], [cy, py2], "r-", linewidth=6,
                solid_capstyle="round")
        ax.plot(x, cy, "ko", markersize=8)

        ax.set_xlabel("Cart position")
        plt.pause(0.05)
        step += 1

    plt.ioff()
    ax.set_title(f"Pole fell after {step} steps!")
    plt.show()


# ─── Plot Returns ─────────────────────────────────────────────────────────────

def plot_returns(returns):
    plt.figure(figsize=(10, 4))
    plt.plot(returns, alpha=0.4, color="steelblue", label="Episode return")
    plt.axhline(np.mean(returns), color="red", linestyle="--",
                label=f"Mean = {np.mean(returns):.1f}")
    plt.axhline(500, color="green", linestyle="--", label="Max possible = 500")
    plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title("Random Policy — CartPole Returns")
    plt.legend(); plt.tight_layout()
    plt.savefig("01_random_returns.png", dpi=150)
    plt.show()
    print("Saved 01_random_returns.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(__doc__)
    returns = run_episodes(n_episodes=200)
    plot_returns(returns)
    print("\nNow watch one episode live:")
    visualise_episode()
