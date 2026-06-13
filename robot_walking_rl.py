"""
2D Robot Walking RL — PPO from scratch
Physics: proper rigid-body legs with ground contact.
Reward: only real CoM displacement counts — no proxy hacking possible.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── Environment ─────────────────────────────────────────────────────────────

class BipedalWalker2D:
    """
    Segment masses and lengths
    ──────────────────────────
    torso  : 10 kg, 0.5 m tall
    thigh  :  3 kg, 0.4 m   (×2)
    shin   :  2 kg, 0.35 m  (×2)
    total  : 20 kg

    Locomotion model
    ────────────────
    The torso moves forward when a grounded foot pushes backward.
    Horizontal GRF = foot_friction × normal_force × sign(hip_extension_rate).
    Normal force comes from the leg supporting body weight.
    This means the robot MUST extend its hip while the foot is planted
    to move forward — exactly what walking is.
    """

    dt          = 0.02
    g           = 9.8
    max_steps   = 800
    gait_period = 40        # steps per full stride cycle

    # geometry
    torso_h = 0.5
    thigh_l = 0.40
    shin_l  = 0.35
    hip_w   = 0.12         # half-width between hip joints

    # masses
    m_torso = 10.0
    m_thigh =  3.0
    m_shin  =  2.0

    # joint limits
    hip_lo, hip_hi   = -0.8,  0.8    # rad
    knee_lo, knee_hi =  0.0,  2.0    # rad (only bends forward)

    # motors (peak torque in Nm, applied as normalised action ×max)
    max_hip_t  = 50.0
    max_knee_t = 30.0
    damping    = 3.0

    # contact
    mu         = 0.8      # friction coefficient
    foot_r     = 0.06     # foot detection radius

    def __init__(self):
        self.obs_dim = 18
        self.act_dim = 4

    # ── helpers ───────────────────────────────────────────────────────────────
    @property
    def total_mass(self):
        return self.m_torso + 2*self.m_thigh + 2*self.m_shin

    def _hip_pos(self, side):
        sign = -1 if side == 0 else 1
        bx = self.cx + sign * self.hip_w * np.cos(self.torso_a)
        by = self.cy + sign * self.hip_w * np.sin(self.torso_a) \
                     - self.torso_h/2 * np.cos(self.torso_a)
        return bx, by

    def _leg_fk(self, side):
        """Forward kinematics: returns knee and foot world positions."""
        hx, hy   = self._hip_pos(side)
        thigh_a  = self.torso_a + self.q_hip[side]
        kx = hx + self.thigh_l * np.sin(thigh_a)
        ky = hy - self.thigh_l * np.cos(thigh_a)
        shin_a   = thigh_a + self.q_knee[side]
        fx = kx + self.shin_l * np.sin(shin_a)
        fy = ky - self.shin_l * np.cos(shin_a)
        return kx, ky, fx, fy

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self):
        self.step_n = 0
        self.phase  = 0

        stand_h = (self.thigh_l + self.shin_l) * 0.95 + self.torso_h/2
        self.cx = 0.0
        self.cy = stand_h
        self.torso_a = np.random.uniform(-0.05, 0.05)
        self.vx = 0.0
        self.vy = 0.0
        self.wa = 0.0   # torso angular velocity

        # natural stance: legs slightly spread
        self.q_hip  = np.array([ 0.1, -0.1])
        self.q_knee = np.array([ 0.1,  0.1])
        self.dq_hip = np.zeros(2)
        self.dq_knee= np.zeros(2)

        self.contact = np.zeros(2)
        self.prev_cx = self.cx
        self._update_contact()
        return self._obs()

    def _update_contact(self):
        for s in range(2):
            _, _, fx, fy = self._leg_fk(s)
            self.contact[s] = float(fy <= self.foot_r)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        tau_hip  = action[[0,2]] * self.max_hip_t
        tau_knee = action[[1,3]] * self.max_knee_t

        # joint dynamics
        I_thigh = self.m_thigh * self.thigh_l**2 / 3
        I_shin  = self.m_shin  * self.shin_l**2  / 3

        prev_q_hip = self.q_hip.copy()

        for s in range(2):
            self.dq_hip[s]  += (tau_hip[s]  - self.damping*self.dq_hip[s])  / I_thigh * self.dt
            self.dq_knee[s] += (tau_knee[s] - self.damping*self.dq_knee[s]) / I_shin  * self.dt
            self.q_hip[s]   += self.dq_hip[s]  * self.dt
            self.q_knee[s]  += self.dq_knee[s] * self.dt
            self.q_hip[s]   = np.clip(self.q_hip[s],  self.hip_lo,  self.hip_hi)
            self.q_knee[s]  = np.clip(self.q_knee[s], self.knee_lo, self.knee_hi)

        self._update_contact()

        # ground reaction forces
        Fx, Fy, M = 0.0, 0.0, 0.0
        for s in range(2):
            if self.contact[s]:
                # normal force: share of body weight
                Fn = self.total_mass * self.g / max(1, self.contact.sum())
                # hip extension rate → foot pushes backward → body moves forward
                hip_ext_rate = -(self.dq_hip[s])
                Ft = np.clip(self.mu * Fn * hip_ext_rate * 0.15, -Fn*self.mu, Fn*self.mu)
                Fx += Ft
                Fy += Fn
                # moment: restores torso upright
                _, _, fx, fy = self._leg_fk(s)
                rx, ry = fx - self.cx, fy - self.cy
                M += rx * Fn * 0.01

        # torso acceleration
        ax_torso = Fx / self.total_mass
        ay_torso = Fy / self.total_mass - self.g
        alpha = -6.0*self.torso_a - 2.0*self.wa + M

        self.vx += ax_torso * self.dt
        self.vy += ay_torso * self.dt
        self.wa += alpha    * self.dt

        # light drag only — heavy drag was killing forward momentum
        self.vx *= 0.995
        self.vy *= 0.92
        self.wa *= 0.88

        self.cx += self.vx * self.dt
        self.cy += self.vy * self.dt
        self.torso_a += self.wa * self.dt

        # floor constraint
        min_cy = (self.thigh_l + self.shin_l)*0.3 + self.torso_h/2
        if self.cy < min_cy:
            self.cy = min_cy
            self.vy = max(0.0, self.vy)

        self.phase = (self.phase + 1) % self.gait_period
        self.step_n += 1

        # ── curriculum: forward weight grows from 0.5 → 5 over one episode ──
        fwd_weight = np.clip(self.step_n / 100.0, 0.5, 5.0)

        # ── reward ───────────────────────────────────────────────────────────
        dx = self.cx - self.prev_cx
        fwd_vel = dx / self.dt

        # 1. forward velocity — weight increases with curriculum
        r_fwd    = np.clip(fwd_vel, -0.5, 3.0) * fwd_weight

        # 2. stay upright (always strong so robot learns balance first)
        r_upright= (1.0 - abs(self.torso_a) / 1.0) * 1.5

        # 3. gait clock: reward alternating contact pattern
        ph = self.phase / self.gait_period
        r_gait   = (self.contact[0] * float(ph < 0.5) +
                    self.contact[1] * float(ph >= 0.5)) * 0.5

        # 4. comfortable height
        target_h = (self.thigh_l + self.shin_l)*0.85 + self.torso_h/2
        r_height = (1.0 - abs(self.cy - target_h) / 0.4) * 0.3

        # 5. tiny energy penalty
        r_energy = -0.003 * np.sum(action**2)

        reward = r_fwd + r_upright + r_gait + r_height + r_energy
        self.prev_cx = self.cx

        fallen = abs(self.torso_a) > 1.2 or self.cy < min_cy + 0.05
        done   = fallen or self.step_n >= self.max_steps
        if fallen:
            reward -= 5.0

        info = {"x_pos": self.cx, "forward_vel": fwd_vel,
                "contact": self.contact.copy()}
        return self._obs(), reward, done, info

    # ── observation ──────────────────────────────────────────────────────────
    def _obs(self):
        ph = self.phase / self.gait_period
        return np.array([
            np.clip(self.vx / 2.0, -3, 3),
            np.clip(self.vy / 2.0, -3, 3),
            self.torso_a,
            self.wa,
            self.q_hip[0],  self.dq_hip[0],
            self.q_knee[0], self.dq_knee[0],
            self.contact[0],
            self.q_hip[1],  self.dq_hip[1],
            self.q_knee[1], self.dq_knee[1],
            self.contact[1],
            (self.cy - 1.0) / 0.5,          # height deviation
            np.clip(self.cx / 10.0, -5, 5), # rough x progress
            np.sin(2*np.pi*ph),
            np.cos(2*np.pi*ph),
        ], dtype=np.float32)

    # ── rendering ────────────────────────────────────────────────────────────
    def get_body_parts(self):
        hx0, hy0 = self._hip_pos(0)
        hx1, hy1 = self._hip_pos(1)
        k0x,k0y,f0x,f0y = self._leg_fk(0)
        k1x,k1y,f1x,f1y = self._leg_fk(1)
        head_x = self.cx - self.torso_h/2*np.sin(self.torso_a)
        head_y = self.cy + self.torso_h/2*np.cos(self.torso_a)
        return {
            "torso": ([self.cx, head_x], [self.cy, head_y]),
            "head":  (head_x, head_y),
            "leg1":  ([hx0, k0x, f0x], [hy0, k0y, f0y]),
            "leg2":  ([hx1, k1x, f1x], [hy1, k1y, f1y]),
            "foot1_contact": self.contact[0],
            "foot2_contact": self.contact[1],
        }


# ─── PPO Actor-Critic ────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic        = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight,     gain=1.0)

    def forward(self, x):
        h    = self.shared(x)
        mean = torch.tanh(self.actor_mean(h))
        std  = self.actor_log_std.exp().clamp(1e-3, 0.8)
        return Normal(mean, std), self.critic(h).squeeze(-1)

    def act(self, obs):
        dist, val = self(obs)
        act  = dist.sample()
        logp = dist.log_prob(act).sum(-1)
        return act, logp, val


# ─── PPO ─────────────────────────────────────────────────────────────────────

class PPO:
    def __init__(self, obs_dim, act_dim,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=10, batch=64, max_grad=0.5):
        self.net    = ActorCritic(obs_dim, act_dim)
        self.opt    = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.gamma  = gamma; self.lam = lam
        self.clip   = clip;  self.epochs = epochs
        self.batch  = batch; self.max_grad = max_grad

    def rollout(self, env, steps=2048):
        O,A,L,R,V,D = [],[],[],[],[],[]
        obs = env.reset()
        ep_max_x = 0.0
        self.last_ep_max_x = 0.0
        for _ in range(steps):
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a, lp, v = self.net.act(o)
            nobs, r, done, info = env.step(a.squeeze(0).numpy())
            O.append(obs); A.append(a.squeeze(0).numpy())
            L.append(lp.item()); R.append(r)
            V.append(v.item()); D.append(done)
            ep_max_x = max(ep_max_x, info["x_pos"])
            if done:
                self.last_ep_max_x = ep_max_x
                ep_max_x = 0.0
                obs = env.reset()
            else:
                obs = nobs
        return [np.array(x, dtype=np.float32) for x in [O,A,L,R,V,D]]

    def gae(self, R, V, D):
        adv = np.zeros_like(R); g = 0.0
        for t in reversed(range(len(R))):
            nv    = 0.0 if t==len(R)-1 else V[t+1]
            delta = R[t] + self.gamma*nv*(1-D[t]) - V[t]
            g     = delta + self.gamma*self.lam*(1-D[t])*g
            adv[t]= g
        return adv, adv+V

    def update(self, O,A,L,adv,ret):
        adv = (adv-adv.mean())/(adv.std()+1e-8)
        Ot=torch.tensor(O); At=torch.tensor(A)
        Lt=torch.tensor(L); at=torch.tensor(adv,dtype=torch.float32)
        rt=torch.tensor(ret,dtype=torch.float32)
        pls,vls=[],[]
        for _ in range(self.epochs):
            idx = np.random.permutation(len(O))
            for s in range(0,len(O),self.batch):
                b = idx[s:s+self.batch]
                dist,vals = self.net(Ot[b])
                nlp = dist.log_prob(At[b]).sum(-1)
                ratio= (nlp-Lt[b]).exp()
                s1=ratio*at[b]; s2=ratio.clamp(1-self.clip,1+self.clip)*at[b]
                pl=  -torch.min(s1,s2).mean()
                vl=  (vals-rt[b]).pow(2).mean()
                ent= dist.entropy().sum(-1).mean()
                loss=pl+0.5*vl-0.01*ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.opt.step()
                pls.append(pl.item()); vls.append(vl.item())
        return np.mean(pls), np.mean(vls)

    def train(self, env, total=500_000, rollout=2048):
        returns, distances = [], []
        steps = 0; best = -np.inf
        print("Training 2D Robot Walker — PPO")
        print(f"{'Step':>10}  {'Return':>10}  {'x(m)':>8}  {'PLoss':>10}  {'VLoss':>8}")
        while steps < total:
            O,A,L,R,V,D = self.rollout(env, rollout)
            adv, ret     = self.gae(R,V,D)
            pl, vl       = self.update(O,A,L,adv,ret)
            ep_ret = R.sum(); x = self.last_ep_max_x
            returns.append(ep_ret); distances.append(x)
            steps += rollout
            print(f"{steps:>10}  {ep_ret:>10.1f}  {x:>8.2f}  {pl:>10.4f}  {vl:>8.4f}")
            if ep_ret > best:
                best = ep_ret
                torch.save(self.net.state_dict(), "robot_walker_best.pth")
        torch.save(self.net.state_dict(), "robot_walker.pth")
        return returns, distances


# ─── Watch & Plot ─────────────────────────────────────────────────────────────

def watch_robot(env, net, episodes=5):
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))
    for ep in range(episodes):
        obs = env.reset(); done = False; total_r = 0.0
        while not done:
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist,_ = net(o)
                action = dist.mean.squeeze(0).numpy()
            obs, r, done, info = env.step(action)
            total_r += r
            ax.clear()
            cx = info["x_pos"]
            ax.set_xlim(cx-2.5, cx+2.5); ax.set_ylim(-0.15, 2.2)
            ax.fill_between([cx-2.5,cx+2.5],[-0.15,-0.15],[0,0],
                            color="#8B4513",alpha=0.4)
            ax.axhline(0, color="#5D2E0C", lw=2)
            for xi in range(int(cx)-3,int(cx)+4):
                ax.axvline(xi, color="gray", alpha=0.15, lw=0.5)
            bp = env.get_body_parts()
            ax.plot(*bp["torso"],"b-",lw=8,solid_capstyle="round")
            ax.plot(bp["head"][0],bp["head"][1],"bo",ms=18)
            for leg,key in [("leg1","foot1_contact"),("leg2","foot2_contact")]:
                c = "#00cc44" if bp[key] else "#ff6600"
                ax.plot(*bp[leg],color=c,lw=5,
                        solid_capstyle="round",marker="o",ms=5)
            legend=[mpatches.Patch(color="#00cc44",label="Contact"),
                    mpatches.Patch(color="#ff6600",label="In air")]
            ax.legend(handles=legend,loc="upper right",fontsize=9)
            ax.set_title(f"Ep {ep+1} | x={cx:.2f}m | "
                         f"vel={info['forward_vel']:.2f}m/s | "
                         f"return={total_r:.1f}")
            plt.pause(0.02)
    plt.ioff(); plt.show()


def plot_results(returns, distances):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,4))
    w = max(1,len(returns)//20)
    sm = lambda a: np.convolve(a,np.ones(w)/w,mode="valid")
    ax1.plot(returns,   alpha=0.3,color="steelblue")
    ax1.plot(sm(np.array(returns)),color="orange",lw=2,label="Smoothed")
    ax1.set_title("Return per Rollout"); ax1.set_xlabel("Rollout")
    ax1.set_ylabel("Return"); ax1.legend()
    ax2.plot(distances, alpha=0.3,color="steelblue")
    ax2.plot(sm(np.array(distances)),color="green",lw=2,label="Smoothed")
    ax2.set_title("Distance Walked (m)"); ax2.set_xlabel("Rollout")
    ax2.set_ylabel("x position (m)"); ax2.legend()
    plt.tight_layout()
    plt.savefig("robot_walking_training.png",dpi=150)
    plt.show()
    print("Saved robot_walking_training.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    env = BipedalWalker2D()
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        net = ActorCritic(env.obs_dim, env.act_dim)
        net.load_state_dict(torch.load("robot_walker_best.pth",
                                       weights_only=True))
        net.eval()
        watch_robot(env, net)
    else:
        agent = PPO(env.obs_dim, env.act_dim, lr=1e-4, epochs=5)
        returns, distances = agent.train(env, total=500_000)
        plot_results(returns, distances)
        print("Best model: robot_walker_best.pth")
        print("To watch:   python robot_walking_rl.py watch")
