"""
Simulation of the paper: "Optimal BIM staffing in construction projects using a queueing model"
(Ham, Moon, Kim, Kim - Automation in Construction 113, 2020)

Contents:
  1. Analytical M/M/s model  (Eqs. 1-9 of the paper)
  2. Economic analysis       (Eqs. 10-13): TC = Cs*s + Cw*L
  3. Discrete-event simulation (Monte Carlo) to validate the analytical model
  4. Reproduction of Tables 7-11 and plots of Figs. 3-8

Units: time in DAYS, costs in KRW per HOUR (as in the paper, 8-h workday).
Dependencies: numpy, matplotlib
"""
import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOURS_PER_DAY = 8

# ── Directory structure ──────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
OUT  = HERE / "Results"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. ANALYTICAL M/M/s MODEL
# ---------------------------------------------------------------------------
@dataclass
class MMs:
    lam: float  # λ: mean arrival rate (RFIs/day)
    mu: float   # μ: mean service rate per server (RFIs/day)
    s: int      # number of servers (BIM staff)

    @property
    def rho(self):
        return self.lam / (self.s * self.mu)

    @property
    def stable(self):
        return self.rho < 1

    @property
    def a(self):  # offered load λ/μ
        return self.lam / self.mu

    def P0(self):  # Eq. (1)
        if not self.stable:
            return float("nan")
        total = sum(self.a**n / math.factorial(n) for n in range(self.s))
        tail = self.a**self.s / (math.factorial(self.s) * (1 - self.rho))
        return 1.0 / (total + tail)

    def Lq(self):  # Eq. (2)
        if not self.stable:
            return float("nan")
        return (self.P0() * self.a**self.s * self.rho
                / (math.factorial(self.s) * (1 - self.rho) ** 2))

    def Wq(self):  # Eq. (3)
        return self.Lq() / self.lam

    def W(self):  # Eq. (4)
        return self.Wq() + 1 / self.mu

    def L(self):  # Eq. (5) Little: L = λW
        return self.lam * self.W()

    def Pn(self, n):  # Eqs. (6)-(7)
        if n <= self.s:
            c = self.a**n / math.factorial(n)
        else:
            c = self.a**n / (math.factorial(self.s) * self.s ** (n - self.s))
        return c * self.P0()

    def prob_W_greater_t(self, t):  # Eq. (8): P(W > t)
        return self._pw_formula(t)

    def _pw_formula(self, t):
        """Eq. (8): P(W>t) = e^{-μt} [1 + P0 (λ/μ)^s / (s!(1-ρ)) * (1 - e^{-μt(s-1-λ/μ)}) / (s-1-λ/μ)] """
        s, mu, rho, a = self.s, self.mu, self.rho, self.a
        den = s - 1 - a
        if abs(den) < 1e-12:
            frac = mu * t
        else:
            frac = (1 - math.exp(-mu * t * den)) / den
        return math.exp(-mu * t) * (
            1 + self.P0() * a**s / (math.factorial(s) * (1 - rho)) * frac
        )

    def prob_Wq_greater_t(self, t):  # Eq. (9): P(Wq > t)
        s, mu, rho = self.s, self.mu, self.rho
        # prob. of waiting = 1 - sum_{n<s} Pn ; then decays exponentially
        p_wait = 1 - sum(self.Pn(n) for n in range(s))
        return p_wait * math.exp(-s * mu * (1 - rho) * t)


# ---------------------------------------------------------------------------
# 2. ECONOMIC ANALYSIS  (Eqs. 10-13)
# ---------------------------------------------------------------------------
def cost_table(lam, mu, Cs, Cw, s_range):
    """TC = Cs*s + Cw*L  (per hour). Returns list of rows and the optimal s."""
    rows, best = [], None
    for s in s_range:
        m = MMs(lam, mu, s)
        if m.stable:
            L = m.L()
            SC, WC = Cs * s, Cw * L
            TC = SC + WC
            if best is None or TC < best[1]:
                best = (s, TC)
        else:
            L = SC = WC = TC = None
            SC = Cs * s
        rows.append(dict(s=s, rho=m.rho, L=L, SC=SC, WC=WC, TC=TC))
    return rows, best[0]


# ---------------------------------------------------------------------------
# 3. DISCRETE-EVENT SIMULATION  (M/M/s, FCFS, infinite queue)
# ---------------------------------------------------------------------------
def simulate_mms(lam, mu, s, n_customers=200_000, warmup=5_000, seed=0):
    """Simulates n_customers RFIs. Returns empirical W, Wq, L, Lq, P(W>1), P(Wq>1), Pn."""
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1 / lam, n_customers))
    services = rng.exponential(1 / mu, n_customers)

    free_at = [0.0] * s          # time at which each server becomes free
    heapq.heapify(free_at)
    start = np.empty(n_customers)
    end = np.empty(n_customers)
    for i in range(n_customers):  # FCFS: the next customer takes the first free server
        t_free = heapq.heappop(free_at)
        start[i] = max(arrivals[i], t_free)
        end[i] = start[i] + services[i]
        heapq.heappush(free_at, end[i])

    sl = slice(warmup, None)
    Wq = start[sl] - arrivals[sl]
    W = end[sl] - arrivals[sl]

    # empirical distribution of the number of customers in the system (time-weighted)
    t0, t1 = arrivals[warmup], arrivals[-1]
    events = np.concatenate([np.c_[arrivals[sl], np.ones(len(W))],
                             np.c_[end[sl], -np.ones(len(W))]])
    events = events[np.argsort(events[:, 0], kind="stable")]
    events = events[events[:, 0] <= t1]
    n_curr, t_prev = 0, t0
    time_in_n = {}
    for t, d in events:
        time_in_n[n_curr] = time_in_n.get(n_curr, 0.0) + (t - t_prev)
        n_curr += int(d)
        t_prev = t
    total = sum(time_in_n.values())
    Pn = {n: v / total for n, v in sorted(time_in_n.items())}

    return dict(
        W=W.mean(), Wq=Wq.mean(),
        L=lam * W.mean(), Lq=lam * Wq.mean(),
        PW1=(W > 1).mean(), PWq1=(Wq > 1).mean(), Pn=Pn,
    )


# ---------------------------------------------------------------------------
# 4. PAPER CASES
# ---------------------------------------------------------------------------
CASES = {
    "Project 1 (Busan H)": dict(lam=22.74, mu=3.00, s_actual=6, s_analysis=11,
                                Cs=28_409, range=range(6, 15), s_opt_paper=10),
    "Project 2 (Incheon G)": dict(lam=1.81, mu=3.00, s_actual=3, s_analysis=3,
                                  Cs=28_409, range=range(1, 6), s_opt_paper=2),
    "Project 3 (Daejeon H)": dict(lam=4.40, mu=1.67, s_actual=3, s_analysis=3,
                                  Cs=18_939, range=range(1, 8), s_opt_paper=4),
}
CW = 33_792  # KRW/hour, waiting cost of one participant


def fmt(x, nd=2):
    return "N/A" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{nd}f}"


def main():
    print("=" * 78)
    print("TABLE 7 / 8 – Performance measures (analytical vs. simulation)")
    print("=" * 78)
    for name, c in CASES.items():
        m = MMs(c["lam"], c["mu"], c["s_analysis"])
        sim = simulate_mms(c["lam"], c["mu"], c["s_analysis"], seed=42)
        print(f"\n{name}: λ={c['lam']}, μ={c['mu']}, s={c['s_analysis']}  "
              f"(actual s = {c['s_actual']}, actual ρ = "
              f"{c['lam']/(c['s_actual']*c['mu']):.2f})")
        print(f"  ρ = {m.rho:.2f}")
        print(f"  {'':10s}{'Analytical':>12s}{'Simulation':>12s}")
        for k, an, si in [("L", m.L(), sim["L"]), ("Lq", m.Lq(), sim["Lq"]),
                          ("W (days)", m.W(), sim["W"]), ("Wq (days)", m.Wq(), sim["Wq"]),
                          ("P(W>1)", m.prob_W_greater_t(1), sim["PW1"]),
                          ("P(Wq>1)", m.prob_Wq_greater_t(1), sim["PWq1"])]:
            print(f"  {k:10s}{fmt(an, 5):>12s}{fmt(si, 5):>12s}")
        print(f"  Mean time in system ≈ {m.W()*HOURS_PER_DAY:.1f} h (8-h workday)")

    print("\n" + "=" * 78)
    print("TABLES 9-11 – Economic analysis: TC = Cs·s + Cw·L  (KRW/hour)")
    print("=" * 78)
    results = {}
    for name, c in CASES.items():
        rows, s_opt = cost_table(c["lam"], c["mu"], c["Cs"], CW, c["range"])
        results[name] = (rows, s_opt)
        print(f"\n{name}")
        print(f"  {'s':>3s}{'ρ':>8s}{'L':>9s}{'Service':>12s}{'Waiting':>12s}{'Total':>12s}")
        for r in rows:
            mark = "  <-- optimal" if r["s"] == s_opt else ""
            print(f"  {r['s']:>3d}{r['rho']:>8.2f}{fmt(r['L']):>9s}"
                  f"{fmt(r['SC'], 0):>12s}{fmt(r['WC'], 0):>12s}{fmt(r['TC'], 0):>12s}{mark}")
        ok = "✔ matches the paper" if s_opt == c["s_opt_paper"] else "✘ differs"
        print(f"  optimal s = {s_opt}  (paper: {c['s_opt_paper']})  {ok}")

    # --- Fig. 3-5: Pn distribution
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, c) in zip(axs, CASES.items()):
        m = MMs(c["lam"], c["mu"], c["s_analysis"])
        sim = simulate_mms(c["lam"], c["mu"], c["s_analysis"], seed=7)
        n_max = 30 if "3" in name else 22
        ns = np.arange(n_max)
        ax.bar(ns - 0.2, [m.Pn(n) for n in ns], 0.4, label="Analytical")
        ax.bar(ns + 0.2, [sim["Pn"].get(n, 0) for n in ns], 0.4, label="Simulation")
        ax.set_title(name)
        ax.set_xlabel("No. of RFIs in the system (n)")
        ax.set_ylabel("Pn")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig_Pn.png", dpi=130)
    plt.close()

    # --- Fig. 6-8: cost curves
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, (rows, s_opt)) in zip(axs, results.items()):
        v = [r for r in rows if r["TC"] is not None]
        ss = [r["s"] for r in v]
        ax.plot(ss, [r["SC"] for r in v], "o-", label="Service cost")
        ax.plot(ss, [r["WC"] for r in v], "s-", label="Waiting cost")
        ax.plot(ss, [r["TC"] for r in v], "^-", lw=2, label="Total cost")
        ax.axvline(s_opt, ls="--", c="gray")
        ax.set_title(f"{name}\noptimal s = {s_opt}")
        ax.set_xlabel("No. of BIM staff (s)")
        ax.set_ylabel("KRW / hour")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig_costs.png", dpi=130)
    plt.close()
    print(f"\nPlots saved to {OUT}: fig_Pn.png, fig_costs.png")


if __name__ == "__main__":
    main()
