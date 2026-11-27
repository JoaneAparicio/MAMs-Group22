"""
Simulation of the paper: "Optimal BIM staffing in construction projects using a queueing model"
(Ham, Moon, Kim, Kim - Automation in Construction 113, 2020)

Contents:
  PART A - HOMOGENEOUS SERVERS (as in the paper)
    1. Analytical M/M/s model  (Eqs. 1-9 of the paper)
    2. Economic analysis       (Eqs. 10-13): TC = Cs*s + Cw*L
    3. Discrete-event simulation (Monte Carlo) to validate the analytical model
    4. Reproduction of Tables 7-11 and plots of Figs. 3-8

  PART B - HETEROGENEOUS SERVERS (extension)
    5. Exact analytical model for M/M/s with K staff types (different μ_k, Cs_k),
       solved as a continuous-time Markov chain (CTMC) with a geometric queue tail
    6. Discrete-event simulation with heterogeneous servers
    7. Economic analysis: TC = Σ_k Cs_k·s_k + Cw·L, optimised over the staff mix

Execution order: first Part A (homogeneous), then Part B (heterogeneous).

Units: time in DAYS, costs in KRW per HOUR (as in the paper, 8-h workday).
Dependencies: numpy, matplotlib
"""
import heapq
import itertools
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
OUT  = HERE / "Paper Results"
OUT.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# PART A - HOMOGENEOUS SERVERS
# ===========================================================================

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
def _time_avg_Pn(arrivals, end, warmup):
    """Time-weighted empirical distribution of the number of customers in the system."""
    sl = slice(warmup, None)
    n = len(arrivals) - warmup
    t0, t1 = arrivals[warmup], arrivals[-1]
    events = np.concatenate([np.c_[arrivals[sl], np.ones(n)],
                             np.c_[end[sl], -np.ones(n)]])
    events = events[np.argsort(events[:, 0], kind="stable")]
    events = events[events[:, 0] <= t1]
    n_curr, t_prev = 0, t0
    time_in_n = {}
    for t, d in events:
        time_in_n[n_curr] = time_in_n.get(n_curr, 0.0) + (t - t_prev)
        n_curr += int(d)
        t_prev = t
    total = sum(time_in_n.values())
    return {k: v / total for k, v in sorted(time_in_n.items())}


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

    return dict(
        W=W.mean(), Wq=Wq.mean(),
        L=lam * W.mean(), Lq=lam * Wq.mean(),
        PW1=(W > 1).mean(), PWq1=(Wq > 1).mean(),
        Pn=_time_avg_Pn(arrivals, end, warmup),
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


def run_homogeneous():
    print("=" * 78)
    print("PART A - HOMOGENEOUS SERVERS (paper)")
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
    return results


# ===========================================================================
# PART B - HETEROGENEOUS SERVERS (extension)
# ===========================================================================

# Staff types, defined RELATIVE to each project's (μ, Cs) from the paper.
# These values are HYPOTHETICAL (the paper has no heterogeneous data): change them freely.
#   speed: service-rate multiplier  (μ_k = speed · μ)
#   cost : hourly-cost multiplier   (Cs_k = cost · Cs)
# With speed 1.5 / 0.5, a 50/50 mix has the same total capacity as the homogeneous
# team, so the comparison isolates the effect of heterogeneity itself.
STAFF_TYPES = {
    "Senior": dict(speed=1.5, cost=1.3),
    "Junior": dict(speed=0.5, cost=0.7),
}
POLICIES = ("fastest", "random")  # how an arriving RFI picks among idle servers


def _hypoexp_sf(a, b, t):
    """P(X + Y > t) with X ~ Exp(a), Y ~ Exp(b) independent."""
    if abs(a - b) < 1e-12:
        return math.exp(-a * t) * (1 + a * t)
    return (b * math.exp(-a * t) - a * math.exp(-b * t)) / (b - a)


# ---------------------------------------------------------------------------
# 5. ANALYTICAL HETEROGENEOUS M/M/s  (exact CTMC)
# ---------------------------------------------------------------------------
class HeteroMMs:
    """
    M/M/s queue with K server types: counts[k] servers with rate mus[k] each.
    FCFS, infinite queue. Routing of an arriving RFI when several servers are idle:
      - "fastest": goes to an idle server of the fastest type
      - "random" : goes to an idle server chosen uniformly at random
    When all servers are busy, the next queued RFI goes to whichever server frees first.

    State: b = (b_1..b_K) busy servers per type. When all are busy (b = counts) the
    total departure rate is the constant M = Σ s_k μ_k, so the queue length is
    geometric: π(full, q) = π(full, 0)·r^q with r = λ/M. The chain is therefore
    solved on the finite set {b} (censored on q = 0) and the tail is added exactly.
    With K = 1 it reduces to the paper's M/M/s model.
    """

    def __init__(self, lam, mus, counts, policy="fastest"):
        self.lam = lam
        self.mus = tuple(float(m) for m in mus)
        self.counts = tuple(int(n) for n in counts)
        self.policy = policy
        self.s = sum(self.counts)
        self.M = sum(n * m for n, m in zip(self.counts, self.mus))
        self.rho = lam / self.M if self.M > 0 else float("inf")
        self.stable = self.rho < 1
        self._order = sorted(range(len(self.mus)), key=lambda k: -self.mus[k])
        if self.stable:
            self._solve()

    def route(self, b):
        """Probability that an arrival finding state b (not full) goes to each type."""
        idle = [n - x for n, x in zip(self.counts, b)]
        if self.policy == "fastest":
            k = next(k for k in self._order if idle[k] > 0)
            return {k: 1.0}
        tot = sum(idle)
        return {k: i / tot for k, i in enumerate(idle) if i > 0}

    def _solve(self):
        K, full = len(self.counts), self.counts
        states = list(itertools.product(*[range(n + 1) for n in self.counts]))
        idx = {st: i for i, st in enumerate(states)}
        Q = np.zeros((len(states), len(states)))
        for st in states:
            i = idx[st]
            if st != full:                       # arrival -> starts service at once
                for k, p in self.route(st).items():
                    nxt = list(st); nxt[k] += 1
                    Q[i, idx[tuple(nxt)]] += self.lam * p
            for k in range(K):                   # departure of a type-k server
                if st[k] > 0:                    # (from full: queue empty in censored chain)
                    nxt = list(st); nxt[k] -= 1
                    Q[i, idx[tuple(nxt)]] += st[k] * self.mus[k]
            Q[i, i] = -Q[i].sum()
        A = Q.T.copy()
        A[-1, :] = 1.0
        rhs = np.zeros(len(states)); rhs[-1] = 1.0
        pc = np.linalg.solve(A, rhs)             # censored stationary distribution
        r = self.rho
        i_full = idx[full]
        norm = pc.sum() - pc[i_full] + pc[i_full] / (1 - r)
        self.pi = {st: pc[idx[st]] / norm for st in states if st != full}
        self.pF0 = pc[i_full] / norm             # P(all busy, empty queue)
        self.r = r

    # --- performance measures --------------------------------------------
    def P_wait(self):  # P(an arrival has to queue) = P(all servers busy)
        return self.pF0 / (1 - self.r) if self.stable else float("nan")

    def Lq(self):
        return self.pF0 * self.r / (1 - self.r) ** 2 if self.stable else float("nan")

    def L(self):
        if not self.stable:
            return float("nan")
        busy = sum(sum(b) * p for b, p in self.pi.items())
        return busy + self.pF0 * (self.s / (1 - self.r) + self.r / (1 - self.r) ** 2)

    def Wq(self):
        return self.Lq() / self.lam

    def W(self):
        return self.L() / self.lam

    def Pn(self, n):
        if n < self.s:
            return sum(p for b, p in self.pi.items() if sum(b) == n)
        return self.pF0 * self.r ** (n - self.s)

    def prob_Wq_greater_t(self, t):
        return self.P_wait() * math.exp(-(self.M - self.lam) * t)

    def prob_W_greater_t(self, t):
        # arrivals finding an idle server: W = service time of the server they get
        p = sum(pb * q * math.exp(-self.mus[k] * t)
                for b, pb in self.pi.items() for k, q in self.route(b).items())
        # arrivals that queue: W = Exp(M-λ) waiting + service of the server that frees,
        # which is type k with probability s_k μ_k / M (independent of the wait)
        p += self.P_wait() * sum(
            n * m / self.M * _hypoexp_sf(self.M - self.lam, m, t)
            for n, m in zip(self.counts, self.mus) if n > 0)
        return p

    def utilization(self):
        """Fraction of time a server of each type is busy."""
        out = []
        for k, n in enumerate(self.counts):
            if n == 0:
                out.append(float("nan")); continue
            eb = sum(b[k] * p for b, p in self.pi.items()) + self.P_wait() * n
            out.append(eb / n)
        return out


# ---------------------------------------------------------------------------
# 6. DISCRETE-EVENT SIMULATION, HETEROGENEOUS SERVERS
# ---------------------------------------------------------------------------
def simulate_hetero(lam, mus, counts, policy="fastest",
                    n_customers=200_000, warmup=5_000, seed=0):
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1 / lam, n_customers))
    work = rng.exponential(1.0, n_customers)      # service = work / μ_server
    pick = rng.random(n_customers)                # for random routing
    srv_mu = [m for m, n in zip(mus, counts) for _ in range(n)]
    srv_type = [k for k, n in enumerate(counts) for _ in range(n)]
    s = len(srv_mu)
    order = sorted(range(s), key=lambda j: -srv_mu[j])  # fastest first

    free_at = [0.0] * s
    start = np.empty(n_customers)
    end = np.empty(n_customers)
    served_type = np.empty(n_customers, dtype=int)
    for i in range(n_customers):
        a = arrivals[i]
        idle = [j for j in order if free_at[j] <= a]
        if idle:   # FCFS => queue is empty, choose among idle servers
            j = idle[0] if policy == "fastest" else idle[int(pick[i] * len(idle))]
            st = a
        else:      # everybody busy: wait for the first server that frees up
            j = min(range(s), key=free_at.__getitem__)
            st = free_at[j]
        start[i] = st
        end[i] = st + work[i] / srv_mu[j]
        free_at[j] = end[i]
        served_type[i] = srv_type[j]

    sl = slice(warmup, None)
    Wq = start[sl] - arrivals[sl]
    W = end[sl] - arrivals[sl]
    span = arrivals[-1] - arrivals[warmup]
    busy = (end - start)[sl]
    util = [busy[served_type[sl] == k].sum() / (n * span) if n else float("nan")
            for k, n in enumerate(counts)]
    return dict(
        W=W.mean(), Wq=Wq.mean(),
        L=lam * W.mean(), Lq=lam * Wq.mean(),
        PW1=(W > 1).mean(), PWq1=(Wq > 1).mean(),
        Pn=_time_avg_Pn(arrivals, end, warmup), util=util,
    )


# ---------------------------------------------------------------------------
# 7. ECONOMIC ANALYSIS, HETEROGENEOUS STAFF MIX
# ---------------------------------------------------------------------------
def staff_params(c):
    """(names, μ_k, Cs_k) of the staff types for a paper case."""
    names = list(STAFF_TYPES)
    mus = [c["mu"] * STAFF_TYPES[t]["speed"] for t in names]
    costs = [c["Cs"] * STAFF_TYPES[t]["cost"] for t in names]
    return names, mus, costs


def hetero_cost_grid(lam, mus, costs, Cw, n_max, policy="fastest"):
    """TC(n_1, n_2) = Cs_1 n_1 + Cs_2 n_2 + Cw·L over a 2-type grid. NaN = unstable."""
    TC = np.full((n_max + 1, n_max + 1), np.nan)
    L = np.full_like(TC, np.nan)
    for n1, n2 in itertools.product(range(n_max + 1), repeat=2):
        if n1 + n2 == 0:
            continue
        m = HeteroMMs(lam, mus, (n1, n2), policy)
        if m.stable:
            L[n1, n2] = m.L()
            TC[n1, n2] = costs[0] * n1 + costs[1] * n2 + Cw * L[n1, n2]
    best = np.unravel_index(np.nanargmin(TC), TC.shape)
    return TC, L, (int(best[0]), int(best[1]))


def run_heterogeneous(homog_results):
    print("\n\n" + "=" * 78)
    print("PART B - HETEROGENEOUS SERVERS (extension)")
    print("=" * 78)
    print("Staff types (relative to each project's μ and Cs):")
    for t, p in STAFF_TYPES.items():
        print(f"  {t:7s} μ_k = {p['speed']:.2f}·μ   Cs_k = {p['cost']:.2f}·Cs")

    # --- sanity check: one single type must reproduce the homogeneous model
    print("\nValidation: HeteroMMs with a single type vs. homogeneous M/M/s")
    for name, c in CASES.items():
        h = HeteroMMs(c["lam"], [c["mu"]], [c["s_analysis"]])
        m = MMs(c["lam"], c["mu"], c["s_analysis"])
        err = max(abs(h.L() - m.L()), abs(h.W() - m.W()),
                  abs(h.prob_W_greater_t(1) - m.prob_W_greater_t(1)),
                  abs(h.prob_Wq_greater_t(1) - m.prob_Wq_greater_t(1)))
        print(f"  {name:24s} max |difference| = {err:.2e}")

    # --- performance measures for a mixed team with the same headcount as the paper
    print("\n" + "=" * 78)
    print("Performance measures – mixed team with the same headcount s as Tables 7/8")
    print("=" * 78)
    mixes = {}
    for name, c in CASES.items():
        names, mus, _ = staff_params(c)
        s = c["s_analysis"]
        mix = (math.ceil(s / 2), s // 2)
        mixes[name] = mix
        hom = MMs(c["lam"], c["mu"], s)
        het = {p: HeteroMMs(c["lam"], mus, mix, p) for p in POLICIES}
        sim = simulate_hetero(c["lam"], mus, mix, "fastest", seed=42)
        print(f"\n{name}: λ={c['lam']}, s={s} → {mix[0]} {names[0]} "
              f"(μ={mus[0]:.2f}) + {mix[1]} {names[1]} (μ={mus[1]:.2f})")
        print(f"  ρ homogeneous = {hom.rho:.3f}   ρ heterogeneous = {het['fastest'].rho:.3f}")
        print(f"  {'':10s}{'Homog.':>11s}{'Het-fast':>11s}{'Het-fast':>11s}{'Het-rand':>11s}")
        print(f"  {'':10s}{'analyt.':>11s}{'analyt.':>11s}{'simul.':>11s}{'analyt.':>11s}")
        hf, hr = het["fastest"], het["random"]
        for k, f in [("L", lambda x: x.L()), ("Lq", lambda x: x.Lq()),
                     ("W (days)", lambda x: x.W()), ("Wq (days)", lambda x: x.Wq()),
                     ("P(W>1)", lambda x: x.prob_W_greater_t(1)),
                     ("P(Wq>1)", lambda x: x.prob_Wq_greater_t(1))]:
            simkey = {"L": "L", "Lq": "Lq", "W (days)": "W", "Wq (days)": "Wq",
                      "P(W>1)": "PW1", "P(Wq>1)": "PWq1"}[k]
            print(f"  {k:10s}{fmt(f(hom), 5):>11s}{fmt(f(hf), 5):>11s}"
                  f"{fmt(sim[simkey], 5):>11s}{fmt(f(hr), 5):>11s}")
        for j, t in enumerate(names):
            print(f"  Utilization {t:7s}  analyt.(fast) = {fmt(hf.utilization()[j], 3)}"
                  f"   simul.(fast) = {fmt(sim['util'][j], 3)}"
                  f"   analyt.(rand) = {fmt(hr.utilization()[j], 3)}")

    # --- economic analysis: optimal staff mix
    print("\n" + "=" * 78)
    print("Economic analysis – TC = Σ Cs_k·s_k + Cw·L  (KRW/hour), routing = fastest")
    print("=" * 78)
    grids = {}
    for name, c in CASES.items():
        names, mus, costs = staff_params(c)
        n_max = max(c["range"])
        TC, L, (b1, b2) = hetero_cost_grid(c["lam"], mus, costs, CW, n_max)
        grids[name] = (TC, (b1, b2), names)
        print(f"\n{name}   (Cs: {names[0]} = {costs[0]:,.0f}, {names[1]} = {costs[1]:,.0f})")
        print(f"  {'s total':>7s}{'best mix':>12s}{'L':>9s}{'Total':>12s}")
        for s_tot in c["range"]:
            cand = [(TC[n1, s_tot - n1], n1) for n1 in range(s_tot + 1)
                    if s_tot - n1 <= n_max and not np.isnan(TC[n1, s_tot - n1])]
            if not cand:
                print(f"  {s_tot:>7d}{'unstable':>12s}")
                continue
            tc, n1 = min(cand)
            mark = "  <-- optimal" if (n1, s_tot - n1) == (b1, b2) else ""
            print(f"  {s_tot:>7d}{f'{n1}S + {s_tot - n1}J':>12s}"
                  f"{fmt(L[n1, s_tot - n1]):>9s}{fmt(tc, 0):>12s}{mark}")
        if b1 + b2 not in c["range"]:
            print(f"  (optimum outside the paper's range) {b1}S + {b2}J  "
                  f"TC = {fmt(TC[b1, b2], 0)}")
        rows, s_opt = homog_results[name]
        tc_hom = next(r["TC"] for r in rows if r["s"] == s_opt)
        diff = (TC[b1, b2] - tc_hom) / tc_hom * 100
        print(f"  optimal mix = {b1} {names[0]} + {b2} {names[1]}  "
              f"TC = {fmt(TC[b1, b2], 0)}   vs. homogeneous s = {s_opt}: "
              f"TC = {fmt(tc_hom, 0)}  ({diff:+.1f}%)")

    # --- Fig: Pn, homogeneous vs heterogeneous
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, c) in zip(axs, CASES.items()):
        _, mus, _ = staff_params(c)
        mix = mixes[name]
        hom = MMs(c["lam"], c["mu"], c["s_analysis"])
        het = HeteroMMs(c["lam"], mus, mix, "fastest")
        sim = simulate_hetero(c["lam"], mus, mix, "fastest", seed=7)
        n_max = 30 if "3" in name else 22
        ns = np.arange(n_max)
        ax.bar(ns, [het.Pn(n) for n in ns], 0.7, color="C0",
               label=f"Heterogeneous {mix[0]}S+{mix[1]}J – analytical")
        ax.plot(ns, [sim["Pn"].get(n, 0) for n in ns], "o", ms=5, color="C1",
                label="Heterogeneous – simulation")
        ax.step(ns, [hom.Pn(n) for n in ns], where="mid", color="0.35", lw=1.5,
                label=f"Homogeneous s={c['s_analysis']} – analytical")
        ax.set_title(name)
        ax.set_xlabel("No. of RFIs in the system (n)")
        ax.set_ylabel("Pn")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "fig_hetero_Pn.png", dpi=130)
    plt.close()

    # --- Fig: total-cost heatmap over the staff mix
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (name, (TC, (b1, b2), names)) in zip(axs, grids.items()):
        tc_min = np.nanmin(TC)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("0.92")  # unstable mixes (ρ ≥ 1)
        im = ax.imshow(np.ma.masked_invalid(TC), origin="lower", cmap=cmap,
                       vmin=tc_min, vmax=2 * tc_min)
        ax.plot(b2, b1, marker="*", ms=16, color="white", mec="black")
        ax.set_title(f"{name}\noptimal mix = {b1} {names[0]} + {b2} {names[1]}")
        ax.set_xlabel(f"No. of {names[1]} staff")
        ax.set_ylabel(f"No. of {names[0]} staff")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("Total cost (KRW/hour, capped at 2× min)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_hetero_costs.png", dpi=130)
    plt.close()
    print(f"\nPlots saved to {OUT}: fig_hetero_Pn.png, fig_hetero_costs.png")


def main():
    homog_results = run_homogeneous()     # PART A: paper (homogeneous servers)
    run_heterogeneous(homog_results)      # PART B: extension (heterogeneous servers)


if __name__ == "__main__":
    main()
