"""Minimal toy model of the DEP trap: diffusion + depth + capacity (from toy/trap_diffusion_toy.ipynb).

One equation for the number N of trapped particles during a frequency staircase:

    dN/dt = g J_D [ (1 - N/N_cap) - (N/N_res) exp(-U(f)) ],    g = 1 / (1 + k_eq tau_l)

with J_D = 4 D c_inf a_trap (diffusion-limited capture), U(f) = U0 Re K(f) S(f) (trap depth in kT),
N_cap = phi_max V_trap / V_NP (packing), N_res = c_inf V_trap, k_eq = (J_D / N_res) exp(-U) and
tau_l = l_trap^2 / (D (1 + max(-U, 0))) the time to cross the trap. Linear in N: each step is solved exactly.

The same equations are implemented in JavaScript in docs/model.js (the GitHub Pages playground).

    python -m toy.trap_toy            prints the hysteresis of the default case (Au@SiO2, 1 mM CTAC)
"""
import numpy as np

EPS0, KT, ETA = 8.8541878128e-12, 4.114e-21, 1.0e-3     # F/m, J, Pa s
EPS_M = 78.5 * EPS0
KCL_MOLAR_COND = 149.9e-4                               # S m^2/mol: medium treated as KCl to get lambda_D
SIGMA_AU, EPS_SIO2 = 4.1e7, 3.9 * EPS0                  # Au core, SiO2 shell
ELL = 1e-6                                              # m, length in Omega (as in the 3D runs)

# Defaults: Au@SiO2 in 1 mM CTAC, the Fig. 3f experiment. K_s = Bikerman conductance x Stern factor 14.6, fitted
# so that the crossover is at sqrt(7 x 10) MHz = 8.4 MHz (measured 7 MHz up, 10 MHz down).
P = dict(
    model="au_sio2",   #      "au_sio2" (core-shell) or "homogeneous" (eps_p, sigma_p, radius)
    a_core=35.0,       # nm      Au core radius
    t_shell=20.0,      # nm      SiO2 shell thickness (outer radius b = 55 nm)
    K_s=1.30,          # nS      surface conductance (1 mM CTAC: 0.22 at 0.1 mM, 3.15 at 10 mM)
    sigma_m=118.0,     # uS/cm   medium conductivity (0.1 mM CTAC: 45, 10 mM: 427)
    debye_nm=9.6,      # nm      Debye length (0.1 mM: 30.4, 10 mM: 4.6); None: from sigma_m as KCl
    Lambda=0.3,        #         Stern factor of the electrodes (screening)
    V0=1.0,            # V       voltage amplitude between neighbouring plates
    gap=0.6,           # um      electrode gap
    a_trap=10.0,       # um      capture radius of the trap (diffusive flux 4 D c a)
    l_trap=1.0,        # um      trap size: particles need at least tau_l = l^2/D to leave it
    V_trap=25.0,       # um^3    trap volume (capacity); 25 um^3 = ROI of the 3D runs
    phi_max=0.64,      #         packing fraction of the full trap
    c_inf=1e11,        # 1/mL    bulk concentration
    t_step=2.0,        # s       time per frequency
    f_min=1e5,         # Hz
    f_max=1e7,         # Hz
    n_f=30,            #         frequencies (log spaced)
    radius=100.0,      # nm      (model="homogeneous")
    eps_p=2.55,        #         (model="homogeneous")
    sigma_p=188.0,     # uS/cm   (model="homogeneous"), surface conductance included
)
# 200 nm PS of Fig. 2
PS = dict(model="homogeneous", radius=100.0, eps_p=2.55, sigma_p=188.0, sigma_m=20.0, debye_nm=None, Lambda=0.0036,
          c_inf=5e10)


def physics(f, p):
    "Re K, screening S, depth U (kT), and the constants of the trap at frequencies f."
    sm = p["sigma_m"] * 1e-4                                              # S/m
    w_ = 2 * np.pi * f
    em = EPS_M - 1j * sm / w_
    if p["model"] == "au_sio2":
        a, b = p["a_core"] * 1e-9, (p["a_core"] + p["t_shell"]) * 1e-9
        ec, es = EPS0 - 1j * SIGMA_AU / w_, EPS_SIO2 + 0j
        G = (ec - es) / (ec + 2 * es)
        ep = es * ((b / a)**3 + 2 * G) / ((b / a)**3 - G) - 1j * 2 * p["K_s"] * 1e-9 / b / w_
    else:
        b = p["radius"] * 1e-9
        ep = p["eps_p"] * EPS0 - 1j * p["sigma_p"] * 1e-4 / w_
    ReK = ((ep - em) / (ep + 2 * em)).real
    if p["debye_nm"] is None:
        lam = 0.304e-9 / np.sqrt(sm / KCL_MOLAR_COND * 1e-3)             # Debye length (m): 0.304 nm / sqrt(c [M])
    else:
        lam = p["debye_nm"] * 1e-9
    Omega = w_ * p["Lambda"] * EPS_M / lam * ELL / sm
    S = Omega**2 / (1 + Omega**2)
    U0 = np.pi * EPS_M * b**3 * (p["V0"] / (p["gap"] * 1e-6))**2 / KT
    U = U0 * ReK * S
    D = KT / (6 * np.pi * ETA * b) * 1e12                                 # um^2/s
    c = p["c_inf"] * 1e-12                                                # 1/um^3
    N_cap = p["phi_max"] * p["V_trap"] / (4 / 3 * np.pi * (b * 1e6)**3)
    N_res = c * p["V_trap"]
    J_D = 4 * D * c * p["a_trap"]                                         # particles/s
    with np.errstate(over="ignore"):
        N_eq = N_cap / (1 + N_cap / N_res * np.exp(-U))
    tau_l = p["l_trap"]**2 / D / (1 + np.maximum(-U, 0))                  # s, diffusion + nDEP drift
    log_x = np.log(J_D * tau_l / N_res) - U                               # log(k_eq tau_l)
    g = 0.5 * (1 - np.tanh(log_x / 2))                                    # 1 / (1 + k_eq tau_l), overflow-safe
    gk = 0.5 * (1 + np.tanh(log_x / 2)) / tau_l                           # g k_eq = escape rate per particle
    return dict(f=f, ReK=ReK, S=S, Omega=Omega, U=U, U0=U0, D=D, N_cap=N_cap, N_res=N_res, J_D=J_D, N_eq=N_eq,
                tau_l=tau_l, g=g, k_out=gk, lam=lam, f_c=sm / (2 * np.pi * EPS_M))


def sweep(ph, direction, t_step, n_out=40):
    "Staircase over the frequencies; exact solution of dN/dt = A - B N on each step. Starts from N = N_res."
    order = np.argsort(ph["f"])[::-1] if direction == "down" else np.argsort(ph["f"])
    J, Nc, Nr = ph["J_D"], ph["N_cap"], ph["N_res"]
    s = np.linspace(0, t_step, n_out + 1)[1:]
    N, t0 = Nr, 0.0
    ts, Ns, ends = [0.0], [N], []
    for i in order:
        A = ph["g"][i] * J                                    # capture (empty trap), particles/s
        B = A / Nc + ph["k_out"][i]                           # dN/dt = A - B N
        N_inf = A / B
        tr = N_inf + (N - N_inf) * np.exp(-B * s)
        ts.extend(t0 + s)
        Ns.extend(tr)
        N, t0 = tr[-1], t0 + t_step
        ends.append(N)
    o = np.argsort(ph["f"][order])
    return dict(t=np.array(ts), N=np.array(Ns), f=ph["f"][order][o], end=np.array(ends)[o])


def run(**kw):
    p = dict(P, **kw)
    f = np.logspace(np.log10(p["f_min"]), np.log10(p["f_max"]), int(p["n_f"]))
    ph = physics(f, p)
    return p, ph, {d: sweep(ph, d, p["t_step"]) for d in ("up", "down")}


def hysteresis(sw):
    """Hysteresis measures of the end-of-step curves N_up(f), N_down(f):
    area = integral |N_down - N_up| d log10 f, normalised by max N (in decades),
    max_gap = max |N_down - N_up| / max N, and the frequency where it happens."""
    f, up, down = sw["up"]["f"], sw["up"]["end"], sw["down"]["end"]
    top = max(up.max(), down.max(), 1e-300)
    gap = np.abs(down - up)
    i = int(np.argmax(gap))
    area = np.sum(0.5 * (gap[1:] + gap[:-1]) * np.diff(np.log10(f)))
    return dict(area=float(area / top),max_gap=float(gap[i] / top), f_max_gap=float(f[i]))


if __name__ == "__main__":
    p, ph, sw = run()
    h = hysteresis(sw)
    print(f"Au@SiO2, 1 mM CTAC: hysteresis area {h['area']:.4f} decades, max gap {h['max_gap']:.4f} "
          f"at {h['f_max_gap'] / 1e6:.3f} MHz")
    for f, u, d in zip(sw["up"]["f"], sw["up"]["end"], sw["down"]["end"]):
        print(f"{f:12.4g} {u:12.5g} {d:12.5g}")
