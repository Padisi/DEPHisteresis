"""
Barrido en frecuencia de DEP con electrodos coplanares (|E|^2 = V0^2 / (pi^2 r^2)).
Misma física que barrido_frecuencias_dep.ipynb; usa RadialFokkerPlanck de fokker_planck.py.
Todas las magnitudes en SI salvo la concentración de entrada (mL^-1).
"""
import numpy as np
from fokker_planck import RadialFokkerPlanck

EPS0 = 8.8541878128e-12
KB = 1.380649e-23


class DEPModel:
    def __init__(self, p):
        self.p = p
        self.kT = KB * p["T"]
        self.eps_m = p["eps_m_r"] * EPS0
        self.eps_p = p["eps_p_r"] * EPS0
        self.gamma = 6 * np.pi * p["eta"] * p["a"]
        self.D = self.kT / self.gamma
        self.V_p = 4 / 3 * np.pi * p["a"] ** 3
        self.n0 = p["n0_mL"] * 1e6                       # m^-3
        self.phi0 = self.n0 * self.V_p
        if not 0 < self.phi0 < p["phi_max"]:
            raise ValueError(f"fracción de volumen inicial phi0 = {self.phi0:.3g} fuera de (0, phi_max)")
        self.freqs = np.logspace(np.log10(p["f_min"]), np.log10(p["f_max"]), int(p["n_freq"]))

    def K_CM(self, f):
        p, w = self.p, 2 * np.pi * np.asarray(f, float)
        ep = self.eps_p - 1j * p["sig_p"] / w
        em = self.eps_m - 1j * p["sig_m"] / w
        return (ep - em) / (ep + 2 * em)

    def Omega(self, f):
        p = self.p
        return 2 * np.pi * np.asarray(f, float) * p["Lam"] * self.eps_m * p["l_c"] / (p["sig_m"] * p["lam_D"])

    def S(self, f):
        O = self.Omega(f)
        return O ** 2 / (1 + O ** 2)

    def A(self, f):
        """U(r) = -A / max(r, r_c)^2  (A > 0: pDEP). V0 = amplitud de pico."""
        return np.pi * self.eps_m * self.p["a"] ** 3 * self.K_CM(f).real * self.S(f) * self.p["V0"] ** 2 / np.pi ** 2

    def drift(self, f):
        A, r_c, g = self.A(f), self.p["r_c"], self.gamma
        def v(r):
            r = np.asarray(r, float)
            return np.where(r > r_c, -2 * A / (g * np.maximum(r, r_c) ** 3), 0.0)
        return v

    def N_roi_per_m(self, fp, phi):
        """Partículas en 0 < r < r_ROI por metro de electrodo (semiplano: pi * int n r dr)."""
        m = fp.r < self.p["r_ROI"]
        return np.pi * np.sum(phi[m] * fp.vol[m]) / self.V_p

    def sweep(self, freqs, progress=None, label=""):
        p = self.p
        N = int(p["N"])
        phi = np.full(N, self.phi0)
        N_end, N_eq = [], []
        for k, f in enumerate(freqs):
            fp = RadialFokkerPlanck(p["R"], N, self.D, self.drift(f), p["phi_max"],
                                    outer_bc="dirichlet", P_outer=self.phi0)
            _, P = fp.solve(phi, p["t_step"], n_out=2)
            phi = P[:, -1]
            N_end.append(self.N_roi_per_m(fp, phi))
            N_eq.append(self.N_roi_per_m(fp, fp.steady_state()))
            if progress is not None:
                progress(label, k + 1, len(freqs))
        return np.array(N_end), np.array(N_eq)


def run(params, progress=None):
    """Devuelve un dict serializable con N en la ROI (por m y total) para ambos sentidos."""
    m = DEPModel(params)
    f = m.freqs
    asc, eq = m.sweep(f, progress, "ascendente")
    desc, _ = m.sweep(f[::-1], progress, "descendente")
    desc = desc[::-1]                                   # reordenado por frecuencia creciente
    L = params["L_el"]
    num = (params["sig_p"] - params["sig_m"]) * (params["sig_p"] + 2 * params["sig_m"])
    den = (m.eps_m - m.eps_p) * (m.eps_p + 2 * m.eps_m)
    f_cross = float(np.sqrt(num / den) / (2 * np.pi)) if num / den > 0 else None
    N_roi0 = np.pi * m.n0 * params["r_ROI"] ** 2 / 2 * L  # N inicial en la ROI (uniforme)
    return dict(
        f=f.tolist(),
        ReK=m.K_CM(f).real.tolist(),
        S=m.S(f).tolist(),
        U_rc_kT=(-m.A(f) / params["r_c"] ** 2 / m.kT).tolist(),
        N_asc=(asc * L).tolist(),
        N_desc=(desc * L).tolist(),
        N_eq=(eq * L).tolist(),
        N_roi0=float(N_roi0),
        phi0=float(m.phi0),
        D=float(m.D),
        f_cross=f_cross,
    )
