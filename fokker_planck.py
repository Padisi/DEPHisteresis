"""
Fokker-Planck 1D radial en coordenadas cilíndricas con volumen excluido.

Ecuación (simetría azimutal y axial, sólo depende de r):

    dP/dt = -(1/r) d/dr [ r J ]
    J(r)  = v(r) P (1 - P/P_max) - D dP/dr

donde v(r) es la velocidad de deriva (v = -U'(r)/gamma; v < 0 empuja hacia
el centro). El factor (1 - P/P_max) sobre el término advectivo implementa el
volumen excluido: la deriva se apaga cuando la densidad local alcanza P_max.

Discretización: volúmenes finitos conservativos en celdas [r_{i-1/2}, r_{i+1/2}]
con volumen de celda (r_{i+1/2}^2 - r_{i-1/2}^2)/2 (la masa total
2*pi*int P r dr se conserva a precisión de máquina con flujo nulo en bordes).
Flujo advectivo upwind con movilidad no lineal:

    v > 0:  F = v P_i     (1 - P_{i+1}/P_max)
    v < 0:  F = v P_{i+1} (1 - P_i    /P_max)

que mantiene 0 <= P <= P_max. Integración temporal implícita (BDF de scipy)
con jacobiano tridiagonal disperso, de modo que el problema rígido de una
trampa muy empinada no obliga a pasos de tiempo diminutos.

Solución estacionaria con flujo nulo (útil para validar):
    D P' = v P (1 - P/P_max)  =>  P(r) = P_max / (1 + exp((U(r) - mu)/D))
(distribución tipo Fermi-Dirac, mu fijado por la masa total).
"""

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq
from scipy.sparse import diags


class RadialFokkerPlanck:
    def __init__(self, R, N, D, drift, P_max, outer_bc="noflux", P_outer=0.0,
                 kappa=None):
        """
        R      : radio del dominio
        N      : número de celdas
        D      : coeficiente de difusión (escalar o función de r)
        drift  : función v(r) (velocidad radial de deriva)
        P_max  : densidad máxima (empaquetamiento)
        outer_bc : "noflux"    (sistema cerrado),
                   "dirichlet" (densidad fija P_outer en r=R),
                   "robin"     (reservorio: J(R) = kappa (c(R) - P_outer), J>0 hacia fuera)
        P_outer  : densidad del reservorio (c_inf)
        kappa    : coeficiente de transferencia con el reservorio (sólo "robin");
                   kappa -> 0 recupera flujo nulo, kappa -> inf recupera Dirichlet
        """
        if outer_bc not in ("noflux", "dirichlet", "robin"):
            raise ValueError(f"outer_bc desconocido: {outer_bc}")
        if outer_bc == "robin" and (kappa is None or kappa < 0):
            raise ValueError("outer_bc='robin' requiere kappa >= 0")
        if outer_bc != "noflux" and not 0 <= P_outer <= P_max:
            raise ValueError("se requiere 0 <= P_outer <= P_max")
        self.R, self.N, self.P_max = R, N, P_max
        self.faces = np.linspace(0.0, R, N + 1)
        self.r = 0.5 * (self.faces[1:] + self.faces[:-1])
        self.dr = R / N
        self.vol = 0.5 * (self.faces[1:] ** 2 - self.faces[:-1] ** 2)  # r dr exacto

        self.D_f = np.broadcast_to(D(self.faces) if callable(D) else D,
                                   self.faces.shape).astype(float)
        self.v_f = np.asarray(drift(self.faces), dtype=float)
        self.drift = drift
        self.D = D
        self.outer_bc = outer_bc
        self.P_outer = P_outer
        self.kappa = kappa

    # ---------------------------------------------------------------- flujos
    def fluxes(self, P):
        """Flujo J en las N+1 caras (ya multiplicado por r de la cara)."""
        Pm = self.P_max
        J = np.zeros(self.N + 1)
        PL, PR = P[:-1], P[1:]                   # caras interiores 1..N-1
        v = self.v_f[1:-1]
        adv = np.where(v > 0, v * PL * (1 - PR / Pm), v * PR * (1 - PL / Pm))
        dif = -self.D_f[1:-1] * (PR - PL) / self.dr
        J[1:-1] = adv + dif
        # r = 0: flujo nulo por simetría (r_face = 0 de todas formas)
        if self.outer_bc == "dirichlet":
            PL, PR, v = P[-1], self.P_outer, self.v_f[-1]
            adv = v * PL * (1 - PR / Pm) if v > 0 else v * PR * (1 - PL / Pm)
            J[-1] = adv - self.D_f[-1] * (PR - PL) / (0.5 * self.dr)
        elif self.outer_bc == "robin":
            # c(R) aproximado por la celda más externa. Como 0 <= P_outer <= P_max,
            # el flujo entrante se anula antes de que P supere P_outer <= P_max.
            J[-1] = self.kappa * (P[-1] - self.P_outer)
        return J * self.faces

    def rhs(self, t, P):
        F = self.fluxes(P)
        return -(F[1:] - F[:-1]) / self.vol

    def jac_sparsity(self):
        return diags([1, 1, 1], [-1, 0, 1], shape=(self.N, self.N), dtype=float)

    # -------------------------------------------------------------- utilidades
    def mass(self, P):
        """Número de partículas por unidad de longitud: 2*pi*int P r dr."""
        P = np.asarray(P)
        vol = self.vol if P.ndim == 1 else self.vol[:, None]
        return 2 * np.pi * np.sum(P * vol, axis=0)

    def solve(self, P0, t_end, n_out=200, method="BDF", rtol=1e-6, atol=1e-10):
        t_eval = np.linspace(0, t_end, n_out)
        sol = solve_ivp(self.rhs, (0, t_end), P0, method=method, t_eval=t_eval,
                        jac_sparsity=self.jac_sparsity(), rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.t, sol.y   # y: (N, n_out)

    def steady_state(self, mass=None):
        """Estacionario Fermi-Dirac con flujo nulo (D constante).

        - "noflux": mu fijado por la masa total `mass`.
        - "dirichlet"/"robin": flujo nulo exige c(R) = P_outer, lo que fija mu
          sin necesidad de `mass` (para "robin", c(R) es la última celda).
        """
        if callable(self.D):
            raise NotImplementedError("steady_state asume D constante")
        r = self.r
        U = -cumulative_trapezoid(self.drift(r), r, initial=0.0)  # U = -int v dr
        x = U / self.D

        def P_of(mu):
            return self.P_max / (1 + np.exp(np.clip(x - mu, -700, 700)))

        if self.outer_bc != "noflux":
            c = self.P_outer
            if c <= 0:
                return np.zeros_like(r)
            if c >= self.P_max:
                return np.full_like(r, self.P_max)
            if self.outer_bc == "dirichlet":     # la condición se impone en r=R
                rr = np.append(r, self.R)
                x = -cumulative_trapezoid(self.drift(rr), rr, initial=0.0) / self.D
                mu = x[-1] - np.log(self.P_max / c - 1)
                x = x[:-1]
            else:
                mu = x[-1] - np.log(self.P_max / c - 1)
            return P_of(mu)

        if mass is None:
            raise ValueError("con outer_bc='noflux' hay que dar la masa total")
        f = lambda mu: self.mass(P_of(mu)) - mass
        mu = brentq(f, x.min() - 800, x.max() + 800)
        return P_of(mu)


# ------------------------------------------------------------- derivas típicas
def steep_trap_drift(v0, r_trap, n, v_cap=None):
    """Trampa empinada U = (v0 r_trap / n) (r/r_trap)^n  ->  v = -v0 (r/r_trap)^(n-1).

    v_cap (opcional) limita |v|: para n grande la deriva lejos de r_trap crece
    como (R/r_trap)^(n-1), lo que no cambia la física (todo cae a la trampa)
    pero vuelve el problema muy rígido y lento de integrar.
    """
    def v(r):
        out = -v0 * (np.asarray(r, dtype=float) / r_trap) ** (n - 1)
        return out if v_cap is None else np.maximum(out, -abs(v_cap))
    return v


def step_trap_drift(v0, r_trap, width):
    """Pared suave: deriva nula dentro de r_trap y -v0 fuera (transición tanh)."""
    return lambda r: -0.5 * v0 * (1 + np.tanh((np.asarray(r) - r_trap) / width))
