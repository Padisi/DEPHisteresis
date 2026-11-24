import numpy as np
import matplotlib.pyplot as plt
from fokker_planck import RadialFokkerPlanck, steep_trap_drift

# Parámetros
R, N = 3.0, 300
D = 0.05
P_max = 0.64
drift = steep_trap_drift(v0=1.0, r_trap=1.0, n=8)   # trampa empinada

fp = RadialFokkerPlanck(R, N, D, drift, P_max)

# Condición inicial: concentración uniforme baja
P0 = np.full(N, 0.1)
M0 = fp.mass(P0)

t, P = fp.solve(P0, t_end=50.0, n_out=6)
P_ss = fp.steady_state(M0)

print(f"masa inicial {M0:.6f}, final {fp.mass(P[:, -1]):.6f}, "
      f"error rel {abs(fp.mass(P[:, -1]) - M0) / M0:.2e}")
print(f"min P {P.min():.3e}, max P {P.max():.4f} (P_max={P_max})")
print(f"|P(t_end) - P_ss|_inf = {np.abs(P[:, -1] - P_ss).max():.2e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k in range(len(t)):
    ax[0].plot(fp.r, P[:, k], label=f"t={t[k]:.0f}")
ax[0].plot(fp.r, P_ss, "k--", label="estacionario (Fermi-Dirac)")
ax[0].axhline(P_max, color="gray", lw=0.8, ls=":")
ax[0].set_xlabel("r"); ax[0].set_ylabel("P(r,t)"); ax[0].legend()

# comparación sin volumen excluido (P_max -> infinito)
fp_free = RadialFokkerPlanck(R, N, D, drift, P_max=1e12)
_, Pf = fp_free.solve(P0, t_end=50.0, n_out=2)
ax[1].plot(fp.r, P[:, -1], label="con volumen excluido")
ax[1].plot(fp.r, Pf[:, -1], label="sin volumen excluido")
ax[1].axhline(P_max, color="gray", lw=0.8, ls=":")
ax[1].set_xlabel("r"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-4, 2)
ax[1].legend()
plt.tight_layout()
plt.savefig("fp_radial.png", dpi=120)
