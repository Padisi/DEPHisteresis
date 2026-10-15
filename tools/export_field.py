"""Export what the browser needs from the 3D field table of the toy (chang_cooper/toy/field_table.npz).

The toy only uses the DEP depth |U|/kT per unit Re K (V0 = 1 V, b = 55 nm) on the grid, for 91 values of the
screening parameter Omega, in two ways (toy/toy_trap.py, precompute):
  * the equilibrium content of the ROI, a weighted sum over the ROI cells (gap edges x widefield focus);
  * the maximum depth over the whole grid.
So we keep the ROI cells whose focus weight is above W_MIN (the rest add < 1e-12 of the signal) and the few
cells that hold the maximum for some Omega (the upper envelope, checked on a fine grid in log Omega).

    PYTHONPATH=.. python tools/export_field.py          (from this repo, inside chang_cooper/)
writes docs/data/field.bin (float32) and docs/data/field.json, and tools/reference.json (outputs of the
notebook's run() for a few parameter sets, to check the JavaScript port with tools/check_port.mjs).
"""
import json
import os

import numpy as np

from toy.toy_trap import FieldTable, roi_weights, precompute, sweep, H
from experiments.electrolyte import EPS0, EPS_WATER, debye_length
from experiments.particles import clausius_mossotti, complex_permittivity
from experiments import observables

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC = os.environ.get("CHANG_COOPER", os.path.dirname(ROOT))      # this repo sits inside chang_cooper/
W_MIN = 1e-12


def envelope_cells(depth):
    """Indices of the cells that are the maximum of the log-Omega interpolated depth for some Omega."""
    flat = depth.reshape(len(depth), -1)
    idx = set()
    for i in range(1, len(flat)):
        for t in np.linspace(0, 1, 101):
            idx.add(int(np.argmax((1 - t) * flat[i - 1] + t * flat[i])))
    return np.array(sorted(idx))


def main():
    table = FieldTable(os.path.join(CC, "toy", "field_table.npz"))
    w = roi_weights(table)
    keep = np.flatnonzero(w.ravel() > W_MIN)
    flat = table.depth.reshape(len(table.Omega), -1)
    env = envelope_cells(table.depth)
    d_focus = observables.widefield(table.gz, z_focus=0.0, dof=0.75).sum() * (table.gz[1] - table.gz[0])
    data = np.concatenate([flat[:, keep], flat[:, env]], axis=1).astype(np.float32)
    data.tofile(os.path.join(ROOT, "docs", "data", "field.bin"))
    meta = dict(Omega=table.Omega.tolist(), n_omega=len(table.Omega), n_roi=len(keep), n_env=len(env),
                w=w.ravel()[keep].tolist(), w_sum=float(w.sum()), cell_volume=H ** 3, d_focus=float(d_focus),
                note="field.bin: float32 [n_omega][n_roi + n_env], |U|/kT per unit Re K at V0 = 1 V, b = 55 nm; "
                     "first n_roi columns: ROI cells with weights w; last n_env: cells holding the max depth")
    with open(os.path.join(ROOT, "docs", "data", "field.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"{len(keep)} ROI cells (of {(w > 0).sum()}), {len(env)} envelope cells, "
          f"{data.nbytes / 1e6:.2f} MB, d_focus = {d_focus:.3f} um")

    # reference outputs of the playground's run() (toy/fig2_ps_playground.ipynb) with an optional ACEO gain
    from toy.fig2_ps import F_DATA
    TOY = dict(J0=42.9, beta=0.602, gamma=0.238, a_eff=10.0, ell=0.8)
    cases = [dict(), dict(A=0.1911), dict(sigma_m=100.0, sigma_p=50.0, radius=60.0, b_eff=60.0, Lambda=0.05, V0=3.0,
                                         h_ch=50.0, A=2.0), dict(Lambda=0.3, V0=0.3, b_eff=500.0)]
    ref = []
    for c in cases:
        p = dict(sigma_m=20.0, sigma_p=188.0, radius=100.0, b_eff=297.0, Lambda=0.0036, V0=1.0, h_ch=9.8, A=0.0)
        p.update(c)
        sm, sp = p["sigma_m"] * 1e-4, p["sigma_p"] * 1e-4
        b, be = p["radius"], max(p["b_eff"], p["radius"])
        ReK = clausius_mossotti(complex_permittivity(2.55 * EPS0, sp, F_DATA),
                                complex_permittivity(EPS_WATER, sm, F_DATA)).real
        lam = debye_length(0.0, c_KCl_mM=sm / 149.9e-4)
        Omega = 2 * np.pi * F_DATA * p["Lambda"] * EPS_WATER * 1e-6 / (sm * lam)
        pre = precompute(F_DATA, ReK, Omega, V0=p["V0"], b_nm=b, c_inf_per_mL=1e11, phi_max=0.64 * (b / be) ** 3,
                         table=table)
        gain = 1 + p["A"] * (1e6 / F_DATA) ** 2
        Hh = p["h_ch"] / d_focus
        out = dict(params=p, Ieq=pre["Ieq"].tolist(), Umax=pre["Umax"].tolist())
        for d in ("up", "down"):
            t, I, f, ends = sweep(pre, d, t_s=2.0, n_out=10, gain=gain, **TOY)
            S, S_end = (I + Hh) / (1 + Hh), (ends + Hh) / (1 + Hh)
            o = np.argsort(f)
            out[d] = dict(I=I.tolist(), end=(S_end / S.max())[o].tolist(), raw_end=ends[o].tolist())
        ref.append(out)
    with open(os.path.join(ROOT, "tools", "reference.json"), "w") as fh:
        json.dump(ref, fh)
    print("wrote docs/data/field.{bin,json} and tools/reference.json")


if __name__ == "__main__":
    main()
