// DEP trap toy model, Fig. 2 (200 nm PS) playground: JavaScript port of chang_cooper/toy/fig2_ps_playground.ipynb
// (run / crossover) and toy/toy_trap.py (precompute / sweep), plus the optional ACEO supply gain of toy/fig2_ps.py.
// Works in the browser (window.Fig2Toy) and in node (module.exports).
(function (root) {
  "use strict";
  const EPS0 = 8.8541878128e-12;
  const KT = 4.114e-21, ETA = 1.0e-3;
  const KCL_S_PER_M_PER_MM = 149.9e-4;
  const C_INF = 1e11, T_STEP = 2.0, F_REF = 1e6;
  const EPS_TABLE = 78.5;   // relative permittivity of the medium in the 3D field table
  // toy internals, calibrated on the 3D Fokker-Planck runs (no flow)
  const TOY = { J0: 42.9, beta: 0.602, gamma: 0.238, a_eff: 10.0, ell: 0.8 };
  // defaults = fit to Fig. 2 (saturation scenario); A = ACEO supply gain amplitude (0 = no ACEO)
  const DEFAULTS = { sigma_m: 20.0, sigma_p: 188.0, radius: 100.0, b_eff: 297.0, Lambda: 0.0036, V0: 1.0, h_ch: 9.8,
                     A: 0.0, eps_p: 2.55, eps_m: 78.5 };
  const A_FIT = 0.1911;   // weak-flow scenario of toy/fig2_ps.py (with its own Lambda, V0, H)

  // draft Fig. 2, read by eye (red = up sweep): 30 frequencies 100 kHz - 10 MHz, normalised to max = 1
  const F_DATA = Array.from({ length: 30 }, (_, i) => Math.pow(10, 5 + 2 * i / 29));
  const UP_DATA = [0.13, 0.23, 0.26, 0.29, 0.36, 0.44, 0.51, 0.57, 0.66, 0.74, 0.80, 0.85, 0.89, 0.91, 0.93,
                   0.94, 0.94, 0.95, 0.95, 0.96, 0.97, 0.99, 0.86, 0.55, 0.36, 0.30, 0.24, 0.22, 0.21, 0.20];
  const DOWN_DATA = [0.29, 0.39, 0.49, 0.64, 0.75, 0.87, 0.93, 0.97, 0.98, 0.95, 0.91, 0.82, 0.75, 0.67, 0.60,
                     0.54, 0.50, 0.44, 0.40, 0.35, 0.24, 0.18, 0.15, 0.10, 0.12, 0.15, 0.13, 0.15, 0.14, 0.12];

  // Re K of a homogeneous sphere (eps_p, sigma_p) in a medium (eps_m, sigma_m); eps relative, sigma in S/m
  function reK(f, sp, sm, ep = 2.55, em = EPS_TABLE) {
    const w = 2 * Math.PI * f;
    const pr = ep * EPS0, pi = -sp / w, mr = em * EPS0, mi = -sm / w;
    const nr = pr - mr, ni = pi - mi, dr = pr + 2 * mr, di = pi + 2 * mi;
    return (nr * dr + ni * di) / (dr * dr + di * di);
  }

  function crossover(sm, sp, ep, em) {
    let prev = null;
    for (let i = 0; i < 2000; i++) {
      const f = Math.pow(10, 3 + 6 * i / 1999), k = reK(f, sp, sm, ep, em);
      if (prev !== null && Math.sign(k) !== Math.sign(prev.k)) return prev.f;
      prev = { f, k };
    }
    return NaN;
  }

  // field: { Omega, logO, nO, nRoi, nEnv, w, wSum, Vg, dFocus, data: Float32Array [nO][nRoi + nEnv] }
  function makeField(meta, buffer) {
    return { Omega: meta.Omega, logO: meta.Omega.map(Math.log), nO: meta.n_omega, nRoi: meta.n_roi, nEnv: meta.n_env,
             w: Float64Array.from(meta.w), wSum: meta.w_sum, Vg: meta.cell_volume, dFocus: meta.d_focus,
             data: new Float32Array(buffer) };
  }

  // linear interpolation in log Omega, as FieldTable.depth_per_ReK: returns [i0, i1, t]
  function interpIndex(field, Omega) {
    const L = field.logO, n = L.length;
    const x = Math.min(Math.max(Math.log(Omega), L[0]), L[n - 1]);
    let i = 0;
    while (i < n && L[i] < x) i++;                     // np.searchsorted (left)
    i = Math.min(Math.max(i, 1), n - 1);
    return [i - 1, i, (x - L[i - 1]) / (L[i] - L[i - 1])];
  }

  // eps_m: relative permittivity of the medium; U scales with it (alpha = 4 pi eps_m b^3 Re K), the table is at 78.5
  function precompute(field, freqs, ReK, Omega, V0, b_nm, phi_max, eps_m = EPS_TABLE) {
    const Vg = field.Vg, stride = field.nRoi + field.nEnv;
    const P_max = phi_max * Vg / (4 / 3 * Math.PI * Math.pow(b_nm * 1e-3, 3));
    const P_res = C_INF * 1e-12 * Vg;
    const scale = V0 * V0 * Math.pow(b_nm / 55.0, 3) * eps_m / EPS_TABLE;
    const Umax = [], Ieq = [];
    for (let j = 0; j < freqs.length; j++) {
      const [i0, i1, t] = interpIndex(field, Omega[j]);
      const o0 = i0 * stride, o1 = i1 * stride, k = ReK[j], d = field.data;
      let sum = 0;
      for (let c = 0; c < field.nRoi; c++) {
        const U = -k * scale * ((1 - t) * d[o0 + c] + t * d[o1 + c]);
        const P = P_max / (1 + (P_max / P_res - 1) * Math.exp(U));   // exp overflow -> P = 0, as numpy
        sum += field.w[c] * P;
      }
      Ieq.push(sum / (field.wSum * P_res));
      let m = 0;
      for (let c = field.nRoi; c < stride; c++) m = Math.max(m, (1 - t) * d[o0 + c] + t * d[o1 + c]);
      Umax.push(Math.abs(k * scale * m));
    }
    const D = KT / (6 * Math.PI * ETA * b_nm * 1e-9) * 1e12;
    return { freqs, ReK, Omega, Umax, Ieq, D, V_roi: 4 * field.wSum * Vg };
  }

  function sweep(pre, direction, gain, t_s = T_STEP, dt = 0.01, n_out = 10) {
    const p = TOY, n = pre.freqs.length;
    const idx = [...Array(n).keys()].sort((a, b) => pre.freqs[a] - pre.freqs[b]);
    const order = direction === "down" ? idx.slice().reverse() : idx;
    const D = pre.D, R = 4 * D * p.a_eff / pre.V_roi, kD = D / (p.ell * p.ell), kb = R / p.J0;
    const n_sub = Math.max(Math.round(t_s / dt), 1), h = t_s / n_sub, every = Math.max(Math.floor(n_sub / n_out), 1);
    let I = 1.0, J = p.J0, t = 0;
    const ts = [0], Is = [1.0], fs = [order[0] !== undefined ? pre.freqs[order[0]] : 0], ends = new Array(n);
    for (const i of order) {
      const Ieq = pre.Ieq[i], U = pre.Umax[i], k = pre.ReK[i];
      const k_out = k > 0 ? kD * Math.exp(-p.beta * U) : kD * (1 + U);
      let kb_i = kb * gain[i], J_target = p.J0;
      if (k <= 0 && p.gamma > 0) { J_target = p.J0 * Math.exp(-p.gamma * U); kb_i += kD; }
      for (let s = 0; s < n_sub; s++) {
        const rate = I < Ieq ? kD * J / Ieq : k_out;
        const I_new = Ieq + (I - Ieq) * Math.exp(-rate * h);
        J = Math.max(J_target + (J - J_target) * Math.exp(-kb_i * h) - (I_new - I), 0);
        I = I_new; t += h;
        if ((s + 1) % every === 0) { ts.push(t); Is.push(I); fs.push(pre.freqs[i]); }
      }
      ends[i] = I;
    }
    return { t: ts, I: Is, fAt: fs, ends };
  }

  // one up + one down sweep at the Fig. 2 frequencies, as run() in the playground notebook
  function run(field, prm) {
    const q = Object.assign({}, DEFAULTS, prm);
    const sm = q.sigma_m * 1e-4, sp = q.sigma_p * 1e-4;
    const b = q.radius, be = Math.max(q.b_eff, q.radius);
    const H = q.h_ch / field.dFocus;
    const freqs = F_DATA;
    const ReK = freqs.map(f => reK(f, sp, sm, q.eps_p, q.eps_m));
    // Debye length of the KCl-equivalent medium: 0.304 nm / sqrt(c [M]) in water, proportional to sqrt(eps_m)
    const lam = 0.304e-9 / Math.sqrt(sm / KCL_S_PER_M_PER_MM * 1e-3) * Math.sqrt(q.eps_m / EPS_TABLE);
    const Omega = freqs.map(f => 2 * Math.PI * f * q.Lambda * q.eps_m * EPS0 * 1e-6 / (sm * lam));
    const phi_max = 0.64 * Math.pow(b / be, 3);
    const pre = precompute(field, freqs, ReK, Omega, q.V0, b, phi_max, q.eps_m);
    const gain = freqs.map(f => 1 + q.A * Math.pow(F_REF / f, 2));
    const out = { pre, H, gain, f_x: crossover(sm, sp, q.eps_p, q.eps_m), f_c: sm / (2 * Math.PI * q.eps_m * EPS0), lam, params: q };
    for (const d of ["up", "down"]) {
      const s = sweep(pre, d, gain);
      const S = s.I.map(I => (I + H) / (1 + H)), Smax = Math.max(...S);
      out[d] = { t: s.t, I: s.I, S: S.map(x => x / Smax), fAt: s.fAt, f: freqs,
                 end: s.ends.map(I => (I + H) / (1 + H) / Smax), raw_end: s.ends };
    }
    let e = 0;
    for (let i = 0; i < 30; i++) e += ((out.up.end[i] - UP_DATA[i]) ** 2 + (out.down.end[i] - DOWN_DATA[i]) ** 2) / 30;
    out.rms = Math.sqrt(e / 2);
    return out;
  }

  // hysteresis of the normalised end-of-step curves: area = integral |down - up| d log10 f (decades),
  // signed (> 0: down sweep brighter), max gap and where
  function hysteresis(m) {
    const f = m.up.f, u = m.up.end, d = m.down.end;
    let area = 0, signed = 0, iMax = 0;
    for (let i = 0; i < f.length; i++) {
      if (Math.abs(d[i] - u[i]) > Math.abs(d[iMax] - u[iMax])) iMax = i;
      if (i > 0) {
        const dx = Math.log10(f[i]) - Math.log10(f[i - 1]);
        area += 0.5 * (Math.abs(d[i] - u[i]) + Math.abs(d[i - 1] - u[i - 1])) * dx;
        signed += 0.5 * ((d[i] - u[i]) + (d[i - 1] - u[i - 1])) * dx;
      }
    }
    return { area, signed, max_gap: Math.abs(d[iMax] - u[iMax]), f_max_gap: f[iMax] };
  }

  const api = { DEFAULTS, TOY, A_FIT, F_REF, F_DATA, UP_DATA, DOWN_DATA, makeField, precompute, sweep, run, hysteresis,
                crossover, reK };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Fig2Toy = api;
})(this);
