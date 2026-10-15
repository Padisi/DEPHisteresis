// Toy model of the DEP trap: diffusion + depth + capacity.
// JavaScript port of toy/trap_toy.py (same equations, same defaults). Works in the browser and in node.
(function (root) {
  "use strict";
  const EPS0 = 8.8541878128e-12, KT = 4.114e-21, ETA = 1.0e-3;   // F/m, J, Pa s
  const EPS_M = 78.5 * EPS0;
  const KCL_MOLAR_COND = 149.9e-4;                                 // S m^2/mol
  const SIGMA_AU = 4.1e7, EPS_SIO2 = 3.9 * EPS0;
  const ELL = 1e-6;                                                // m, length in Omega

  const DEFAULTS = {
    model: "au_sio2", a_core: 35.0, t_shell: 20.0, K_s: 1.30, sigma_m: 118.0, debye_nm: 9.6, Lambda: 0.3,
    V0: 1.0, gap: 0.6, a_trap: 10.0, l_trap: 1.0, V_trap: 25.0, phi_max: 0.64, c_inf: 1e11, t_step: 2.0,
    f_min: 1e5, f_max: 1e7, n_f: 30, radius: 100.0, eps_p: 2.55, sigma_p: 188.0,
  };
  const PRESETS = {
    "Au@SiO2, 0.1 mM CTAC": { model: "au_sio2", sigma_m: 45.0, debye_nm: 30.4, K_s: 0.22 },
    "Au@SiO2, 1 mM CTAC": { model: "au_sio2" },
    "Au@SiO2, 10 mM CTAC": { model: "au_sio2", sigma_m: 427.0, debye_nm: 4.6, K_s: 3.15 },
    "200 nm PS (Fig. 2)": { model: "homogeneous", radius: 100.0, eps_p: 2.55, sigma_p: 188.0, sigma_m: 20.0,
                            debye_nm: null, Lambda: 0.0036, c_inf: 5e10 },
  };

  // complex numbers as [re, im]
  const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const csub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  const cscale = (a, s) => [a[0] * s, a[1] * s];
  const cdiv = (a, b) => {
    const d = b[0] * b[0] + b[1] * b[1];
    return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
  };

  function logspace(lo, hi, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.pow(10, n === 1 ? lo : lo + (hi - lo) * i / (n - 1));
    return out;
  }

  function physics(f, p) {
    const sm = p.sigma_m * 1e-4;
    let b;
    if (p.model === "au_sio2") b = (p.a_core + p.t_shell) * 1e-9;
    else b = p.radius * 1e-9;
    const lam = (p.debye_nm === null || p.debye_nm === undefined)
      ? 0.304e-9 / Math.sqrt(sm / KCL_MOLAR_COND * 1e-3) : p.debye_nm * 1e-9;
    const U0 = Math.PI * EPS_M * b ** 3 * (p.V0 / (p.gap * 1e-6)) ** 2 / KT;
    const D = KT / (6 * Math.PI * ETA * b) * 1e12;                    // um^2/s
    const c = p.c_inf * 1e-12;                                        // 1/um^3
    const N_cap = p.phi_max * p.V_trap / (4 / 3 * Math.PI * (b * 1e6) ** 3);
    const N_res = c * p.V_trap;
    const J_D = 4 * D * c * p.a_trap;
    const n = f.length;
    const ReK = new Array(n), S = new Array(n), Omega = new Array(n), U = new Array(n), N_eq = new Array(n),
      tau_l = new Array(n), g = new Array(n), k_out = new Array(n);
    for (let i = 0; i < n; i++) {
      const w = 2 * Math.PI * f[i];
      const em = [EPS_M, -sm / w];
      let ep;
      if (p.model === "au_sio2") {
        const a = p.a_core * 1e-9, r3 = (b / a) ** 3;
        const ec = [EPS0, -SIGMA_AU / w], es = [EPS_SIO2, 0];
        const G = cdiv(csub(ec, es), cadd(ec, cscale(es, 2)));
        ep = cdiv(cmul(es, cadd([r3, 0], cscale(G, 2))), csub([r3, 0], G));
        ep = [ep[0], ep[1] - 2 * p.K_s * 1e-9 / b / w];
      } else {
        ep = [p.eps_p * EPS0, -p.sigma_p * 1e-4 / w];
      }
      ReK[i] = cdiv(csub(ep, em), cadd(ep, cscale(em, 2)))[0];
      Omega[i] = w * p.Lambda * EPS_M / lam * ELL / sm;
      S[i] = Omega[i] ** 2 / (1 + Omega[i] ** 2);
      U[i] = U0 * ReK[i] * S[i];
      N_eq[i] = N_cap / (1 + N_cap / N_res * Math.exp(-U[i]));       // exp overflow -> Infinity -> N_eq = 0
      tau_l[i] = p.l_trap ** 2 / D / (1 + Math.max(-U[i], 0));
      const log_x = Math.log(J_D * tau_l[i] / N_res) - U[i];          // log(k_eq tau_l)
      g[i] = 0.5 * (1 - Math.tanh(log_x / 2));
      k_out[i] = 0.5 * (1 + Math.tanh(log_x / 2)) / tau_l[i];
    }
    return { f, ReK, S, Omega, U, U0, D, N_cap, N_res, J_D, N_eq, tau_l, g, k_out, lam, f_c: sm / (2 * Math.PI * EPS_M) };
  }

  function sweep(ph, direction, t_step, n_out = 40) {
    const n = ph.f.length;
    const idx = [...Array(n).keys()].sort((i, j) => ph.f[i] - ph.f[j]);
    const order = direction === "down" ? idx.slice().reverse() : idx;
    let N = ph.N_res, t0 = 0;
    const t = [0], Nt = [N], endByIndex = new Array(n);
    for (const i of order) {
      const A = ph.g[i] * ph.J_D;
      const B = A / ph.N_cap + ph.k_out[i];
      const N_inf = A / B;
      const N_start = N;
      for (let k = 1; k <= n_out; k++) {
        const s = t_step * k / n_out;
        N = N_inf + (N_start - N_inf) * Math.exp(-B * s);
        t.push(t0 + s); Nt.push(N);
      }
      t0 += t_step;
      endByIndex[i] = N;
    }
    return { t, N: Nt, f: idx.map(i => ph.f[i]), end: idx.map(i => endByIndex[i]) };
  }

  function run(overrides) {
    const p = Object.assign({}, DEFAULTS, overrides || {});
    const f = logspace(Math.log10(p.f_min), Math.log10(p.f_max), Math.round(p.n_f));
    const ph = physics(f, p);
    return { p, ph, sw: { up: sweep(ph, "up", p.t_step), down: sweep(ph, "down", p.t_step) } };
  }

  // area = integral |N_down - N_up| d log10 f / max N (decades); max_gap = max |N_down - N_up| / max N
  function hysteresis(sw) {
    const f = sw.up.f, up = sw.up.end, down = sw.down.end;
    const top = Math.max(...up, ...down, 1e-300);
    let area = 0, iMax = 0;
    const gap = up.map((u, i) => Math.abs(down[i] - u));
    for (let i = 0; i < gap.length; i++) {
      if (gap[i] > gap[iMax]) iMax = i;
      if (i > 0) area += 0.5 * (gap[i] + gap[i - 1]) * (Math.log10(f[i]) - Math.log10(f[i - 1]));
    }
    // signed area: > 0 when the down sweep holds more particles than the up sweep
    let signed = 0;
    for (let i = 1; i < f.length; i++)
      signed += 0.5 * ((down[i] - up[i]) + (down[i - 1] - up[i - 1])) * (Math.log10(f[i]) - Math.log10(f[i - 1]));
    return { area: area / top, signed: signed / top, max_gap: gap[iMax] / top, f_max_gap: f[iMax], top,
             max_gap_abs: gap[iMax] };
  }

  function crossovers(ph) {
    const out = [];
    for (let i = 0; i + 1 < ph.f.length; i++)
      if (Math.sign(ph.ReK[i]) !== Math.sign(ph.ReK[i + 1])) out.push(ph.f[i]);
    return out;
  }

  const api = { DEFAULTS, PRESETS, physics, sweep, run, hysteresis, crossovers, logspace };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TrapToy = api;
})(this);
