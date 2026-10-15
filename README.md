# DEPHisteresis

Toy model of nanoparticle trapping in a quadrupole dielectrophoresis (DEP) trap, and of the hysteresis between
up and down frequency sweeps.

**Playground:** https://padisi.github.io/DEPHisteresis/ (sliders for every parameter; it shows the up/down sweeps,
the hysteresis measures and the model notes, and exports the curves as CSV).

The toy reduces the 3D Fokker–Planck (Smoluchowski) simulations of the trap, which use a Chang–Cooper scheme and take
1–2 h per sweep on a GPU, to one equation for the number of trapped particles:

    dN/dt = g J_D [ (1 - N/N_cap) - (N/N_res) exp(-U(f)) ],    g = 1 / (1 + k_eq tau_l)

* `J_D = 4 D c_inf a`: diffusion-limited capture
* `U(f) = U0 Re K(f) S(f)`: trap depth (Clausius–Mossotti factor x electrode screening)
* `N_cap = phi_max V_t / V_NP`: packing capacity, and `N_res = c_inf V_t`: content without a trap
* `tau_l = l_t^2 / (D (1 + max(-U, 0)))`: minimum time to leave the trap

Each frequency step is solved exactly. The page explains the model in full.

## Files

| path | content |
|---|---|
| `toy/trap_toy.py` | Python reference implementation and the hysteresis measures (`python -m toy.trap_toy`) |
| `toy/trap_diffusion_toy.ipynb` | Jupyter notebook with ipywidgets sliders |
| `docs/model.js` | the same model in JavaScript (matches the Python to round-off) |
| `docs/index.html` | the GitHub Pages playground |

## Hysteresis measures

With `N_up(f)` and `N_down(f)` the content at the end of each step, and `N_max` the largest of either:

* **area** = ∫ |N_down − N_up| d log10 f / N_max (in decades)
* **signed area**: the same with N_down − N_up (> 0 when the down sweep holds more particles)
* **max gap** = max |N_down − N_up| / N_max, and the frequency where it happens
