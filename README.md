# DEPHisteresis

Interactive playground of the DEP trap toy model for Fig. 2 of the draft (200 nm PS beads), with ACEO on/off.

**Playground:** https://padisi.github.io/DEPHisteresis/

It is the Sliders section of `chang_cooper/toy/fig2_ps_playground.ipynb` in the browser: the same seven sliders
(sigma_m, sigma_p, radius, b_eff, Lambda, V0_, h_ch) and one plot, I_Fluo (norm.) of the up and down sweeps against
the draft. It adds an ACEO switch, the supply gain 1 + A (1 MHz / f)² of the weak-flow scenario in `toy/fig2_ps.py`.
With ACEO off (A = 0) the result is exactly the notebook's.

## Files

| path | content |
|---|---|
| `docs/index.html` | the page (published to the `gh-pages` branch by `.github/workflows/pages.yml`) |
| `docs/model.js` | the toy in JavaScript: `precompute` and `sweep` of `toy/toy_trap.py`, `run` of the notebook |
| `docs/data/field.{bin,json}` | the DEP depth per unit Re K from `toy/field_table.npz`, on the ROI cells and the max-depth cells |
| `tools/export_field.py` | regenerates `docs/data` and `tools/reference.json` from the chang_cooper repo |
| `tools/check_port.js` | checks `docs/model.js` against the Python toy (agreement ~1e-10) |

    PYTHONPATH=.. python tools/export_field.py     # this repo sits inside chang_cooper/
    node tools/check_port.js
