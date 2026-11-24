// Ejecuta el modelo Python (fokker_planck.py + dep_sweep.py) con Pyodide en segundo plano.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js");

let pyodideReady = (async () => {
  postMessage({ type: "status", text: "Cargando Python (Pyodide, numpy, scipy)… la primera vez tarda unos segundos." });
  const py = await loadPyodide();
  await py.loadPackage(["numpy", "scipy"]);
  for (const name of ["fokker_planck.py", "dep_sweep.py"]) {
    const src = await (await fetch(name + "?v=" + Date.now())).text();
    py.FS.writeFile(name, src);
  }
  py.runPython("import warnings; warnings.filterwarnings('ignore'); import dep_sweep");
  postMessage({ type: "ready" });
  return py;
})();

onmessage = async (ev) => {
  const py = await pyodideReady;
  try {
    py.globals.set("params_js", py.toPy(ev.data.params));
    py.globals.set("progress", (label, k, n) => postMessage({ type: "progress", label, k, n }));
    const t0 = performance.now();
    const out = py.runPython("import json; json.dumps(dep_sweep.run(params_js, progress))");
    postMessage({ type: "result", result: JSON.parse(out), seconds: (performance.now() - t0) / 1000 });
  } catch (err) {
    const msg = String(err.message || err).trim().split("\n").filter(Boolean).pop();
    postMessage({ type: "error", text: msg });
  }
};
