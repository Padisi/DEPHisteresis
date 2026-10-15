// Compare docs/model.js with the Python toy (tools/reference.json from tools/export_field.py).
//     node tools/check_port.js
const fs = require("fs"), path = require("path");
const T = require("../docs/model.js");
const dir = path.join(__dirname, "..", "docs", "data");
const meta = JSON.parse(fs.readFileSync(path.join(dir, "field.json")));
const buf = fs.readFileSync(path.join(dir, "field.bin"));
const field = T.makeField(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "reference.json")));
const rel = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i]) / (Math.abs(b[i]) + 1e-9)));
let worst = 0;
for (const r of ref) {
  const m = T.run(field, r.params);
  const e = { Ieq: rel(m.pre.Ieq, r.Ieq), Umax: rel(m.pre.Umax, r.Umax) };
  for (const d of ["up", "down"]) {
    e[d + "_end"] = rel(m[d].end, r[d].end);
    e[d + "_raw"] = rel(m[d].raw_end, r[d].raw_end);
    e[d + "_I"] = rel(m[d].I, r[d].I);
  }
  const w = Math.max(...Object.values(e));
  worst = Math.max(worst, w);
  console.log(JSON.stringify(r.params), "max rel err", w.toExponential(2), "rms vs draft", m.rms.toFixed(4));
}
console.log(worst < 1e-4 ? "OK" : "MISMATCH", "worst", worst.toExponential(2));
process.exit(worst < 1e-4 ? 0 : 1);
