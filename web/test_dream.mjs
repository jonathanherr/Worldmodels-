// Parity test: the JS dream step must match PyTorch's reference step in model.json.
// Usage: node web/test_dream.mjs web
import { readFileSync } from "node:fs";
import { DreamModel } from "./dream.js";

const dir = process.argv[2] ?? "web";
const meta = JSON.parse(readFileSync(`${dir}/model.json`, "utf8"));
const buf = readFileSync(`${dir}/weights.bin`);
const model = new DreamModel(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const ref = meta.reference;
const s0 = meta.starts[0];

const t0 = performance.now();
const next = model.step({ h: Float32Array.from(s0.h), z: Float32Array.from(s0.z) }, ref.action);
const frame = model.decode(next.z);
const ms = performance.now() - t0;

const maxDiff = (a, b) => Math.max(...b.map((v, i) => Math.abs(a[i] - v)));
const sample = Array.from({ length: ref.frame_sample.length }, (_, k) => frame[k * 97]);
const report = {
  h: maxDiff(next.h, ref.h),
  z: maxDiff(next.z, ref.z),
  frame: maxDiff(sample, ref.frame_sample),
  reward: maxDiff(next.rewardLogits, ref.reward_logits),
  continue: Math.abs(Math.log(next.pContinue / (1 - next.pContinue)) - ref.continue_logit),
};
console.log("max abs differences:", report, `step+decode ${ms.toFixed(1)} ms`);
const ok = Object.values(report).every((d) => d < 1e-3);
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
