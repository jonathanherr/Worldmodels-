// Runs the world model's dream loop in plain JavaScript (no ML library needed).
// Mirrors wm/model.py: GRUCell -> prior MLP -> layer norm -> decoder, plus the two heads.
// Works in the browser and in Node (for the parity test in test_dream.mjs).

const silu = (x) => x / (1 + Math.exp(-x));
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export class DreamModel {
  constructor(meta, buffer) {
    this.meta = meta;
    const all = new Float32Array(buffer);
    this.t = {};
    for (const [name, info] of Object.entries(meta.tensors)) {
      this.t[name] = { data: all.subarray(info.offset, info.offset + info.length), shape: info.shape };
    }
  }

  // y = act(W x + b), with W and b read from `${name}.${wKey}` and `${name}.${bKey}`.
  linear(name, x, act, wKey = "weight", bKey = "bias") {
    const W = this.t[`${name}.${wKey}`], b = this.t[`${name}.${bKey}`].data;
    const [out, inp] = W.shape, w = W.data, y = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let s = b[o];
      const row = o * inp;
      for (let i = 0; i < inp; i++) s += w[row + i] * x[i];
      y[o] = act ? act(s) : s;
    }
    return y;
  }

  mlp(name, x) {
    return this.linear(`${name}.2`, this.linear(`${name}.0`, x, silu));
  }

  layerNorm(x) {
    let mean = 0, v = 0;
    for (const a of x) mean += a;
    mean /= x.length;
    for (const a of x) v += (a - mean) ** 2;
    const inv = 1 / Math.sqrt(v / x.length + 1e-5);
    return x.map((a) => (a - mean) * inv);
  }

  gru(h, x) {
    const H = this.meta.hidden;
    const gi = this.linear("gru", x, null, "weight_ih", "bias_ih");
    const gh = this.linear("gru", h, null, "weight_hh", "bias_hh");
    const out = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      const r = sigmoid(gi[j] + gh[j]);
      const z = sigmoid(gi[H + j] + gh[H + j]);
      const n = Math.tanh(gi[2 * H + j] + r * gh[2 * H + j]);
      out[j] = (1 - z) * n + z * h[j];
    }
    return out;
  }

  // One step of imagination: returns the new state plus what the heads predict.
  step(state, action) {
    const x = new Float32Array(this.meta.latent + this.meta.num_actions);
    x.set(state.z);
    x[this.meta.latent + action] = 1;
    const h = this.gru(state.h, x);
    const z = this.layerNorm(this.mlp("prior", h));
    const rl = this.mlp("reward_head", h);
    const cont = sigmoid(this.mlp("continue_head", h)[0]);
    return { h, z, rewardLogits: rl, pContinue: cont };
  }

  // Decoder: latent -> 64x64xC image in [0, 1], height-width-channel order.
  decode(z) {
    const C = this.meta.decoder_width;
    let x = this.linear("decoder.fc", z, silu);  // 4x4xC, already HWC
    let size = 4;
    for (let k = 0; k < 4; k++) {
      x = this.deconv(`decoder.deconv${k}`, x, size, k < 3 ? silu : sigmoid);
      size *= 2;
    }
    return x;
  }

  // ConvTranspose2d(kernel 4, stride 2, padding 1) on an HWC tensor; weights are (in, 4, 4, out).
  deconv(name, inp, size, act) {
    const W = this.t[`${name}.weight`], b = this.t[`${name}.bias`].data;
    const [cin, , , cout] = W.shape, w = W.data, outSize = size * 2;
    const out = new Float32Array(outSize * outSize * cout);
    for (let p = 0; p < outSize * outSize; p++) out.set(b, p * cout);
    for (let iy = 0; iy < size; iy++) {
      for (let ix = 0; ix < size; ix++) {
        const ibase = (iy * size + ix) * cin;
        for (let ky = 0; ky < 4; ky++) {
          const oy = 2 * iy - 1 + ky;
          if (oy < 0 || oy >= outSize) continue;
          for (let kx = 0; kx < 4; kx++) {
            const ox = 2 * ix - 1 + kx;
            if (ox < 0 || ox >= outSize) continue;
            const obase = (oy * outSize + ox) * cout;
            for (let i = 0; i < cin; i++) {
              const v = inp[ibase + i];
              if (v === 0) continue;
              const wbase = ((i * 4 + ky) * 4 + kx) * cout;
              for (let o = 0; o < cout; o++) out[obase + o] += v * w[wbase + o];
            }
          }
        }
      }
    }
    for (let j = 0; j < out.length; j++) out[j] = act(out[j]);
    return out;
  }
}
