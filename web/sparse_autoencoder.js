// Inference for the sparse autoencoder; reads the model.bin written by `main.py export-web`.
// Everything is in -1..1. In sparse input, anything that isn't a number in that range counts as missing.

const relu = (x) => (x > 0 ? x : 0);

function dense({ w, b, rows, cols }, x, activation) {
  const y = new Float64Array(rows);
  for (let j = 0; j < rows; j++) {
    let sum = b[j];
    const row = j * cols;
    for (let i = 0; i < cols; i++) sum += w[row + i] * x[i];
    y[j] = activation(sum);
  }
  return y;
}

function checkArray(values, length, name) {
  if (!(Array.isArray(values) || ArrayBuffer.isView(values)) || values.length !== length) {
    throw new TypeError(`${name} must be an array of ${length} numbers`);
  }
}

export class SparseAutoencoder {
  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`failed to load ${url}: ${response.status}`);
    return new SparseAutoencoder(await response.arrayBuffer());
  }

  constructor(buffer) {
    if (!(buffer instanceof ArrayBuffer)) throw new TypeError("expected an ArrayBuffer");
    const headerLength = new DataView(buffer).getUint32(0, true);
    const shapes = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
    const arrays = {};
    let offset = 4 + headerLength;
    for (const [name, shape] of Object.entries(shapes)) {
      const size = shape.reduce((a, b) => a * b, 1);
      arrays[name] = new Float32Array(buffer, offset, size);
      offset += size * 4;
    }
    const layer = (name) => {
      const [rows, cols] = shapes[`${name}.weight`];
      return { w: arrays[`${name}.weight`], b: arrays[`${name}.bias`], rows, cols };
    };
    this.encoder = ["encoder.0", "encoder.2", "encoder.4"].map(layer);
    this.decoder = ["decoder.0", "decoder.2"].map(layer);
    this.fill = arrays.fill;
    this.features = this.fill.length;
    this.latentSize = this.encoder[2].rows;

    // A dropped feature adds the same amount every time, so only the kept features cost anything.
    const { w, b, rows, cols } = this.encoder[0];
    if (cols !== 2 * this.features) throw new Error("model file is inconsistent");
    this.valueWeights = new Float32Array(this.features * rows); // indexed [feature][hidden unit]
    this.maskWeights = new Float32Array(this.features * rows);
    this.baseline = Float64Array.from(b);
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < this.features; i++) {
        this.valueWeights[i * rows + j] = w[j * cols + i];
        this.maskWeights[i * rows + j] = w[j * cols + this.features + i];
        this.baseline[j] += w[j * cols + i] * this.fill[i];
      }
    }
  }

  front(sparse) {
    checkArray(sparse, this.features, "sparse");
    const hiddenSize = this.encoder[0].rows;
    const hidden = Float64Array.from(this.baseline);
    for (let i = 0; i < this.features; i++) {
      const v = sparse[i];
      if (!(typeof v === "number" && v >= -1 && v <= 1)) continue; // NaN fails the range test too
      const delta = v - this.fill[i];
      const at = i * hiddenSize;
      for (let j = 0; j < hiddenSize; j++) {
        hidden[j] += this.valueWeights[at + j] * delta + this.maskWeights[at + j];
      }
    }
    const x = dense(this.encoder[1], hidden.map(relu), relu);
    return Float32Array.from(dense(this.encoder[2], x, Math.tanh));
  }

  back(latent) {
    checkArray(latent, this.latentSize, "latent");
    if (!latent.every((v) => typeof v === "number" && v >= -1 && v <= 1)) {
      throw new RangeError("latent values must be numbers from -1 to 1");
    }
    const x = dense(this.decoder[0], latent, relu);
    return Float32Array.from(dense(this.decoder[1], x, Math.tanh));
  }

  full(sparse) {
    return this.back(this.front(sparse));
  }
}