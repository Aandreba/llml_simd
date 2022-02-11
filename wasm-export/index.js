const llml = require("./llml_simd")
const f32x16 = llml.f32x16;

llml.start();
let alpha = f32x16.random().sadd(3)
let beta = f32x16.random().sadd(2)

let dot = alpha.mul(beta).sum();
console.log(alpha, beta.toArray(), dot);