![Crates.io](https://img.shields.io/crates/v/llml_simd)
![NPM](https://img.shields.io/npm/v/llml_simd)
![Rust docs](https://img.shields.io/docsrs/llml_simd)

# LLML's SIMD
SIMD (**S**ingle **I**nstruction **M**ultiple **D**ata) extension for a variety of targets.
This project was initially started to facilitate the expansion of [LLML](https://github.com/Aandreba/llml)'s suported targets & features

## Contents ##
This crate/library contains bindings to native-level SIMD instructions (alongside some polyfill) for SSE, AVX (see [AVX support](##AVX-Support)), NEON & WASM. It also contains naive implementations of all data-types (see [naive implementation](##Naive-implementation))

## No std ##
```llml_simd``` is a no_std crate. This means that it can be used for embeded systems projects seamlessly.

## Naive implementation ##
If no supported target is detected (or if you enable the feature ```force_naive```), Rust will compile the the SIMD vectors in naive mode. This mode represent's all data types as an array, executing most of the methods via iterators. This mode, while **not recommended**, is usefull if you intend to share code with some other programm that doesn't have SIMD support.

> **Warning**\
> While not explicitly SIMD, ```rustc``` might still optimize some parts of the code to utilize SIMD instructions if it can and you allow it to.
> If you want to fully disable SIMD instructions, use ```--target-feature=-sse``` on x86/x86_64 and ```--target-feature=-neon``` on arm/aarch64 (naive mode will be used automatically in those cases, not requiring to enable ```force_naive```)

## AVX Support ##
If Rust detects ```avx``` as a target feature **and** you have the ```use_avx``` feature enabled (see [features](##Features)), ```llml_simd``` will compile all vectors over 128-bit long with AVX instructions, increasing performance significantly.

## JavaScript Library ##
Thanks to WASM, ```llml_simd``` is available for JavaScript/TypeScript via npm.\
You can install it into your Node project with ```npm i llml_simd```

## Features ##
| Feature           | Description                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| ```force_naive``` | Forces naive types (see [Naive implementation](##Naive-implementation))                             |
| ```use_avx```     | Enables the use of AVX SIMD types (see [AVX support](##AVX-Support))                                |
| ```random```      | Enables random generation of vectors via [rand](https://github.com/rust-random/rand)                |
| ```serialize```   | Enables serialization and deserialization of vectors via [serde](https://github.com/serde-rs/serde) |

## Examples ##
### Dot product (Rust) ###
```rust
use llml_simd::float::single::f32x4;

pub fn main() {
    let alpha = f32x4::new([1., 2., 3., 4.]);
    let beta = f32x4::new([5., 6., 7., 8.]);

    let dot = (alpha * beta).sum();
    assert_eq!(dot, 70.);
}
```

### Dot product (JavaScript/TypeScript) ###
```typescript
import { f32x4 } from llml_simd

let alpha = new f32x4(new Float32Array([1, 2, 3, 4]))
let beta = new f32x4(new Float32Array([5, 6, 7, 8]))

let dot = alpha.mul(beta).sum()
console.assert(dot === 70)
```