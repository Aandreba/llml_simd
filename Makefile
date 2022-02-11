WASM := ../wasm/

bench: 
	cargo bench --all --features random

wasm:
	cd wasm-export && wasm-pack build --target nodejs --out-dir ${WASM}

publish:
	cargo test --all --all-features
	cd llml_simd_proc && cargo test --all --all-features
	cd llml_simd_proc && cargo publish
	cargo publish
	cd wasm-export && wasm-pack built --target nodejs --out-dir ${WASM}
	cp README.md ${WASM}/README.md
	cd ${WASM} && npm publish