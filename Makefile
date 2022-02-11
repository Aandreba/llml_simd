WASM := ../wasm/

check:
	cargo check --target=aarch64-apple-darwin
	cargo check --target=x86_64-apple-darwin
	set RUSTFLAGS="-C --target-feature=+avx"
	cargo check --target=x86_64-pc-windows-msvc
	env -u RUSTFLAGS
	cargo check --target=wasm32-unknown-unknown
	set RUSTFLAGS="-C --target-feature=+simd128"
	cargo check --target=wasm32-unknown-unknown
	env -u RUSTFLAGS

bench: 
	cargo bench --all --features random

wasm:
	cd wasm-export && wasm-pack build --target nodejs --out-dir ${WASM}

publish:
	cd llml_simd_proc && cargo check
	cd llml_simd_proc && cargo test --all --all-features
	cd llml_simd_proc && cargo publish
	make publish-straight

publish-straight:
	make check
	cargo test --all --all-features
	cargo publish
	cd wasm-export && wasm-pack build --target nodejs --out-dir ${WASM}
	cp README.md ${WASM}/README.md
	cd ${WASM} && npm publish