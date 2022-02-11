WASM := wasm

check:
	cargo check --features force_naive
	cargo check --target=aarch64-apple-darwin
	cargo check --target=x86_64-apple-darwin
	export RUSTFLAGS="-Ctarget-feature=+avx"
	cargo check --target=x86_64-pc-windows-msvc
	export RUSTFLAGS="-Ctarget-feature=+simd128"
	cargo check --target=wasm32-unknown-unknown
	env -u RUSTFLAGS

bench: 
	cargo bench --all --features random

wasm:
	cd wasm-export && wasm-pack build --target nodejs --out-dir ../${WASM}/

publish:
	cd llml_simd_proc && cargo check
	cd llml_simd_proc && cargo test --all --all-features
	cd llml_simd_proc && cargo publish
	make publish-straight

publish-straight:
	make check
	cargo test --all --all-features
	cargo test --all --features random serialize
	cargo publish
	make publish-wasm

publish-wasm:
	cd wasm-export && export RUSTFLAGS="-Ctarget-feature=+simd128" && cargo check --target wasm32-unknown-unknown
	cd wasm-export && export RUSTFLAGS="-Ctarget-feature=+simd128" && wasm-pack build --target nodejs --out-dir ../${WASM}/
	cp README.md ${WASM}/README.md