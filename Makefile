WASM := wasm

check-apple:
	cargo check --features force_naive
	cargo check --target=aarch64-apple-darwin
	cargo check --target=x86_64-apple-darwin
	export RUSTFLAGS="-Ctarget-feature=+avx"
	cargo check --target=x86_64-pc-windows-msvc
	export RUSTFLAGS="-Ctarget-feature=+simd128"
	cargo check --target=wasm32-unknown-unknown
	env -u RUSTFLAGS

check-linux:
	cargo check --features force_naive
	cargo check --target=aarch64-unknown-linux-gnu
	cargo check --target=x86_64-unknown-linux-gnu
	export RUSTFLAGS="-Ctarget-feature=+avx"
	cargo check --target=x86_64-unknown-linux-gnu
	export RUSTFLAGS="-Ctarget-feature=+simd128"
	cargo check --target=wasm32-unknown-unknown
	env -u RUSTFLAGS

check-tests:
	cargo check --features force_naive --tests
	cargo check --target=aarch64-apple-darwin --tests
	cargo check --target=x86_64-apple-darwin --tests
	export RUSTFLAGS="-Ctarget-feature=+avx"
	cargo check --target=x86_64-pc-windows-msvc --tests
	export RUSTFLAGS="-Ctarget-feature=+simd128"
	cargo check --target=wasm32-unknown-unknown --tests
	env -u RUSTFLAGS

test-all:
	cargo test --all --features force_naive
	cargo test --target=aarch64-apple-darwin
	cargo test --target=x86_64-apple-darwin
	export RUSTFLAGS="-Ctarget-feature=+simd128"
	cargo test --target=wasm32-unknown-unknown
	export RUSTFLAGS="-Ctarget-feature=+avx"
	cargo test --target=x86_64-pc-windows-msvc
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
	make check-linux
	cargo test --all --all-features
	cargo test --all --features random serialize
	cargo publish
	make publish-wasm

publish-wasm:
	cd wasm-export && export RUSTFLAGS="-Ctarget-feature=+simd128" && cargo check --target wasm32-unknown-unknown
	cd wasm-export && export RUSTFLAGS="-Ctarget-feature=+simd128" && wasm-pack build --target nodejs --out-dir ../${WASM}/
	cp README.md ${WASM}/README.md