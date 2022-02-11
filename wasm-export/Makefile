hello:
	export RUSTFLAGS="-Ctarget-feature=+simd128" && wasm-pack build --target nodejs --out-dir ./llml_simd/

test:
	make hello
	node .