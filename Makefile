WASM := ./wasm/

bench: 
	cargo bench --all --features random

wasm:
	cd wasm-export
	wasm-pack build --target nodejs --out-dir ${WASM} -- --features wasm_dylib
	mv -r ./wasm ../wasm