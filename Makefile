WASM := ../wasm/

bench: 
	cargo bench --all --features random

wasm:
	cd wasm-export && wasm-pack build --target nodejs --out-dir ${WASM}

publish:
	cargo test --all --all-features
	cargo publish
	cd wasm-export && wasm-pack publish --access=public