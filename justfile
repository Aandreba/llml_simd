bench:
    cargo bench --all --features random
    
coverage target:
    cargo build --target {{target}}
    cargo test --target {{target}}
    grcov . -s . --binary-path ./target/debug/ -t files --branch --ignore-not-existing -o ./lcov.txt