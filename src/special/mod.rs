flat_mod!(floatext);

cfg_if::cfg_if! {
    if #[cfg(not(all(feature = "use_avx", target_feature = "avx")))] {
        flat_mod!(simd128);
    }
}