macro_rules! arch_use {
    () => {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "arm")] {
                use std::arch::arm::*;
            } else {
                use std::arch::aarch64::*;
            }
        }
    };
}

flat_mod!(straight, composite);