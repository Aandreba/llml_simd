macro_rules! arch_use {
    () => {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "arm")] {
                use core::arch::arm::*;
            } else {
                use core::arch::aarch64::*;
            }
        }
    };
}

flat_mod!(straight, composite);