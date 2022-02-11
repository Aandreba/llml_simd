flat_mod!(float);

cfg_if::cfg_if! {
    if #[cfg(feature = "random")] {
        flat_mod!(random);
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "serialize")] {
        flat_mod!(serialize);
    }
}