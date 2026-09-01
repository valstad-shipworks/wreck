//! Build-time MIR analysis of this crate's `#[kernel]`s — the whole pipeline lives in `hydroplane-auto`.
fn main() {
    hydroplane_auto::build_script();
    quad_load_cfg();
}

/// Sets `has_quad_load` on targets with a four-lane deinterleaving load of sixteen consecutive
/// floats, which the ray-vs-polytope plane walk uses to read four planes per step.
fn quad_load_cfg() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let available = match arch.as_str() {
        "aarch64" => true,
        "x86" | "x86_64" => features.split(',').any(|f| f == "sse2"),
        _ => false,
    };
    if available {
        println!("cargo::rustc-cfg=has_quad_load");
    }
}
