use std::io::Read;
use std::path::PathBuf;

/// Download the quantized BGE-small-en-v1.5 ONNX model and its tokenizer
/// files into `<workspace>/models/` so they can be embedded at compile time
/// via `include_bytes!`.  Already-downloaded files are skipped.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Rerun if any model file is missing (cargo treats a missing
    // rerun-if-changed path as "always rerun").
    for file in &[
        "model.onnx",
        "tokenizer.json",
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ] {
        println!("cargo:rerun-if-changed=models/{file}");
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let model_dir = manifest_dir.join("models");
    std::fs::create_dir_all(&model_dir).expect("Failed to create models/ directory");

    // Qdrant's quantized export of BAAI/bge-small-en-v1.5 (~22 MB).
    let base = "https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q/resolve/main";

    let downloads: &[(&str, &str)] = &[
        ("model_optimized.onnx", "model.onnx"),
        ("tokenizer.json", "tokenizer.json"),
        ("config.json", "config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
    ];

    for (remote, local) in downloads {
        let dest = model_dir.join(local);
        if dest.exists() {
            continue;
        }

        let url = format!("{base}/{remote}");
        eprintln!("cargo: downloading {remote} …");

        let response = ureq::get(&url)
            .call()
            .unwrap_or_else(|e| panic!("Failed to download {url}: {e}"));

        let mut data = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut data)
            .expect("Failed to read download body");

        std::fs::write(&dest, &data)
            .unwrap_or_else(|e| panic!("Failed to write {}: {e}", dest.display()));

        eprintln!(
            "cargo: saved {} ({:.1} MB)",
            dest.display(),
            data.len() as f64 / 1_000_000.0
        );
    }

    // Bake the absolute model directory path into the binary so the runtime
    // knows where to load the model from without a network call.
    let abs = model_dir
        .canonicalize()
        .expect("Failed to resolve models/ path");
    println!("cargo:rustc-env=FLINT_MODELS_DIR={}", abs.display());
}
