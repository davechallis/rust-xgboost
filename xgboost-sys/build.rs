extern crate bindgen;

use std::process::Command;
use std::env;
use std::path::PathBuf;
use std::fs::canonicalize;

fn main() {
    let xgb_root = canonicalize("xgboost").unwrap();

    // TODO: allow for dynamic/static linking
    // TODO: check whether rabit should be built/linked
    if !xgb_root.join("lib").exists() {
        // TODO: better checks for build completion, currently xgboost's build script can run
        // `make clean_all` if openmp build fails
        Command::new(xgb_root.join("build.sh"))
            .current_dir(&xgb_root)
            .status()
            .expect("Failed to execute XGBoost build.sh script.");
    }

     let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", xgb_root.join("include").display()))
        .clang_arg(format!("-I{}", xgb_root.join("rabit/include").display()))
        .generate()
        .expect("Unable to generate bindings.");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    println!("cargo:rustc-flags=-l dylib=c++");
    println!("cargo:rustc-link-search={}", xgb_root.join("lib").display());
    println!("cargo:rustc-link-search={}", xgb_root.join("rabit/lib").display());
    println!("cargo:rustc-link-search={}", xgb_root.join("dmlc-core").display());
    println!("cargo:rustc-link-lib=static=rabit_empty");
    println!("cargo:rustc-link-lib=static=dmlc");
    println!("cargo:rustc-link-lib=static=xgboost");
}
