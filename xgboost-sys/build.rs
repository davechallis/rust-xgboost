extern crate bindgen;

use std::process::Command;
use std::env;
use std::path::PathBuf;

fn main() {
    // TODO: allow for dynamic/static linking
    // TODO: check whether rabit should be built/linked
    // TODO: check whether build already completed
    Command::new("./build.sh")
        .current_dir("./vendor")
        .status()
        .expect("Failed to execute XGBoost build.sh script.");

     let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-Ivendor/include")
        .clang_arg("-Ivendor/rabit/include")
        .generate()
        .expect("Unable to generate bindings.");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    println!("cargo:rustc-flags=-l dylib=c++");
    println!("cargo:rustc-link-search=vendor/lib");
    println!("cargo:rustc-link-search=vendor/rabit/lib");
    println!("cargo:rustc-link-lib=static=rabit_empty");
    println!("cargo:rustc-link-lib=static=xgboost");
}
