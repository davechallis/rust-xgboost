extern crate bindgen;

use std::process::Command;
use std::env;
use std::path::{PathBuf, Path};

fn main() {
    // TODO: allow for dynamic/static linking
    // TODO: check whether rabit should be built/linked
    if (!Path::new("xgboost/lib").exists()) {
        // TODO: better checks for build completion, currently xgboost's build script can run
        // `make clean_all` if openmp build fails
        Command::new("./build.sh")
            .current_dir("./xgboost")
            .status()
            .expect("Failed to execute XGBoost build.sh script.");
    }

     let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-Ixgboost/include")
        .clang_arg("-Ixgboost/rabit/include")
        .generate()
        .expect("Unable to generate bindings.");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    println!("cargo:rustc-flags=-l dylib=c++");
    println!("cargo:rustc-link-search=xgboost/lib");
    println!("cargo:rustc-link-search=xgboost/rabit/lib");
    println!("cargo:rustc-link-lib=static=rabit_empty");
    println!("cargo:rustc-link-lib=static=xgboost");
}
