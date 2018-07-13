//! Rust wrapper around the [XGBoost](https://xgboost.readthedocs.io/en/latest/) machine learning library.
//!
//! # Basic usage example
//!
//! ```
//! extern crate xgboost;
//! extern crate ndarray;
//!
//! use ndarray::arr2;
//!
//! use xgboost::{parameters, dmatrix::DMatrix, booster::Booster};
//!
//! fn main() {
//!     // training matrix with 5 training examples and 3 features
//!     let x_train: &[[f32; 3]] = &[[1.0, 1.0, 1.0],
//!                                  [1.0, 1.0, 0.0],
//!                                  [1.0, 1.0, 1.0],
//!                                  [0.0, 0.0, 0.0],
//!                                  [1.0, 1.0, 1.0]];
//!     let y_train: &[f32] = &[1.0, 1.0, 1.0, 0.0, 1.0];
//!
//!     // convert training data into XGBoost's matrix format, and set ground truth labels
//!     let mut dtrain = DMatrix::from_dense(&arr2(x_train)).unwrap();
//!     dtrain.set_labels(y_train).unwrap();
//!
//!     let x_test: &[[f32; 3]] = &[[0.7, 0.9, 0.6]];
//!     let y_test: &[f32] = &[1.0];
//!     let mut dtest = DMatrix::from_dense(&arr2(x_test)).unwrap();
//!     dtest.set_labels(y_test).unwrap();
//!
//!     // build overall training parameters
//!     let params = parameters::ParametersBuilder::default().build().unwrap();
//!
//!     // specify datasets to evaluate against during training
//!     let evaluation_sets = [(&dtrain, "train"), (&dtest, "test")];
//!
//!     // train model, and print evaluation data
//!     let bst = Booster::train(&params, &dtrain, 3, &evaluation_sets).unwrap();
//!
//!     println!("{}", bst.predict(&dtest).unwrap());
//! }
//! ```
//!
//! # Status
//!
//! The crate is still in the early stages of development, so the API is likely to be fairly
//! unstable.
#[macro_use]
extern crate derive_builder;
#[macro_use]
extern crate log;
#[cfg_attr(test, macro_use)]
extern crate ndarray;
extern crate xgboost_sys;
extern crate libc;
extern crate tempfile;

#[macro_use]
macro_rules! xgb_call {
    ($x:expr) => {
        XGBError::check_return_value(unsafe { $x })
    };
}

mod error;
use error::{XGBResult, XGBError};

pub mod dmatrix;
pub mod booster;
pub mod parameters;
