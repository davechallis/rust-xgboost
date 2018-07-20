//! Rust wrapper around the [XGBoost](https://xgboost.readthedocs.io/en/latest/) machine learning library.
//!
//! # Basic usage example
//!
//! ```
//! extern crate xgboost;
//!
//! use xgboost::{parameters, dmatrix::DMatrix, booster::Booster};
//!
//! fn main() {
//!     // training matrix with 5 training examples and 3 features
//!     let x_train = &[1.0, 1.0, 1.0,
//!                     1.0, 1.0, 0.0,
//!                     1.0, 1.0, 1.0,
//!                     0.0, 0.0, 0.0,
//!                     1.0, 1.0, 1.0];
//!     let x_train_num_rows = 5;
//!     let y_train = &[1.0, 1.0, 1.0, 0.0, 1.0];
//!
//!     // convert training data into XGBoost's matrix format, and set ground truth labels
//!     let mut dtrain = DMatrix::from_dense(x_train, x_train_num_rows).unwrap();
//!     dtrain.set_labels(y_train).unwrap();
//!
//!     let x_test = &[0.7, 0.9, 0.6];
//!     let x_test_num_rows = 1;
//!     let y_test = &[1.0];
//!     let mut dtest = DMatrix::from_dense(x_test, x_test_num_rows).unwrap();
//!     dtest.set_labels(y_test).unwrap();
//!
//!     // build overall training parameters
//!     let params = parameters::ParametersBuilder::default().build().unwrap();
//!
//!     // specify datasets to evaluate against during training
//!     let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];
//!
//!     // train model, and print evaluation data
//!     let bst = Booster::train(&params, &dtrain, 3, evaluation_sets).unwrap();
//!
//!     println!("{:?}", bst.predict(&dtest).unwrap());
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
