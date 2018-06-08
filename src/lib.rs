#[macro_use]
extern crate derive_builder;
#[macro_use]
extern crate log;
#[macro_use]
extern crate ndarray;
extern crate xgboost_sys;
extern crate libc;

#[macro_use]
macro_rules! xgb_call {
    ($x:expr) => {
        XGBError::check_return_value(unsafe { $x })
    };
}

mod error;
use error::{XGBResult, XGBError};

mod utils;

pub mod dmatrix;
pub mod booster;
pub mod parameters;
