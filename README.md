# rust-xgboost

[![Travis Build Status](https://travis-ci.com/davechallis/rust-xgboost.svg?branch=master)](https://travis-ci.com/davechallis/rust-xgboost)

Rust bindings for the [XGBoost](https://xgboost.ai) gradient boosting library.

Basic usage example:

```rust
extern crate xgboost;

use xgboost::{parameters, dmatrix::DMatrix, booster::Booster};

fn main() {
    // training matrix with 5 training examples and 3 features
    let x_train = &[1.0, 1.0, 1.0,
                    1.0, 1.0, 0.0,
                    1.0, 1.0, 1.0,
                    0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0];
    let num_rows = 5;
    let y_train = &[1.0, 1.0, 1.0, 0.0, 1.0];

    // convert training data into XGBoost's matrix format
    let mut dtrain = DMatrix::from_dense(x_train, num_rows).unwrap();

    // set ground truth labels for the training matrix
    dtrain.set_labels(y_train).unwrap();

    // test matrix with 1 row
    let x_test = &[0.7, 0.9, 0.6];
    let num_rows = 1;
    let y_test = &[1.0];
    let mut dtest = DMatrix::from_dense(x_test, num_rows).unwrap();
    dtest.set_labels(y_test).unwrap();

    // build overall training parameters
    let params = parameters::ParametersBuilder::default().build().unwrap();

    // specify datasets to evaluate against during training
    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // train model, and print evaluation data
    let bst = Booster::train(&params, &dtrain, 3, evaluation_sets).unwrap();

    println!("{:?}", bst.predict(&dtest).unwrap());
}
```

See the [examples](https://github.com/davechallis/rust-xgboost/tree/master/examples) directory for
demonstration of different features.

## Status

Currently in a very early stage of development, so the API shouldn't be considered stable yet.

Builds against XGBoost 0.72.

### Platforms

Tested:

* Linux
* Mac OS

Unsupported:

* Windows
