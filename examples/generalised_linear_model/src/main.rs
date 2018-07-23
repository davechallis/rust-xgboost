//! Example of how to fit a generalised linear model in XGBoost.

extern crate xgboost;
extern crate ndarray;
extern crate env_logger;

use xgboost::{parameters, DMatrix, Booster};

fn main() {
    // Initialise logging, run with e.g. RUST_LOG=xgboost=debug to see more details
    env_logger::init();

    // Load train and test matrices from text files (in LibSVM format)
    println!("Custom objective example...");
    let dtrain = DMatrix::load("../../xgboost-sys/xgboost/demo/data/agaricus.txt.train").unwrap();
    let dtest = DMatrix::load("../../xgboost-sys/xgboost/demo/data/agaricus.txt.test").unwrap();

    // Configure objectives, metrics, etc.
    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::BinaryLogistic)
        .build().unwrap();

    // Configure linear model parameters
    let linear_params = parameters::linear::LinearBoosterParametersBuilder::default()
        .alpha(0.0001)
        .lambda(1.0)
        .build().unwrap();

    // Configure booster to use a linear model
    let booster_params = parameters::booster::BoosterParameters::GbLinear(linear_params);

    // Overall configuration for XGBoost
    let params = parameters::ParametersBuilder::default()
        .learning_params(learning_params)
        .booster_params(booster_params)
        .silent(true)
        .build().unwrap();

    // Specify datasets to evaluate against during training
    let evaluation_sets = [(&dtest, "test"), (&dtrain, "train")];

    // Number of boosting rounds to run during training
    let num_round = 4;

    // Train booster model, and print evaluation metrics
    println!("\nTraining tree booster...");
    let bst = Booster::train(&params, &dtrain, num_round, &evaluation_sets).unwrap();

    // Get predictions probabilities for given matrix (as ndarray::Array1)
    let preds = bst.predict(&dtest).unwrap();

    // Get predicted labels for each test example (0.0 or 1.0 in this case)
    let labels = dtest.get_labels().unwrap();

    // Print error rate
    let mut num_errors = 0;
    for (pred, label) in preds.iter().zip(labels) {
        let pred = if *pred > 0.5 { 1.0 } else { 0.0 };
        if pred != *label {
            num_errors += 1;
        }
    }
    println!("error={} ({}/{} correct)",
             num_errors as f32 / preds.len() as f32, preds.len() - num_errors, preds.len());
}
