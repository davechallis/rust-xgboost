extern crate xgboost;

use xgboost::{parameters, dmatrix::DMatrix, booster::Booster};

fn main() {
    // Load train and test matrices from text files (in LibSVM format).
    println!("Loading train and test matrices...");
    let dtrain = DMatrix::load("../../xgboost-sys/xgboost/demo/data/agaricus.txt.train", false).unwrap();
    let dtest = DMatrix::load("../../xgboost-sys/xgboost/demo/data/agaricus.txt.test", false).unwrap();

    // Configure booster to use tree model, and configure tree parameters.
    let booster_params = parameters::booster::BoosterParameters::GbTree(
        parameters::tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build().unwrap()
    );

    // Configure objectives, metrics, etc.
    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::BinaryLogistic)
        .build().unwrap();

    // Overall configuration for XGBoost.
    let params = parameters::ParametersBuilder::default()
        .booster_params(booster_params)
        .learning_params(learning_params)
        .silent(false)
        .build().unwrap();

    // Specify datasets to evaluate against during training.
    let evaluation_sets = [(&dtest, "test"), (&dtrain, "train")];

    // Number of boosting rounds to run during training.
    let num_round = 2;

    // Train booster model, and print evaluation metrics.
    println!("\nTraining tree booster...");
    let bst = Booster::train(&params, &dtrain, num_round, &evaluation_sets).unwrap();

    // Get predictions probabilities for given matrix (as ndarray::Array1).
    let preds = bst.predict(&dtest).unwrap();

    // Get predicted labels for each test example (i.e. 0 or 1).
    println!("\nChecking predictions...");
    let labels = dtest.get_labels().unwrap();
    println!("First 3 predicated labels: {} {} {}", labels[0], labels[1], labels[2]);

    // Print error rate.
    let num_correct: usize = preds.iter()
        .map(|&v| if v > 0.5 { 1 } else { 0 })
        .sum();
    println!("error={} ({}/{} correct)", num_correct as f32 / preds.len() as f32, num_correct, preds.len());

    // Save and load model file.
    println!("\nSaving and loading Booster model...");
    bst.save("xgb.model").unwrap();
    let bst = Booster::load("xgb.model").unwrap();
    let preds2 = bst.predict(&dtest).unwrap();
    assert_eq!(preds, preds2);

    // Save and load data matrix file.
    println!("\nSaving and loading matrix data...");
    dtest.save("test.dmat", false).unwrap();
    let dtest2 = DMatrix::load("test.dmat", false).unwrap();
    assert_eq!(bst.predict(&dtest2).unwrap(), preds);

    // Error handling example.
    println!("\nError message example...");
    let result = Booster::load("/does/not/exist");
    match result {
        Ok(_bst) => (),
        Err(err) => println!("Failed to load booster: {}", err),
    }


}
