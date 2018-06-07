extern crate xgboost;

use xgboost::{parameters, dmatrix::DMatrix, booster::Booster};

fn main() {
    let dmat_train = DMatrix::create_from_file("../../xgboost-sys/xgboost/demo/data/agaricus.txt.train", true).unwrap();
    let dmat_test = DMatrix::create_from_file("../../xgboost-sys/xgboost/demo/data/agaricus.txt.test", true).unwrap();

    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
        .max_depth(2)
        .eta(1.0)
        .build().unwrap();
    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::BinaryLogistic)
        .eval_metrics(Some(vec![parameters::learning::EvaluationMetric::LogLoss,
                                parameters::learning::EvaluationMetric::RMSE]))
        .build().unwrap();
    let params = parameters::ParametersBuilder::default()
        .booster_params(parameters::booster::BoosterParameters::GbTree(tree_params))
        .learning_params(learning_params)
        .silent(true)
        .build().unwrap();

    let mut booster = Booster::create(&[&dmat_train, &dmat_test], &params).unwrap();

    for i in 0..100000 {
        booster.update(&dmat_train, i).unwrap();
        let eval = booster.eval_set(&[&dmat_train, &dmat_test], &["train", "test"], i).unwrap();
        println!("{}", eval);
    }
}
