use log;
use dmatrix::DMatrix;
use booster::Booster;
use parameters::Parameters;
use super::XGBResult;
use xgboost_sys;

pub fn train(
    params: Parameters,
    dtrain: &DMatrix,
    num_boost_round: u32,
    evals: Option<&[&DMatrix]>,
    xgb_model: Option<Booster>
) -> XGBResult<()> {
    let mats = match evals {
        Some(eval_mats) => {
            let mut mats = vec![dtrain];
            mats.extend_from_slice(eval_mats);
            mats
        },
        None => vec![dtrain],
    };

    let mut bst = Booster::create(&mats, &params)?;
    let num_parallel_tree = 1;

    // load distributed code checkpoint from rabit
    let mut version = bst.load_rabit_checkpoint()?;
    debug!("Loaded Rabit checkpoint: version={}", version);
    assert!(unsafe { xgboost_sys::RabitGetWorldSize() != 1 || version == 0 });

    let rank = unsafe { xgboost_sys::RabitGetRank() };
    let start_iteration = version / 2;
    let mut nboost = start_iteration;

    for i in start_iteration..num_boost_round as i32 {
        // distributed code: need to resume to this point
        // skip first update if a recovery step
        if version % 2 == 0 {
            bst.update(&dtrain, i)?;
            bst.save_rabit_checkpoint();
        }

        assert!(unsafe { xgboost_sys::RabitGetWorldSize() == 1 || version == xgboost_sys::RabitVersionNumber() });

        nboost += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use parameters::{self, learning, tree};

    #[test]
    fn training() {
        let dmat_train = DMatrix::create_from_file("xgboost-sys/xgboost/demo/data/agaricus.txt.train", true).unwrap();
        let dmat_test = DMatrix::create_from_file("xgboost-sys/xgboost/demo/data/agaricus.txt.test", true).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![parameters::learning::EvaluationMetric::LogLoss]))
            .build()
            .unwrap();
        let params = parameters::ParametersBuilder::default()
            .booster_params(parameters::booster::BoosterParameters::GbTree(tree_params))
            .learning_params(learning_params)
            .silent(true)
            .build()
            .unwrap();

        train(params, &dmat_train, 10, None, None);
    }
}
