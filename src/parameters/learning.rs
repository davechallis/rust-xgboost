use std::default::Default;
use dmatrix::DMatrix;

pub type CustomObjective = fn(&[f32], &DMatrix) -> (Vec<f32>, Vec<f32>);

pub enum Objective {
    RegLinear,
    RegLogistic,
    BinaryLogistic,
    BinaryLogitRaw,
    GpuRegLinear,
    GpuRegLogistic,
    GpuBinaryLogistic,
    GpuBinaryLogitRaw,
    CountPoisson,
    SurvivalCox,
    MultiSoftmax,
    MultiSoftprob,
    RankPairwise,
    RegGamma,
    RegTweedie,
    Custom(CustomObjective),
}

impl Copy for Objective {}

impl Clone for Objective {
    fn clone(&self) -> Self { *self }
}

impl ToString for Objective {
    fn to_string(&self) -> String {
        match *self {
            Objective::RegLinear => "reg:linear".to_owned(),
            Objective::RegLogistic => "reg:logistic".to_owned(),
            Objective::BinaryLogistic => "binary:logistic".to_owned(),
            Objective::BinaryLogitRaw => "binary:logitraw".to_owned(),
            Objective::GpuRegLinear => "gpu:reg:linear".to_owned(),
            Objective::GpuRegLogistic => "gpu:reg:logistic".to_owned(),
            Objective::GpuBinaryLogistic => "gpu:binary:logistic".to_owned(),
            Objective::GpuBinaryLogitRaw => "gpu:binary:logitraw".to_owned(),
            Objective::CountPoisson => "count:poisson".to_owned(),
            Objective::SurvivalCox => "survival:cox".to_owned(),
            Objective::MultiSoftmax => "multi:softmax".to_owned(),
            Objective::MultiSoftprob => "multi:softprob".to_owned(),
            Objective::RankPairwise => "rank:pairwise".to_owned(),
            Objective::RegGamma => "reg:gamma".to_owned(),
            Objective::RegTweedie => "reg:tweedie".to_owned(),
            Objective::Custom(_) => panic!("to_string should never be called for Custom"),
        }
    }
}

impl Default for Objective {
    fn default() -> Self { Objective::RegLinear }
}

/// Type of evaluation metrics to use during learning.
#[derive(Clone)]
pub enum Metrics {
    /// Automatically selects metrics based on learning objective.
    Auto,

    /// Use custom list of metrics.
    Custom(Vec<EvaluationMetric>),
}

type CustomEvaluationMetric = fn(&[f32], &DMatrix) -> f32;

#[derive(Clone)]
pub enum EvaluationMetric {
    /// Root Mean Square Error.
    RMSE,

    /// Mean Absolute Error.
    MAE,

    /// Negative log-likelihood.
    LogLoss,

    // TODO: use error as field if set to 0.5
    /// Binary classification error rate. It is calculated as #(wrong cases)/#(all cases).
    /// For the predictions, the evaluation will regard the instances with prediction value larger than
    /// given threshold as positive instances, and the others as negative instances.
    BinaryErrorRate(f32),

    /// Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
    MultiClassErrorRate,

    /// Multiclass logloss.
    MultiClassLogLoss,

    /// Area under the curve for ranking evaluation.
    AUC,

    /// Normalized Discounted Cumulative Gain.
    NDCG,

    /// NDCG with top N positions cut off.
    NDCGCut(u32),

    /// NDCG with scores of lists without any positive samples evaluated as 0 instead of 1.
    NDCGNegative,

    /// NDCG with scores of lists without any positive samples evaluated as 0 instead of 1, and top
    /// N positions cut off.
    NDCGCutNegative(u32),

    /// Mean average precision.
    MAP,

    /// MAP with top N positions cut off.
    MAPCut(u32),

    /// MAP with scores of lists without any positive samples evaluated as 0 instead of 1.
    MAPNegative,

    /// MAP with scores of lists without any positive samples evaluated as 0 instead of 1, and top
    /// N positions cut off.
    MAPCutNegative(u32),

    /// Negative log likelihood for Poisson regression.
    PoissonLogLoss,

    /// Negative log likelihood for Gamma regression.
    GammaLogLoss,

    /// Negative log likelihood for Cox proportional hazards regression.
    CoxLogLoss,

    /// Residual deviance for Gamma regression.
    GammaDeviance,

    /// Negative log likelihood for Tweedie regression (at a specified value of the tweedie_variance_power parameter).
    TweedieLogLoss,

    Custom(String, CustomEvaluationMetric),
}

impl ToString for EvaluationMetric {
    fn to_string(&self) -> String {
        match *self {
            EvaluationMetric::RMSE => "rmse".to_owned(),
            EvaluationMetric::MAE => "mae".to_owned(),
            EvaluationMetric::LogLoss => "logloss".to_owned(),
            EvaluationMetric::BinaryErrorRate(t) => {
                if t == 0.5 {
                    "error".to_owned()
                } else {
                    format!("error@{}", t)
                }
            },
            EvaluationMetric::MultiClassErrorRate => "merror".to_owned(),
            EvaluationMetric::MultiClassLogLoss   => "mlogloss".to_owned(),
            EvaluationMetric::AUC                 => "auc".to_owned(),
            EvaluationMetric::NDCG                => "ndcg".to_owned(),
            EvaluationMetric::NDCGCut(n)          => format!("ndcg@{}", n),
            EvaluationMetric::NDCGNegative        => "ndcg-".to_owned(),
            EvaluationMetric::NDCGCutNegative(n)  => format!("ndcg@{}-", n),
            EvaluationMetric::MAP                 => "map".to_owned(),
            EvaluationMetric::MAPCut(n)           => format!("map@{}", n),
            EvaluationMetric::MAPNegative         => "map-".to_owned(),
            EvaluationMetric::MAPCutNegative(n)   => format!("map@{}-", n),
            EvaluationMetric::PoissonLogLoss      => "poisson-nloglik".to_owned(),
            EvaluationMetric::GammaLogLoss        => "gamma-nloglik".to_owned(),
            EvaluationMetric::CoxLogLoss          => "cox-nloglik".to_owned(),
            EvaluationMetric::GammaDeviance       => "gamma-deviance".to_owned(),
            EvaluationMetric::TweedieLogLoss      => "tweedie-nloglik".to_owned(),
            EvaluationMetric::Custom(_, _)        => panic!("to_string should never be called for Custom"),
        }
    }
}

/// Parameters that configure the learning objective.
///
/// See [`LearningTaskParametersBuilder`](struct.LearningTaskParametersBuilder.html), for details
/// on parameters.
#[derive(Builder, Clone)]
#[builder(default)]
pub struct LearningTaskParameters {
    pub(crate) objective: Objective,
    base_score: f32,
    pub(crate) eval_metrics: Metrics,
    seed: u64,
}

impl Default for LearningTaskParameters {
    fn default() -> Self {
        LearningTaskParameters {
            objective: Objective::default(),
            base_score: 0.5,
            eval_metrics: Metrics::Auto,
            seed: 0,
        }
    }
}

impl LearningTaskParameters {
    pub(crate) fn as_string_pairs(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();

        match self.objective {
            Objective::Custom(_) => (),
            objective => v.push(("objective".to_owned(), objective.to_string())),
        }
        v.push(("base_score".to_owned(), self.base_score.to_string()));
        v.push(("seed".to_owned(), self.seed.to_string()));

        if let Metrics::Custom(eval_metrics) = &self.eval_metrics {
            for metric in eval_metrics {
                match metric {
                    EvaluationMetric::Custom(_, _) => (),
                    metric                         => v.push(("eval_metric".to_owned(), metric.to_string())),
                }
            }
        }

        v
    }
}
