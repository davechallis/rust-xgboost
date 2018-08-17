//! Builders for parameters that control various aspects of training.
use std::default::Default;
use std::fmt::{self, Display};

pub mod tree;
pub mod learning;
pub mod linear;
pub mod dart;
mod booster;

use super::DMatrix;
pub use self::booster::BoosterType;
use super::booster::CustomObjective;

/// Parameters for training boosters.
/// Created using [`BoosterParametersBuilder`](struct.BoosterParametersBuilder.html).
#[derive(Builder, Clone)]
#[builder(default)]
pub struct BoosterParameters {
    /// Type of booster (tree, linear or DART) along with its parameters.
    ///
    /// *default*: [`GbTree`](enum.BoosterType.html#variant.GbTree)
    booster_type: booster::BoosterType,

    /// Configuration for the learning objective.
    pub(crate) learning_params: learning::LearningTaskParameters,

    /// Whether to print XGBoost's C library's messages or not.
    ///
    /// *default*: `false`
    verbose: bool,

    /// Number of parallel threads XGboost will use (if compiled with multiprocessing support).
    ///
    /// *default*: `None` (XGBoost will automatically determing max threads to use)
    threads: Option<u32>,
}

impl Default for BoosterParameters {
    fn default() -> Self {
        BoosterParameters {
            booster_type: booster::BoosterType::default(),
            learning_params: learning::LearningTaskParameters::default(),
            verbose: false,
            threads: None,
        }
    }
}

impl BoosterParameters {
    /// Get type of booster (tree, linear or DART) along with its parameters.
    pub fn booster_type(&self) -> &booster::BoosterType {
        &self.booster_type
    }

    /// Set type of booster (tree, linear or DART) along with its parameters.
    pub fn set_booster_type<T: Into<booster::BoosterType>>(&mut self, booster_type: T) {
        self.booster_type = booster_type.into();
    }

    /// Get configuration for the learning objective.
    pub fn learning_params(&self) -> &learning::LearningTaskParameters {
        &self.learning_params
    }

    /// Set configuration for the learning objective.
    pub fn set_learning_params<T: Into<learning::LearningTaskParameters>>(&mut self, learning_params: T) {
        self.learning_params = learning_params.into();
    }

    /// Check whether verbose output is enabled or not.
    pub fn verbose(&self) -> bool {
        self.verbose
    }

    /// Set to `true` to enable verbose output from XGBoost's C library.
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Get number of parallel threads XGboost will use (if compiled with multiprocessing support).
    ///
    /// If `None`, XGBoost will determine the number of threads to use automatically.
    pub fn threads(&self) -> &Option<u32> {
        &self.threads
    }

    /// Set number of parallel threads XGBoost will use (if compiled with multiprocessing support).
    ///
    /// If `None`, XGBoost will determine the number of threads to use automatically.
    pub fn set_threads<T: Into<Option<u32>>>(&mut self, threads: T) {
        self.threads = threads.into();
    }

    pub(crate) fn as_string_pairs(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();

        v.extend(self.booster_type.as_string_pairs());
        v.extend(self.learning_params.as_string_pairs());

        v.push(("silent".to_owned(), (!self.verbose as u8).to_string()));

        if let Some(nthread) = self.threads {
            v.push(("nthread".to_owned(), nthread.to_string()));
        }

        v
    }
}

type CustomEvaluationMetric = fn(&[f32], &DMatrix) -> f32;

/// Parameters used by the [`Booster::train`](../struct.Booster.html#method.train) method for training new models.
/// Created using [`TrainingParametersBuilder`](struct.TrainingParametersBuilder.html).
#[derive(Builder, Clone)]
pub struct TrainingParameters<'a> {
    pub(crate) dtrain: &'a DMatrix,

    #[builder(default="10")]
    pub(crate) boost_rounds: u32,

    #[builder(default="BoosterParameters::default()")]
    pub(crate) booster_params: BoosterParameters,

    #[builder(default="None")]
    pub(crate) evaluation_sets: Option<&'a[(&'a DMatrix, &'a str)]>,

    #[builder(default="None")]
    pub(crate) custom_objective_fn: Option<CustomObjective>,

    #[builder(default="None")]
    pub(crate) custom_evaluation_fn: Option<CustomEvaluationMetric>,
    // TODO: callbacks
}


///// BoosterParameters for Tweedie Regression.
//#[derive(Builder)]
//#[builder(build_fn(validate = "Self::validate"))]
//#[builder(default)]
//pub struct TweedieRegressionParameters {
//    /// Parameter that controls the variance of the Tweedie distribution.
//    ///
//    /// * var(y) ~ E(y)^tweedie_variance_power
//    /// * range: (1.0, 2.0)
//    /// * set closer to 2 to shift towards a gamma distribution
//    /// * set closer to 1 to shift towards a Poisson distribution
//    tweedie_variance_power: f32,
//}
//
//impl Default for TweedieRegressionParameters {
//    fn default() -> Self {
//        TweedieRegressionParameters {
//            tweedie_variance_power: 1.5,
//        }
//    }
//}
//
//impl TweedieRegressionParametersBuilder {
//    fn validate(&self) -> Result<(), String> {
//        Interval::new_open_open(1.0, 2.0).validate(&self.tweedie_variance_power, "tweedie_variance_power")?;
//        Ok(())
//    }
//}

enum Inclusion {
    Open,
    Closed,
}

struct Interval<T> {
    min: T,
    min_inclusion: Inclusion,
    max: T,
    max_inclusion: Inclusion,
}

impl<T: Display> Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let lower = match self.min_inclusion {
            Inclusion::Closed => '[',
            Inclusion::Open   => '(',
        };
        let upper = match self.max_inclusion {
            Inclusion::Closed => ']',
            Inclusion::Open   => ')',
        };
        write!(f, "{}{}, {}{}", lower, self.min, self.max, upper)
    }
}

impl<T: PartialOrd + Display> Interval<T> {
    fn new(min: T, min_inclusion: Inclusion, max: T, max_inclusion: Inclusion) -> Self {
        Interval { min, min_inclusion, max, max_inclusion }
    }

    fn new_open_open(min: T, max: T) -> Self {
        Interval::new(min, Inclusion::Open, max, Inclusion::Open)
    }

    fn new_open_closed(min: T, max: T) -> Self {
        Interval::new(min, Inclusion::Open, max, Inclusion::Closed)
    }

    fn new_closed_closed(min: T, max: T) -> Self {
        Interval::new(min, Inclusion::Closed, max, Inclusion::Closed)
    }

    fn contains(&self, val: &T) -> bool {
        match self.min_inclusion {
            Inclusion::Closed => if !(val >= &self.min) { return false; },
            Inclusion::Open => if !(val > &self.min) { return false; },
        }
        match self.max_inclusion {
            Inclusion::Closed => if !(val <= &self.max) { return false; },
            Inclusion::Open => if !(val < &self.max) { return false; },
        }
        true
    }

    fn validate(&self, val: &Option<T>, name: &str) -> Result<(), String> {
        match val {
            Some(ref val) => {
                if self.contains(&val) {
                    Ok(())
                } else {
                    Err(format!("Invalid value for '{}' parameter, {} is not in range {}.", name, &val, self))
                }
            },
            None => Ok(())
        }
    }
}
