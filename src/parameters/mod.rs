use std::default::Default;
use std::fmt::{self, Display};

pub mod tree;
pub mod learning;
pub mod linear;
pub mod dart;
pub mod booster;

#[derive(Builder)]
#[builder(default)]
pub struct Parameters {
    /// Which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model
    /// while gblinear uses linear function.
    booster_params: booster::BoosterParameters,

    learning_params: learning::LearningTaskParameters,

    /// Whether to print running messages or not.
    silent: bool,

    /// Number or parallel threads run xgboost if set.
    ///
    /// default: max number of threads available
    nthread: Option<u32>,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            booster_params: booster::BoosterParameters::default(),
            learning_params: learning::LearningTaskParameters::default(),
            silent: false,
            nthread: None,
        }
    }
}

impl Parameters {
    pub(crate) fn as_string_pairs(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();

        v.extend(self.booster_params.as_string_pairs());
        v.extend(self.learning_params.as_string_pairs());

        v.push(("silent".to_owned(), (self.silent as u8).to_string()));

        if let Some(nthread) = self.nthread {
            v.push(("nthread".to_owned(), nthread.to_string()));
        }

        v
    }
}

/// Parameters for Tweedie Regression.
#[derive(Builder)]
#[builder(build_fn(validate = "Self::validate"))]
#[builder(default)]
pub struct TweedieRegressionParameters {
    /// Parameter that controls the variance of the Tweedie distribution.
    ///
    /// * var(y) ~ E(y)^tweedie_variance_power
    /// * range: (1.0, 2.0)
    /// * set closer to 2 to shift towards a gamma distribution
    /// * set closer to 1 to shift towards a Poisson distribution
    tweedie_variance_power: f32,
}

impl Default for TweedieRegressionParameters {
    fn default() -> Self {
        TweedieRegressionParameters {
            tweedie_variance_power: 1.5,
        }
    }
}

impl TweedieRegressionParametersBuilder {
    fn validate(&self) -> Result<(), String> {
        Interval::new_open_open(1.0, 2.0).validate(self.tweedie_variance_power, "tweedie_variance_power")?;
        Ok(())
    }
}

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

    fn validate(&self, val: Option<T>, name: &str) -> Result<(), String> {
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

