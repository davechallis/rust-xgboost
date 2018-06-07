use std::default::Default;

use super::{tree, linear, dart};

#[derive(Clone)]
pub enum BoosterParameters {
    GbTree(tree::TreeBoosterParameters),
    GbLinear(linear::LinearBoosterParameters),
    Dart(dart::DartBoosterParameters),
}

impl Default for BoosterParameters {
    fn default() -> Self { BoosterParameters::GbTree(tree::TreeBoosterParameters::default()) }
}

impl BoosterParameters {
    pub(crate) fn as_string_pairs(&self) -> Vec<(String, String)> {
        match *self {
            BoosterParameters::GbTree(ref p) => p.as_string_pairs(),
            BoosterParameters::GbLinear(ref p) => p.as_string_pairs(),
            BoosterParameters::Dart(ref p) => p.as_string_pairs()
        }
    }
}
