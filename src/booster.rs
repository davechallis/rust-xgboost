use std::{slice, ffi, ptr};
use error::XGBError;
use dmatrix::DMatrix;

use xgboost_sys;

use super::XGBResult;
use parameters::Parameters;

pub struct Booster {
    handle: xgboost_sys::BoosterHandle,
}

impl Booster {
    pub fn create(dmats: &[&DMatrix], params: &Parameters) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        // TODO: check this is safe, if any dmats are freed
        let s: Vec<xgboost_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();
        xgb_call!(xgboost_sys::XGBoosterCreate(s.as_ptr(), dmats.len() as u64, &mut handle))?;

        let mut booster = Booster { handle };
        booster.set_params(params)?;
        Ok(booster)
    }

    pub fn set_params(&mut self, p: &Parameters) -> XGBResult<()> {
        for (key, value) in p.as_string_pairs() {
            self.set_param(&key, &value)?;
        }
        Ok(())
    }

    pub fn update(&mut self, dtrain: &DMatrix, iteration: i32) -> XGBResult<()> {
        xgb_call!(xgboost_sys::XGBoosterUpdateOneIter(self.handle, iteration, dtrain.handle))
    }

    pub fn boost(&mut self, dtrain: &DMatrix, grad: &[f32], hess: &[f32]) -> XGBResult<()> {
        // TODO: error handling
        assert_eq!(grad.len(), hess.len());

        // TODO: _validate_feature_names
        let mut grad_vec = grad.to_vec();
        let mut hess_vec = hess.to_vec();
        xgb_call!(xgboost_sys::XGBoosterBoostOneIter(self.handle,
                                                     dtrain.handle,
                                                     grad_vec.as_mut_ptr(),
                                                     hess_vec.as_mut_ptr(),
                                                     grad_vec.len() as u64))
    }

    pub fn eval_set(&self, dmats: &[&DMatrix], names: &[&str], iteration: i32) -> XGBResult<String> {
        assert_eq!(dmats.len(), names.len());
        let mut s: Vec<xgboost_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();

        let mut evnames = {
            let mut evnames = Vec::new();
            for name in names {
                evnames.push(ffi::CString::new(*name).unwrap().as_ptr());
            }
            evnames
        };
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterEvalOneIter(self.handle,
                                                    iteration,
                                                    s.as_mut_ptr(),
                                                    evnames.as_mut_ptr(),
                                                    dmats.len() as u64,
                                                    &mut out_result))?;
        let out = unsafe { ffi::CStr::from_ptr(out_result).to_str().unwrap().to_owned() };
        Ok(out)
    }

    fn get_attribute_names(&self) -> XGBResult<Vec<String>> {
        let mut out_len = 0;
        let mut out = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterGetAttrNames(self.handle, &mut out_len, &mut out))?;

        let out_ptr_slice = unsafe { slice::from_raw_parts(out, out_len as usize) };
        let out_vec = out_ptr_slice.iter()
            .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
            .collect();
        Ok(out_vec)
    }

    pub fn predict(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        // TODO: bitmask options etc.
        let option_mask = 0;
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(self.handle,
                                          dmat.handle,
                                          option_mask,
                                          ntree_limit,
                                          &mut out_len,
                                          &mut out_result))?;

        let s = unsafe { slice::from_raw_parts(out_result, out_len as usize) };
        Ok(s.to_vec())
    }

    pub fn get_attribute(&self, key: &str) -> XGBResult<Option<String>> {
        let key = ffi::CString::new(key).unwrap();
        let mut out_buf = ptr::null();
        let mut success = 0;
        xgb_call!(xgboost_sys::XGBoosterGetAttr(self.handle, key.as_ptr(), &mut out_buf, &mut success))?;
        if success == 0 {
            return Ok(None);
        }
        assert!(success == 1);

        let c_str: &ffi::CStr = unsafe { ffi::CStr::from_ptr(out_buf) };
        let out = c_str.to_str().unwrap();
        Ok(Some(out.to_owned()))
    }

    fn set_attribute(&mut self, key: &str, value: &str) -> XGBResult<()> {
        let key = ffi::CString::new(key).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_sys::XGBoosterSetAttr(self.handle, key.as_ptr(), value.as_ptr()))
    }

    fn set_param(&mut self, name: &str, value: &str) -> XGBResult<()> {
        let name = ffi::CString::new(name).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_sys::XGBoosterSetParam(self.handle, name.as_ptr(), value.as_ptr()))
    }

    pub(crate) fn load_rabit_checkpoint(&self) -> XGBResult<i32> {
        let mut version = 0;
        xgb_call!(xgboost_sys::XGBoosterLoadRabitCheckpoint(self.handle, &mut version))?;
        Ok(version)
    }

    pub(crate) fn save_rabit_checkpoint(&self) -> XGBResult<()> {
        xgb_call!(xgboost_sys::XGBoosterSaveRabitCheckpoint(self.handle))
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        xgb_call!(xgboost_sys::XGBoosterFree(self.handle)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parameters;

    fn read_train_matrix() -> XGBResult<DMatrix> {
        DMatrix::create_from_file("xgboost-sys/xgboost/demo/data/agaricus.txt.train", true)
    }

    fn load_test_booster() -> Booster {
        let dmat = read_train_matrix().expect("Reading train matrix failed");
        Booster::create(&[&dmat], &Parameters::default()).expect("Creating Booster failed")
    }

    #[test]
    fn set_booster_param() {
        let mut booster = load_test_booster();
        let res = booster.set_param("key", "value");
        assert!(res.is_ok());
    }

    #[test]
    fn load_rabit_version() {
        let version = load_test_booster().load_rabit_checkpoint().unwrap();
        assert_eq!(version, 0);
    }

    #[test]
    fn get_set_attr() {
        let mut booster = load_test_booster();
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster.set_attribute("foo", "bar").expect("Setting attribute failed");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn get_attribute_names() {
        let mut booster = load_test_booster();
        let attrs = booster.get_attribute_names().expect("Getting attributes failed");
        assert_eq!(attrs, Vec::<String>::new());

        booster.set_attribute("foo", "bar").expect("Setting attribute failed");
        booster.set_attribute("another", "another").expect("Setting attribute failed");
        booster.set_attribute("4", "4").expect("Setting attribute failed");
        booster.set_attribute("an even longer attribute name?", "").expect("Setting attribute failed");

        let mut expected = vec!["foo", "another", "4", "an even longer attribute name?"];
        expected.sort();
        let mut attrs = booster.get_attribute_names().expect("Getting attributes failed");
        attrs.sort();
        assert_eq!(attrs, expected);
    }

    #[test]
    fn foo() {
        let dmat_train = DMatrix::create_from_file("xgboost-sys/xgboost/demo/data/agaricus.txt.train", true).unwrap();
        let dmat_test = DMatrix::create_from_file("xgboost-sys/xgboost/demo/data/agaricus.txt.test", true).unwrap();

        let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
            .objective(parameters::learning::Objective::BinaryLogistic)
            .eval_metrics(Some(vec![parameters::learning::EvaluationMetric::LogLoss]))
            .build()
            .unwrap();
        let params = parameters::ParametersBuilder::default()
            .booster_params(parameters::booster::BoosterParameters::GbTree(tree_params))
            .learning_params(learning_params)
            .silent(true)
            .build()
            .unwrap();
        let mut booster = Booster::create(&[&dmat_train, &dmat_test], &params).unwrap();

        for i in 0..10 {
            booster.update(&dmat_train, i).expect("update failed");
            booster.eval_set(&[&dmat_train, &dmat_test], &["train", "test"], i).unwrap();
        }

        let v = booster.predict(&dmat_test).unwrap();
        assert_eq!(v.len(), 1611);

        // first 10 predictions
        let expected_start = [0.0050151693,
                              0.9884467,
                              0.0050151693,
                              0.0050151693,
                              0.026636455,
                              0.11789363,
                              0.9884467,
                              0.01231471,
                              0.9884467,
                              0.00013656063];

        // last 10 predictions
        let expected_end = [0.002520344,
                            0.00060917926,
                            0.99881005,
                            0.00060917926,
                            0.00060917926,
                            0.00060917926,
                            0.00060917926,
                            0.9981102,
                            0.002855195,
                            0.9981102];
        let eps = 1e-6;
        for (pred, expected) in v.iter().zip(&expected_start) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }
        for (pred, expected) in v[v.len()-expected_end.len()..v.len()].iter().zip(&expected_end) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }
    }
}
