use std::path::Path;
use std::ffi::CString;
use super::{XGBError, XGBResult};

pub fn cstring_from_path<P: AsRef<Path>>(path: P) -> XGBResult<CString> {
    let path = path.as_ref();
    let path_str = match path.to_str() {
        Some(s) => s,
        None    => {
            let msg = format!("Could not encode path '{}' as UTF-8 string", path.to_string_lossy());
            return Err(XGBError::new(&msg));
        },
    };

    Ok(CString::new(path_str).unwrap())
}
