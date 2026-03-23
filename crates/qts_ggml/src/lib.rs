//! Minimal wrappers around [`ggml_sys`]. Expand as the TTS stack needs more surface area.

pub use ggml_sys as sys;

use std::ptr::NonNull;

/// Owned `ggml_context` with a fixed-size arena (see upstream `ggml_init`).
pub struct Context {
    raw: NonNull<sys::ggml_context>,
}

impl Context {
    pub fn with_buffer_size(mem_size: usize) -> Option<Self> {
        unsafe {
            let params = sys::ggml_init_params {
                mem_size,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: false,
            };
            let p = sys::ggml_init(params);
            NonNull::new(p).map(|raw| Self { raw })
        }
    }

    #[must_use]
    pub fn as_ptr(&self) -> *mut sys::ggml_context {
        self.raw.as_ptr()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_free(self.raw.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_add_via_sys() {
        let ctx = Context::with_buffer_size(16 * 1024 * 1024).expect("ctx");
        unsafe {
            let a = sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_F32, 1);
            let b = sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_F32, 1);
            assert!(!sys::ggml_set_f32(a, 2.0).is_null());
            assert!(!sys::ggml_set_f32(b, 3.0).is_null());
            let sum = sys::ggml_add(ctx.as_ptr(), a, b);
            let gf = sys::ggml_new_graph(ctx.as_ptr());
            sys::ggml_build_forward_expand(gf, sum);
            let st = sys::ggml_graph_compute_with_ctx(ctx.as_ptr(), gf, 1);
            assert_eq!(st, sys::ggml_status_GGML_STATUS_SUCCESS);
            assert!((sys::ggml_get_f32_1d(sum, 0) - 5.0).abs() < 1e-5);
        }
    }
}
