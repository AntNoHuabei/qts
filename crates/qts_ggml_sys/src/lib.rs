#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::missing_safety_doc,
    clippy::useless_transmute
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_add_graph() {
        unsafe {
            let params = ggml_init_params {
                mem_size: 16 * 1024 * 1024,
                mem_buffer: core::ptr::null_mut(),
                no_alloc: false,
            };
            let ctx = ggml_init(params);
            assert!(!ctx.is_null());

            let a = ggml_new_tensor_1d(ctx, ggml_type_GGML_TYPE_F32, 1);
            let b = ggml_new_tensor_1d(ctx, ggml_type_GGML_TYPE_F32, 1);
            assert!(!ggml_set_f32(a, 2.0).is_null());
            assert!(!ggml_set_f32(b, 3.0).is_null());
            let sum = ggml_add(ctx, a, b);

            let gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, sum);
            let status = ggml_graph_compute_with_ctx(ctx, gf, 1);
            assert_eq!(status, ggml_status_GGML_STATUS_SUCCESS);
            let v = ggml_get_f32_1d(sum, 0);
            assert!((v - 5.0).abs() < 1e-5);

            ggml_free(ctx);
        }
    }
}
