//! Audio tokenizer decoder (vocoder) metadata, weights, and GGML graph execution.

use std::cell::{RefCell, RefMut};
use std::cmp::max;
use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::rc::Rc;

use ggml::sys;

use crate::{model::GgufFile, Qwen3TtsError};

#[derive(Debug, Clone)]
pub struct VocoderConfig {
    pub sample_rate: i32,
    pub n_codebooks: i32,
    pub codebook_size: i32,
    pub codebook_dim: i32,
    pub latent_dim: i32,
    pub hidden_dim: i32,
    pub n_pre_tfm_layers: i32,
    pub n_heads: i32,
    pub ffn_dim: i32,
    pub decoder_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            n_codebooks: 16,
            codebook_size: 2_048,
            codebook_dim: 256,
            latent_dim: 1_024,
            hidden_dim: 512,
            n_pre_tfm_layers: 8,
            n_heads: 16,
            ffn_dim: 1_024,
            decoder_dim: 1_536,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}

pub struct Vocoder {
    config: VocoderConfig,
    weights: VocoderWeights,
}

impl Vocoder {
    pub fn load_from_gguf(file: &GgufFile) -> Result<Self, Qwen3TtsError> {
        unsafe {
            sys::ggml_cpu_init();
        }

        let mut cfg = VocoderConfig::default();
        cfg.sample_rate =
            file.get_u32("qwen3-tts.tokenizer.sample_rate", cfg.sample_rate as u32) as i32;
        cfg.n_codebooks =
            file.get_u32("qwen3-tts.tokenizer.num_codebooks", cfg.n_codebooks as u32) as i32;
        cfg.codebook_size = file.get_u32(
            "qwen3-tts.tokenizer.codebook_size",
            cfg.codebook_size as u32,
        ) as i32;

        let weights = VocoderWeights::load(file, &cfg, BackendSet::new()?)?;
        Ok(Self {
            config: cfg,
            weights,
        })
    }

    #[must_use]
    pub fn config(&self) -> &VocoderConfig {
        &self.config
    }

    pub fn decode(
        &self,
        codes: &[i32],
        n_frames: usize,
        thread_count: usize,
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if codes.len() != n_frames * self.config.n_codebooks as usize {
            return Err(Qwen3TtsError::InvalidInput(
                "vocoder codes shape is invalid".into(),
            ));
        }

        let graph_nodes = 32_768;
        let ctx = ComputeContext::new_graph(graph_nodes)?;
        let graph = unsafe { sys::ggml_new_graph_custom(ctx.as_ptr(), graph_nodes, false) };
        let graph = NonNull::new(graph)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to allocate vocoder graph".into()))?;

        let cb_tensors = (0..self.config.n_codebooks as usize)
            .map(|_| unsafe {
                sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, n_frames as i64)
            })
            .collect::<Vec<_>>();
        let cb_tensors = cb_tensors
            .into_iter()
            .map(|tensor| {
                NonNull::new(tensor)
                    .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to allocate code tensor".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let positions = unsafe {
            sys::ggml_new_tensor_1d(ctx.as_ptr(), sys::ggml_type_GGML_TYPE_I32, n_frames as i64)
        };
        let positions = NonNull::new(positions)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to allocate positions tensor".into()))?;
        let pos_values = (0..n_frames as i32).collect::<Vec<_>>();

        let mut cb_values = Vec::with_capacity(self.config.n_codebooks as usize);

        for cb in 0..self.config.n_codebooks as usize {
            let mut cb_codes = vec![0i32; n_frames];
            for frame in 0..n_frames {
                cb_codes[frame] = codes[frame * self.config.n_codebooks as usize + cb];
            }
            cb_values.push(cb_codes);
        }

        let first_emb =
            unsafe { sys::ggml_get_rows(ctx.as_ptr(), self.weights.vq_first_codebook.as_ptr(), cb_tensors[0].as_ptr()) };
        let first_emb_2d = unsafe {
            sys::ggml_reshape_2d(
                ctx.as_ptr(),
                first_emb,
                self.config.codebook_dim as i64,
                n_frames as i64,
            )
        };
        let first_proj_weight = unsafe {
            sys::ggml_reshape_2d(
                ctx.as_ptr(),
                self.weights.vq_first_output_proj.as_ptr(),
                self.config.codebook_dim as i64,
                self.config.hidden_dim as i64,
            )
        };
        let first_proj_2d = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), first_proj_weight, first_emb_2d) };

        let rest_proj_weight = unsafe {
            sys::ggml_reshape_2d(
                ctx.as_ptr(),
                self.weights.vq_rest_output_proj.as_ptr(),
                self.config.codebook_dim as i64,
                self.config.hidden_dim as i64,
            )
        };
        let mut rest_proj_2d: *mut sys::ggml_tensor = std::ptr::null_mut();
        for cb in 0..self.weights.vq_rest_codebook.len() {
            let rest_emb =
                unsafe { sys::ggml_get_rows(ctx.as_ptr(), self.weights.vq_rest_codebook[cb].as_ptr(), cb_tensors[cb + 1].as_ptr()) };
            let rest_emb_2d = unsafe {
                sys::ggml_reshape_2d(
                    ctx.as_ptr(),
                    rest_emb,
                    self.config.codebook_dim as i64,
                    n_frames as i64,
                )
            };
            let cb_proj_2d = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), rest_proj_weight, rest_emb_2d) };
            rest_proj_2d = if rest_proj_2d.is_null() {
                cb_proj_2d
            } else {
                unsafe { sys::ggml_add(ctx.as_ptr(), rest_proj_2d, cb_proj_2d) }
            };
        }

        let latent_2d = unsafe { sys::ggml_add(ctx.as_ptr(), first_proj_2d, rest_proj_2d) };
        let latent_t = unsafe { sys::ggml_transpose(ctx.as_ptr(), latent_2d) };
        let latent_cont = unsafe { sys::ggml_cont(ctx.as_ptr(), latent_t) };
        let latent = unsafe {
            sys::ggml_reshape_3d(
                ctx.as_ptr(),
                latent_cont,
                n_frames as i64,
                self.config.hidden_dim as i64,
                1,
            )
        };

        let latent_for_conv = unsafe { sys::ggml_cont(ctx.as_ptr(), latent) };
        let latent_padded =
            unsafe { sys::ggml_pad_ext(ctx.as_ptr(), latent_for_conv, 2, 0, 0, 0, 0, 0, 0, 0) };
        let mut cur =
            unsafe { sys::ggml_conv_1d(ctx.as_ptr(), self.weights.pre_conv_w.as_ptr(), latent_padded, 1, 0, 1) };
        if let Some(pre_conv_b) = self.weights.pre_conv_b {
            let bias = unsafe { sys::ggml_reshape_3d(ctx.as_ptr(), pre_conv_b.as_ptr(), 1, self.config.latent_dim as i64, 1) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, bias) };
        }

        let cur_2d = unsafe {
            sys::ggml_reshape_2d(
                ctx.as_ptr(),
                cur,
                n_frames as i64,
                self.config.latent_dim as i64,
            )
        };
        let cur_t = unsafe { sys::ggml_transpose(ctx.as_ptr(), cur_2d) };
        cur = unsafe { sys::ggml_cont(ctx.as_ptr(), cur_t) };
        cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), self.weights.pre_tfm_input_proj_w.as_ptr(), cur) };
        if let Some(input_bias) = self.weights.pre_tfm_input_proj_b {
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, input_bias.as_ptr()) };
        }

        for layer in &self.weights.pre_tfm_layers {
            cur = self.apply_pre_tfm_layer(ctx.as_ptr(), cur, layer, n_frames, positions.as_ptr())?;
        }

        if let Some(norm) = self.weights.pre_tfm_norm_w {
            cur = self.apply_rms_norm(ctx.as_ptr(), cur, norm.as_ptr(), self.config.rms_norm_eps);
        }
        cur = unsafe { sys::ggml_mul_mat(ctx.as_ptr(), self.weights.pre_tfm_output_proj_w.as_ptr(), cur) };
        if let Some(output_bias) = self.weights.pre_tfm_output_proj_b {
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, output_bias.as_ptr()) };
        }

        cur = unsafe { sys::ggml_permute(ctx.as_ptr(), cur, 1, 0, 2, 3) };
        cur = unsafe { sys::ggml_cont(ctx.as_ptr(), cur) };
        cur = unsafe {
            sys::ggml_reshape_3d(
                ctx.as_ptr(),
                cur,
                n_frames as i64,
                self.config.latent_dim as i64,
                1,
            )
        };

        for block in &self.weights.upsample {
            cur = self.apply_upsample_block(ctx.as_ptr(), cur, block)?;
        }

        cur = unsafe { sys::ggml_pad_ext(ctx.as_ptr(), cur, 6, 0, 0, 0, 0, 0, 0, 0) };
        cur = unsafe { sys::ggml_conv_1d(ctx.as_ptr(), self.weights.dec0_conv_w.as_ptr(), cur, 1, 0, 1) };
        if let Some(dec0_bias) = self.weights.dec0_conv_b {
            let bias =
                unsafe { sys::ggml_reshape_3d(ctx.as_ptr(), dec0_bias.as_ptr(), 1, self.config.decoder_dim as i64, 1) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, bias) };
        }

        for (block, upsample_rate) in self.weights.dec_blocks.iter().zip([8, 5, 4, 3]) {
            cur = self.apply_decoder_block(ctx.as_ptr(), cur, block, upsample_rate)?;
        }

        if let (Some(alpha), Some(beta)) = (self.weights.dec5_snake_alpha, self.weights.dec5_snake_beta) {
            cur = self.apply_snake(ctx.as_ptr(), cur, alpha.as_ptr(), beta.as_ptr());
        }

        cur = unsafe { sys::ggml_pad_ext(ctx.as_ptr(), cur, 6, 0, 0, 0, 0, 0, 0, 0) };
        cur = unsafe { sys::ggml_conv_1d(ctx.as_ptr(), self.weights.dec6_conv_w.as_ptr(), cur, 1, 0, 1) };
        if let Some(dec6_bias) = self.weights.dec6_conv_b {
            let bias = unsafe { sys::ggml_reshape_3d(ctx.as_ptr(), dec6_bias.as_ptr(), 1, 1, 1) };
            cur = unsafe { sys::ggml_add(ctx.as_ptr(), cur, bias) };
        }

        cur = unsafe { sys::ggml_tanh(ctx.as_ptr(), cur) };
        cur = unsafe { sys::ggml_reshape_1d(ctx.as_ptr(), cur, (*cur).ne[0]) };
        cur = unsafe { sys::ggml_cont(ctx.as_ptr(), cur) };
        unsafe {
            sys::ggml_build_forward_expand(graph.as_ptr(), cur);
        }

        let n_samples = unsafe { (*cur).ne[0] as usize };
        let mut audio = vec![0.0f32; n_samples];
        let mut uploads = Vec::with_capacity(cb_tensors.len() + 1);
        uploads.push(TensorUpload {
            tensor: positions.as_ptr(),
            bytes: slice_as_bytes(pos_values.as_slice()),
        });
        for (tensor, values) in cb_tensors.iter().zip(cb_values.iter()) {
            uploads.push(TensorUpload {
                tensor: tensor.as_ptr(),
                bytes: slice_as_bytes(values.as_slice()),
            });
        }
        execute_graph(
            &self.weights._backends,
            graph,
            graph_nodes,
            uploads.as_slice(),
            &mut [TensorDownload {
                tensor: cur,
                bytes: slice_as_bytes_mut(audio.as_mut_slice()),
            }],
            thread_count,
            "vocoder graph execution failed",
        )?;
        Ok(audio)
    }

    fn apply_snake(
        &self,
        ctx: *mut sys::ggml_context,
        x: *mut sys::ggml_tensor,
        alpha: *mut sys::ggml_tensor,
        beta: *mut sys::ggml_tensor,
    ) -> *mut sys::ggml_tensor {
        let seq_len = unsafe { (*x).ne[0] };
        let channels = unsafe { (*x).ne[1] };

        let alpha_exp = unsafe { sys::ggml_exp(ctx, alpha) };
        let alpha_3d = unsafe { sys::ggml_reshape_3d(ctx, alpha_exp, 1, channels, 1) };
        let _ = seq_len;
        let ax = unsafe { sys::ggml_mul(ctx, x, alpha_3d) };
        let sin_ax = unsafe { sys::ggml_sin(ctx, ax) };
        let sin_sq = unsafe { sys::ggml_sqr(ctx, sin_ax) };

        let neg_beta = unsafe { sys::ggml_scale(ctx, beta, -1.0) };
        let inv_beta_exp = unsafe { sys::ggml_exp(ctx, neg_beta) };
        let inv_beta_3d = unsafe { sys::ggml_reshape_3d(ctx, inv_beta_exp, 1, channels, 1) };
        let scaled_sin = unsafe { sys::ggml_mul(ctx, sin_sq, inv_beta_3d) };
        unsafe { sys::ggml_add(ctx, x, scaled_sin) }
    }

    fn apply_rms_norm(
        &self,
        ctx: *mut sys::ggml_context,
        x: *mut sys::ggml_tensor,
        w: *mut sys::ggml_tensor,
        eps: f32,
    ) -> *mut sys::ggml_tensor {
        let normed = unsafe { sys::ggml_rms_norm(ctx, x, eps) };
        unsafe { sys::ggml_mul(ctx, normed, w) }
    }

    fn apply_pre_tfm_layer(
        &self,
        ctx: *mut sys::ggml_context,
        x: *mut sys::ggml_tensor,
        layer: &PreTfmLayerWeights,
        n_frames: usize,
        positions: *mut sys::ggml_tensor,
    ) -> Result<*mut sys::ggml_tensor, Qwen3TtsError> {
        let head_dim = self.config.latent_dim / self.config.n_heads;
        let residual = x;
        let mut normed = self.apply_rms_norm(
            ctx,
            x,
            layer.attn_norm_w.as_ptr(),
            self.config.rms_norm_eps,
        );
        let mut q_cur = unsafe { sys::ggml_mul_mat(ctx, layer.attn_q_w.as_ptr(), normed) };
        let mut k_cur = unsafe { sys::ggml_mul_mat(ctx, layer.attn_k_w.as_ptr(), normed) };
        let mut v_cur = unsafe { sys::ggml_mul_mat(ctx, layer.attn_v_w.as_ptr(), normed) };

        q_cur = unsafe {
            sys::ggml_reshape_3d(ctx, q_cur, head_dim as i64, self.config.n_heads as i64, n_frames as i64)
        };
        k_cur = unsafe {
            sys::ggml_reshape_3d(ctx, k_cur, head_dim as i64, self.config.n_heads as i64, n_frames as i64)
        };
        v_cur = unsafe {
            sys::ggml_reshape_3d(ctx, v_cur, head_dim as i64, self.config.n_heads as i64, n_frames as i64)
        };

        q_cur = unsafe {
            sys::ggml_rope_ext(
                ctx,
                q_cur,
                positions,
                std::ptr::null_mut(),
                head_dim,
                sys::GGML_ROPE_TYPE_NEOX as i32,
                0,
                self.config.rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
        };
        k_cur = unsafe {
            sys::ggml_rope_ext(
                ctx,
                k_cur,
                positions,
                std::ptr::null_mut(),
                head_dim,
                sys::GGML_ROPE_TYPE_NEOX as i32,
                0,
                self.config.rope_theta,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
        };

        let q = unsafe { sys::ggml_permute(ctx, q_cur, 0, 2, 1, 3) };
        let k = unsafe { sys::ggml_permute(ctx, k_cur, 0, 2, 1, 3) };
        let mut v = unsafe { sys::ggml_permute(ctx, v_cur, 0, 2, 1, 3) };
        let mut kq = unsafe { sys::ggml_mul_mat(ctx, k, q) };
        kq = unsafe { sys::ggml_scale(ctx, kq, 1.0 / (head_dim as f32).sqrt()) };
        kq = unsafe { sys::ggml_diag_mask_inf(ctx, kq, 0) };
        kq = unsafe { sys::ggml_soft_max(ctx, kq) };
        v = unsafe { sys::ggml_cont(ctx, sys::ggml_transpose(ctx, v)) };
        let mut kqv = unsafe { sys::ggml_mul_mat(ctx, v, kq) };
        kqv = unsafe { sys::ggml_permute(ctx, kqv, 0, 2, 1, 3) };
        let mut attn_out =
            unsafe { sys::ggml_cont_2d(ctx, kqv, self.config.n_heads as i64 * head_dim as i64, n_frames as i64) };
        attn_out = unsafe { sys::ggml_mul_mat(ctx, layer.attn_output_w.as_ptr(), attn_out) };
        if let Some(attn_scale) = layer.attn_scale {
            attn_out = unsafe { sys::ggml_mul(ctx, attn_out, attn_scale.as_ptr()) };
        }

        let mut x = unsafe { sys::ggml_add(ctx, residual, attn_out) };
        let residual = x;
        normed = self.apply_rms_norm(ctx, x, layer.ffn_norm_w.as_ptr(), self.config.rms_norm_eps);
        let mut gate = unsafe { sys::ggml_mul_mat(ctx, layer.ffn_gate_w.as_ptr(), normed) };
        let up = unsafe { sys::ggml_mul_mat(ctx, layer.ffn_up_w.as_ptr(), normed) };
        gate = unsafe { sys::ggml_silu(ctx, gate) };
        let mut ffn_out = unsafe { sys::ggml_mul(ctx, gate, up) };
        ffn_out = unsafe { sys::ggml_mul_mat(ctx, layer.ffn_down_w.as_ptr(), ffn_out) };
        if let Some(ffn_scale) = layer.ffn_scale {
            ffn_out = unsafe { sys::ggml_mul(ctx, ffn_out, ffn_scale.as_ptr()) };
        }
        x = unsafe { sys::ggml_add(ctx, residual, ffn_out) };
        Ok(x)
    }

    fn apply_upsample_block(
        &self,
        ctx: *mut sys::ggml_context,
        x: *mut sys::ggml_tensor,
        block: &UpsampleBlockWeights,
    ) -> Result<*mut sys::ggml_tensor, Qwen3TtsError> {
        let seq_len = unsafe { (*x).ne[0] };
        let channels = unsafe { (*x).ne[1] };
        let mut x_2d = unsafe { sys::ggml_reshape_2d(ctx, x, seq_len, channels) };
        x_2d = unsafe { sys::ggml_conv_transpose_1d(ctx, block.conv_w.as_ptr(), x_2d, 2, 0, 1) };
        let new_seq_len = unsafe { (*x_2d).ne[0] };
        let mut x = unsafe { sys::ggml_reshape_3d(ctx, x_2d, new_seq_len, channels, 1) };
        if let Some(conv_b) = block.conv_b {
            x = unsafe { sys::ggml_add(ctx, x, sys::ggml_reshape_3d(ctx, conv_b.as_ptr(), 1, channels, 1)) };
        }

        let residual = x;
        if let Some(dwconv_w) = block.dwconv_w {
            x = unsafe { sys::ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0) };
            x = unsafe { sys::ggml_conv_1d_dw(ctx, dwconv_w.as_ptr(), x, 1, 0, 1) };
            if let Some(dwconv_b) = block.dwconv_b {
                x = unsafe { sys::ggml_add(ctx, x, sys::ggml_reshape_3d(ctx, dwconv_b.as_ptr(), 1, channels, 1)) };
            }
        }

        x = unsafe { sys::ggml_permute(ctx, x, 1, 0, 2, 3) };
        x = unsafe { sys::ggml_cont(ctx, x) };
        if let (Some(norm_w), Some(norm_b)) = (block.norm_w, block.norm_b) {
            x = unsafe { sys::ggml_norm(ctx, x, 1e-6) };
            x = unsafe { sys::ggml_mul(ctx, x, norm_w.as_ptr()) };
            x = unsafe { sys::ggml_add(ctx, x, norm_b.as_ptr()) };
        }
        x = unsafe { sys::ggml_mul_mat(ctx, block.pwconv1_w.as_ptr(), x) };
        if let Some(pwconv1_b) = block.pwconv1_b {
            x = unsafe { sys::ggml_add(ctx, x, pwconv1_b.as_ptr()) };
        }
        x = unsafe { sys::ggml_gelu(ctx, x) };
        x = unsafe { sys::ggml_mul_mat(ctx, block.pwconv2_w.as_ptr(), x) };
        if let Some(pwconv2_b) = block.pwconv2_b {
            x = unsafe { sys::ggml_add(ctx, x, pwconv2_b.as_ptr()) };
        }
        x = unsafe { sys::ggml_permute(ctx, x, 1, 0, 2, 3) };
        x = unsafe { sys::ggml_cont(ctx, x) };
        if let Some(gamma) = block.gamma {
            let gamma_3d = unsafe { sys::ggml_reshape_3d(ctx, gamma.as_ptr(), 1, channels, 1) };
            let _ = new_seq_len;
            x = unsafe { sys::ggml_mul(ctx, x, gamma_3d) };
        }
        Ok(unsafe { sys::ggml_add(ctx, residual, x) })
    }

    fn apply_residual_block(
        &self,
        ctx: *mut sys::ggml_context,
        x: *mut sys::ggml_tensor,
        block: &ResidualBlockWeights,
    ) -> Result<*mut sys::ggml_tensor, Qwen3TtsError> {
        let residual = x;
        let mut x = x;
        if let (Some(alpha), Some(beta)) = (block.act1_alpha, block.act1_beta) {
            x = self.apply_snake(ctx, x, alpha.as_ptr(), beta.as_ptr());
        }
        let out_channels = unsafe { (*block.conv1_w.as_ptr()).ne[2] };
        let padding = 6 * block.dilation;
        x = unsafe { sys::ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0) };
        x = unsafe { sys::ggml_conv_1d(ctx, block.conv1_w.as_ptr(), x, 1, 0, block.dilation) };
        if let Some(conv1_b) = block.conv1_b {
            x = unsafe { sys::ggml_add(ctx, x, sys::ggml_reshape_3d(ctx, conv1_b.as_ptr(), 1, out_channels, 1)) };
        }
        if let (Some(alpha), Some(beta)) = (block.act2_alpha, block.act2_beta) {
            x = self.apply_snake(ctx, x, alpha.as_ptr(), beta.as_ptr());
        }
        let out_channels = unsafe { (*block.conv2_w.as_ptr()).ne[2] };
        x = unsafe { sys::ggml_conv_1d(ctx, block.conv2_w.as_ptr(), x, 1, 0, 1) };
        if let Some(conv2_b) = block.conv2_b {
            x = unsafe { sys::ggml_add(ctx, x, sys::ggml_reshape_3d(ctx, conv2_b.as_ptr(), 1, out_channels, 1)) };
        }
        Ok(unsafe { sys::ggml_add(ctx, residual, x) })
    }

    fn apply_decoder_block(
        &self,
        ctx: *mut sys::ggml_context,
        x: *mut sys::ggml_tensor,
        block: &DecoderBlockWeights,
        upsample_rate: i32,
    ) -> Result<*mut sys::ggml_tensor, Qwen3TtsError> {
        let mut x = x;
        if let (Some(alpha), Some(beta)) = (block.snake_alpha, block.snake_beta) {
            x = self.apply_snake(ctx, x, alpha.as_ptr(), beta.as_ptr());
        }
        let seq_len = unsafe { (*x).ne[0] };
        let in_channels = unsafe { (*x).ne[1] };
        let out_channels = unsafe { (*block.conv_t_w.as_ptr()).ne[1] };
        let kernel_size = unsafe { (*block.conv_t_w.as_ptr()).ne[0] as i32 };

        let x_2d = unsafe { sys::ggml_reshape_2d(ctx, x, seq_len, in_channels) };
        let x_2d =
            unsafe { sys::ggml_conv_transpose_1d(ctx, block.conv_t_w.as_ptr(), x_2d, upsample_rate, 0, 1) };
        let new_seq_len = unsafe { (*x_2d).ne[0] };
        let mut x = unsafe { sys::ggml_reshape_3d(ctx, x_2d, new_seq_len, out_channels, 1) };
        let pad = kernel_size - upsample_rate;
        let out_seq_len = new_seq_len - (pad * 2) as i64;
        x = unsafe {
            sys::ggml_view_3d(
                ctx,
                x,
                out_seq_len,
                out_channels,
                1,
                (*x).nb[1],
                (*x).nb[2],
                pad as usize * (*x).nb[0],
            )
        };
        x = unsafe { sys::ggml_cont(ctx, x) };
        if let Some(conv_t_b) = block.conv_t_b {
            x = unsafe { sys::ggml_add(ctx, x, sys::ggml_reshape_3d(ctx, conv_t_b.as_ptr(), 1, out_channels, 1)) };
        }
        for residual in &block.res {
            x = self.apply_residual_block(ctx, x, residual)?;
        }
        Ok(x)
    }
}

struct VocoderWeights {
    _ctx: OwnedContext,
    _backends: BackendSet,
    _buffer: OwnedBuffer,
    vq_first_output_proj: NonNull<sys::ggml_tensor>,
    vq_first_codebook: NonNull<sys::ggml_tensor>,
    _vq_first_usage: Option<NonNull<sys::ggml_tensor>>,
    vq_rest_output_proj: NonNull<sys::ggml_tensor>,
    vq_rest_codebook: Vec<NonNull<sys::ggml_tensor>>,
    _vq_rest_usage: Vec<Option<NonNull<sys::ggml_tensor>>>,
    pre_conv_w: NonNull<sys::ggml_tensor>,
    pre_conv_b: Option<NonNull<sys::ggml_tensor>>,
    pre_tfm_input_proj_w: NonNull<sys::ggml_tensor>,
    pre_tfm_input_proj_b: Option<NonNull<sys::ggml_tensor>>,
    pre_tfm_layers: Vec<PreTfmLayerWeights>,
    pre_tfm_norm_w: Option<NonNull<sys::ggml_tensor>>,
    pre_tfm_output_proj_w: NonNull<sys::ggml_tensor>,
    pre_tfm_output_proj_b: Option<NonNull<sys::ggml_tensor>>,
    upsample: Vec<UpsampleBlockWeights>,
    dec0_conv_w: NonNull<sys::ggml_tensor>,
    dec0_conv_b: Option<NonNull<sys::ggml_tensor>>,
    dec_blocks: Vec<DecoderBlockWeights>,
    dec5_snake_alpha: Option<NonNull<sys::ggml_tensor>>,
    dec5_snake_beta: Option<NonNull<sys::ggml_tensor>>,
    dec6_conv_w: NonNull<sys::ggml_tensor>,
    dec6_conv_b: Option<NonNull<sys::ggml_tensor>>,
}

impl VocoderWeights {
    fn load(file: &GgufFile, cfg: &VocoderConfig, backends: BackendSet) -> Result<Self, Qwen3TtsError> {
        let ctx = OwnedContext::new_for_tensor_metadata(320)?;

        let vq_first_output_proj =
            load_tensor_into_context(file, ctx.as_ptr(), "tok_dec.vq_first.output_proj.weight")?;
        let vq_first_codebook =
            load_tensor_into_context(file, ctx.as_ptr(), "tok_dec.vq_first.0.codebook")?;
        let vq_first_usage =
            load_optional_tensor_into_context(file, ctx.as_ptr(), "tok_dec.vq_first.0.usage")?;
        let vq_rest_output_proj =
            load_tensor_into_context(file, ctx.as_ptr(), "tok_dec.vq_rest.output_proj.weight")?;
        let mut vq_rest_codebook = Vec::with_capacity((cfg.n_codebooks - 1) as usize);
        let mut vq_rest_usage = Vec::with_capacity((cfg.n_codebooks - 1) as usize);
        for cb_idx in 0..(cfg.n_codebooks - 1) {
            vq_rest_codebook.push(load_tensor_into_context(
                file,
                ctx.as_ptr(),
                &format!("tok_dec.vq_rest.{cb_idx}.codebook"),
            )?);
            vq_rest_usage.push(load_optional_tensor_into_context(
                file,
                ctx.as_ptr(),
                &format!("tok_dec.vq_rest.{cb_idx}.usage"),
            )?);
        }

        let pre_conv_w = load_tensor_into_context(file, ctx.as_ptr(), "tok_dec.pre_conv.weight")?;
        let pre_conv_b = load_optional_tensor_into_context(file, ctx.as_ptr(), "tok_dec.pre_conv.bias")?;
        let pre_tfm_input_proj_w =
            load_tensor_into_context(file, ctx.as_ptr(), "tok_dec.pre_tfm.input_proj.weight")?;
        let pre_tfm_input_proj_b =
            load_optional_tensor_into_context(file, ctx.as_ptr(), "tok_dec.pre_tfm.input_proj.bias")?;
        let pre_tfm_norm_w =
            load_optional_tensor_into_context(file, ctx.as_ptr(), "tok_dec.pre_tfm.norm.weight")?;
        let pre_tfm_output_proj_w =
            load_tensor_into_context(file, ctx.as_ptr(), "tok_dec.pre_tfm.output_proj.weight")?;
        let pre_tfm_output_proj_b =
            load_optional_tensor_into_context(file, ctx.as_ptr(), "tok_dec.pre_tfm.output_proj.bias")?;

        let mut pre_tfm_layers = Vec::with_capacity(cfg.n_pre_tfm_layers as usize);
        for layer_idx in 0..cfg.n_pre_tfm_layers {
            let prefix = format!("tok_dec.pre_tfm.blk.{layer_idx}.");
            pre_tfm_layers.push(PreTfmLayerWeights {
                attn_norm_w: load_tensor_into_context(file, ctx.as_ptr(), &(prefix.clone() + "attn_norm.weight"))?,
                attn_q_w: load_tensor_into_context(file, ctx.as_ptr(), &(prefix.clone() + "attn_q.weight"))?,
                attn_k_w: load_tensor_into_context(file, ctx.as_ptr(), &(prefix.clone() + "attn_k.weight"))?,
                attn_v_w: load_tensor_into_context(file, ctx.as_ptr(), &(prefix.clone() + "attn_v.weight"))?,
                attn_output_w: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &(prefix.clone() + "attn_output.weight"),
                )?,
                attn_scale: load_optional_tensor_into_context(file, ctx.as_ptr(), &(prefix.clone() + "attn_scale"))?,
                ffn_norm_w: load_tensor_into_context(file, ctx.as_ptr(), &(prefix.clone() + "ffn_norm.weight"))?,
                ffn_gate_w: load_tensor_into_context(file, ctx.as_ptr(), &(prefix.clone() + "ffn_gate.weight"))?,
                ffn_up_w: load_tensor_into_context(file, ctx.as_ptr(), &(prefix.clone() + "ffn_up.weight"))?,
                ffn_down_w: load_tensor_into_context(file, ctx.as_ptr(), &(prefix.clone() + "ffn_down.weight"))?,
                ffn_scale: load_optional_tensor_into_context(file, ctx.as_ptr(), &(prefix + "ffn_scale"))?,
            });
        }

        let mut upsample = Vec::with_capacity(2);
        for block_idx in 0..2 {
            upsample.push(UpsampleBlockWeights {
                conv_w: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.conv.weight"),
                )?,
                conv_b: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.conv.bias"),
                )?,
                dwconv_w: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.dwconv.weight"),
                )?,
                dwconv_b: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.dwconv.bias"),
                )?,
                norm_w: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.norm.weight"),
                )?,
                norm_b: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.norm.bias"),
                )?,
                pwconv1_w: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.pwconv1.weight"),
                )?,
                pwconv1_b: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.pwconv1.bias"),
                )?,
                pwconv2_w: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.pwconv2.weight"),
                )?,
                pwconv2_b: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.pwconv2.bias"),
                )?,
                gamma: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.upsample.{block_idx}.gamma"),
                )?,
            });
        }

        let dec0_conv_w = load_tensor_into_context(file, ctx.as_ptr(), "tok_dec.dec.0.conv.weight")?;
        let dec0_conv_b = load_optional_tensor_into_context(file, ctx.as_ptr(), "tok_dec.dec.0.conv.bias")?;
        let mut dec_blocks = Vec::with_capacity(4);
        for block_idx in 1..=4 {
            let dilations = [1, 3, 9];
            let mut residuals = Vec::with_capacity(3);
            for (residual_idx, dilation) in (2..=4).zip(dilations) {
                residuals.push(ResidualBlockWeights {
                    dilation,
                    act1_alpha: load_optional_tensor_into_context(
                        file,
                        ctx.as_ptr(),
                        &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.act1.alpha"),
                    )?,
                    act1_beta: load_optional_tensor_into_context(
                        file,
                        ctx.as_ptr(),
                        &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.act1.beta"),
                    )?,
                    conv1_w: load_tensor_into_context(
                        file,
                        ctx.as_ptr(),
                        &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.conv1.weight"),
                    )?,
                    conv1_b: load_optional_tensor_into_context(
                        file,
                        ctx.as_ptr(),
                        &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.conv1.bias"),
                    )?,
                    act2_alpha: load_optional_tensor_into_context(
                        file,
                        ctx.as_ptr(),
                        &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.act2.alpha"),
                    )?,
                    act2_beta: load_optional_tensor_into_context(
                        file,
                        ctx.as_ptr(),
                        &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.act2.beta"),
                    )?,
                    conv2_w: load_tensor_into_context(
                        file,
                        ctx.as_ptr(),
                        &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.conv2.weight"),
                    )?,
                    conv2_b: load_optional_tensor_into_context(
                        file,
                        ctx.as_ptr(),
                        &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.conv2.bias"),
                    )?,
                });
            }
            dec_blocks.push(DecoderBlockWeights {
                snake_alpha: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.dec.{block_idx}.snake.alpha"),
                )?,
                snake_beta: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.dec.{block_idx}.snake.beta"),
                )?,
                conv_t_w: load_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.dec.{block_idx}.conv_t.weight"),
                )?,
                conv_t_b: load_optional_tensor_into_context(
                    file,
                    ctx.as_ptr(),
                    &format!("tok_dec.dec.{block_idx}.conv_t.bias"),
                )?,
                res: residuals,
            });
        }

        let dec5_snake_alpha =
            load_optional_tensor_into_context(file, ctx.as_ptr(), "tok_dec.dec.5.snake.alpha")?;
        let dec5_snake_beta =
            load_optional_tensor_into_context(file, ctx.as_ptr(), "tok_dec.dec.5.snake.beta")?;
        let dec6_conv_w = load_tensor_into_context(file, ctx.as_ptr(), "tok_dec.dec.6.conv.weight")?;
        let dec6_conv_b = load_optional_tensor_into_context(file, ctx.as_ptr(), "tok_dec.dec.6.conv.bias")?;

        let buffer = OwnedBuffer::alloc(ctx.as_ptr(), backends.primary_ptr())?;
        for name in [
            "tok_dec.vq_first.output_proj.weight",
            "tok_dec.vq_first.0.codebook",
            "tok_dec.vq_first.0.usage",
            "tok_dec.vq_rest.output_proj.weight",
            "tok_dec.pre_conv.weight",
            "tok_dec.pre_conv.bias",
            "tok_dec.pre_tfm.input_proj.weight",
            "tok_dec.pre_tfm.input_proj.bias",
            "tok_dec.pre_tfm.norm.weight",
            "tok_dec.pre_tfm.output_proj.weight",
            "tok_dec.pre_tfm.output_proj.bias",
            "tok_dec.dec.0.conv.weight",
            "tok_dec.dec.0.conv.bias",
            "tok_dec.dec.5.snake.alpha",
            "tok_dec.dec.5.snake.beta",
            "tok_dec.dec.6.conv.weight",
            "tok_dec.dec.6.conv.bias",
        ] {
            upload_named_tensor(file, name, &[
                ("tok_dec.vq_first.output_proj.weight", Some(vq_first_output_proj)),
                ("tok_dec.vq_first.0.codebook", Some(vq_first_codebook)),
                ("tok_dec.vq_first.0.usage", vq_first_usage),
                ("tok_dec.vq_rest.output_proj.weight", Some(vq_rest_output_proj)),
                ("tok_dec.pre_conv.weight", Some(pre_conv_w)),
                ("tok_dec.pre_conv.bias", pre_conv_b),
                ("tok_dec.pre_tfm.input_proj.weight", Some(pre_tfm_input_proj_w)),
                ("tok_dec.pre_tfm.input_proj.bias", pre_tfm_input_proj_b),
                ("tok_dec.pre_tfm.norm.weight", pre_tfm_norm_w),
                ("tok_dec.pre_tfm.output_proj.weight", Some(pre_tfm_output_proj_w)),
                ("tok_dec.pre_tfm.output_proj.bias", pre_tfm_output_proj_b),
                ("tok_dec.dec.0.conv.weight", Some(dec0_conv_w)),
                ("tok_dec.dec.0.conv.bias", dec0_conv_b),
                ("tok_dec.dec.5.snake.alpha", dec5_snake_alpha),
                ("tok_dec.dec.5.snake.beta", dec5_snake_beta),
                ("tok_dec.dec.6.conv.weight", Some(dec6_conv_w)),
                ("tok_dec.dec.6.conv.bias", dec6_conv_b),
            ])?;
        }
        for cb_idx in 0..(cfg.n_codebooks - 1) as usize {
            upload_tensor(file, &format!("tok_dec.vq_rest.{cb_idx}.codebook"), vq_rest_codebook[cb_idx])?;
            upload_optional_tensor(file, &format!("tok_dec.vq_rest.{cb_idx}.usage"), vq_rest_usage[cb_idx])?;
        }
        for (layer_idx, layer) in pre_tfm_layers.iter().enumerate() {
            upload_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.attn_norm.weight"), layer.attn_norm_w)?;
            upload_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.attn_q.weight"), layer.attn_q_w)?;
            upload_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.attn_k.weight"), layer.attn_k_w)?;
            upload_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.attn_v.weight"), layer.attn_v_w)?;
            upload_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.attn_output.weight"), layer.attn_output_w)?;
            upload_optional_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.attn_scale"), layer.attn_scale)?;
            upload_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.ffn_norm.weight"), layer.ffn_norm_w)?;
            upload_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.ffn_gate.weight"), layer.ffn_gate_w)?;
            upload_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.ffn_up.weight"), layer.ffn_up_w)?;
            upload_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.ffn_down.weight"), layer.ffn_down_w)?;
            upload_optional_tensor(file, &format!("tok_dec.pre_tfm.blk.{layer_idx}.ffn_scale"), layer.ffn_scale)?;
        }
        for (block_idx, block) in upsample.iter().enumerate() {
            upload_tensor(file, &format!("tok_dec.upsample.{block_idx}.conv.weight"), block.conv_w)?;
            upload_optional_tensor(file, &format!("tok_dec.upsample.{block_idx}.conv.bias"), block.conv_b)?;
            upload_optional_tensor(file, &format!("tok_dec.upsample.{block_idx}.dwconv.weight"), block.dwconv_w)?;
            upload_optional_tensor(file, &format!("tok_dec.upsample.{block_idx}.dwconv.bias"), block.dwconv_b)?;
            upload_optional_tensor(file, &format!("tok_dec.upsample.{block_idx}.norm.weight"), block.norm_w)?;
            upload_optional_tensor(file, &format!("tok_dec.upsample.{block_idx}.norm.bias"), block.norm_b)?;
            upload_tensor(file, &format!("tok_dec.upsample.{block_idx}.pwconv1.weight"), block.pwconv1_w)?;
            upload_optional_tensor(file, &format!("tok_dec.upsample.{block_idx}.pwconv1.bias"), block.pwconv1_b)?;
            upload_tensor(file, &format!("tok_dec.upsample.{block_idx}.pwconv2.weight"), block.pwconv2_w)?;
            upload_optional_tensor(file, &format!("tok_dec.upsample.{block_idx}.pwconv2.bias"), block.pwconv2_b)?;
            upload_optional_tensor(file, &format!("tok_dec.upsample.{block_idx}.gamma"), block.gamma)?;
        }
        for (block_offset, block) in dec_blocks.iter().enumerate() {
            let block_idx = block_offset + 1;
            upload_optional_tensor(file, &format!("tok_dec.dec.{block_idx}.snake.alpha"), block.snake_alpha)?;
            upload_optional_tensor(file, &format!("tok_dec.dec.{block_idx}.snake.beta"), block.snake_beta)?;
            upload_tensor(file, &format!("tok_dec.dec.{block_idx}.conv_t.weight"), block.conv_t_w)?;
            upload_optional_tensor(file, &format!("tok_dec.dec.{block_idx}.conv_t.bias"), block.conv_t_b)?;
            for (res_offset, residual) in block.res.iter().enumerate() {
                let residual_idx = res_offset + 2;
                upload_optional_tensor(file, &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.act1.alpha"), residual.act1_alpha)?;
                upload_optional_tensor(file, &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.act1.beta"), residual.act1_beta)?;
                upload_tensor(file, &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.conv1.weight"), residual.conv1_w)?;
                upload_optional_tensor(file, &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.conv1.bias"), residual.conv1_b)?;
                upload_optional_tensor(file, &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.act2.alpha"), residual.act2_alpha)?;
                upload_optional_tensor(file, &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.act2.beta"), residual.act2_beta)?;
                upload_tensor(file, &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.conv2.weight"), residual.conv2_w)?;
                upload_optional_tensor(file, &format!("tok_dec.dec.{block_idx}.res.{residual_idx}.conv2.bias"), residual.conv2_b)?;
            }
        }

        normalize_codebook(vq_first_codebook, vq_first_usage)?;
        for (codebook, usage) in vq_rest_codebook.iter().copied().zip(vq_rest_usage.iter().copied()) {
            normalize_codebook(codebook, usage)?;
        }

        Ok(Self {
            _ctx: ctx,
            _backends: backends,
            _buffer: buffer,
            vq_first_output_proj,
            vq_first_codebook,
            _vq_first_usage: vq_first_usage,
            vq_rest_output_proj,
            vq_rest_codebook,
            _vq_rest_usage: vq_rest_usage,
            pre_conv_w,
            pre_conv_b,
            pre_tfm_input_proj_w,
            pre_tfm_input_proj_b,
            pre_tfm_layers,
            pre_tfm_norm_w,
            pre_tfm_output_proj_w,
            pre_tfm_output_proj_b,
            upsample,
            dec0_conv_w,
            dec0_conv_b,
            dec_blocks,
            dec5_snake_alpha,
            dec5_snake_beta,
            dec6_conv_w,
            dec6_conv_b,
        })
    }
}

struct PreTfmLayerWeights {
    attn_norm_w: NonNull<sys::ggml_tensor>,
    attn_q_w: NonNull<sys::ggml_tensor>,
    attn_k_w: NonNull<sys::ggml_tensor>,
    attn_v_w: NonNull<sys::ggml_tensor>,
    attn_output_w: NonNull<sys::ggml_tensor>,
    attn_scale: Option<NonNull<sys::ggml_tensor>>,
    ffn_norm_w: NonNull<sys::ggml_tensor>,
    ffn_gate_w: NonNull<sys::ggml_tensor>,
    ffn_up_w: NonNull<sys::ggml_tensor>,
    ffn_down_w: NonNull<sys::ggml_tensor>,
    ffn_scale: Option<NonNull<sys::ggml_tensor>>,
}

struct UpsampleBlockWeights {
    conv_w: NonNull<sys::ggml_tensor>,
    conv_b: Option<NonNull<sys::ggml_tensor>>,
    dwconv_w: Option<NonNull<sys::ggml_tensor>>,
    dwconv_b: Option<NonNull<sys::ggml_tensor>>,
    norm_w: Option<NonNull<sys::ggml_tensor>>,
    norm_b: Option<NonNull<sys::ggml_tensor>>,
    pwconv1_w: NonNull<sys::ggml_tensor>,
    pwconv1_b: Option<NonNull<sys::ggml_tensor>>,
    pwconv2_w: NonNull<sys::ggml_tensor>,
    pwconv2_b: Option<NonNull<sys::ggml_tensor>>,
    gamma: Option<NonNull<sys::ggml_tensor>>,
}

struct ResidualBlockWeights {
    dilation: i32,
    act1_alpha: Option<NonNull<sys::ggml_tensor>>,
    act1_beta: Option<NonNull<sys::ggml_tensor>>,
    conv1_w: NonNull<sys::ggml_tensor>,
    conv1_b: Option<NonNull<sys::ggml_tensor>>,
    act2_alpha: Option<NonNull<sys::ggml_tensor>>,
    act2_beta: Option<NonNull<sys::ggml_tensor>>,
    conv2_w: NonNull<sys::ggml_tensor>,
    conv2_b: Option<NonNull<sys::ggml_tensor>>,
}

struct DecoderBlockWeights {
    snake_alpha: Option<NonNull<sys::ggml_tensor>>,
    snake_beta: Option<NonNull<sys::ggml_tensor>>,
    conv_t_w: NonNull<sys::ggml_tensor>,
    conv_t_b: Option<NonNull<sys::ggml_tensor>>,
    res: Vec<ResidualBlockWeights>,
}

struct OwnedContext {
    raw: NonNull<sys::ggml_context>,
}

impl OwnedContext {
    fn new(mem_size: usize, no_alloc: bool) -> Result<Self, Qwen3TtsError> {
        let raw = unsafe {
            sys::ggml_init(sys::ggml_init_params {
                mem_size,
                mem_buffer: std::ptr::null_mut(),
                no_alloc,
            })
        };
        let raw = NonNull::new(raw)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to initialize ggml context".into()))?;
        Ok(Self { raw })
    }

    fn new_for_tensor_metadata(n_tensors: usize) -> Result<Self, Qwen3TtsError> {
        Self::new(max(1, n_tensors) * unsafe { sys::ggml_tensor_overhead() }, true)
    }

    fn as_ptr(&self) -> *mut sys::ggml_context {
        self.raw.as_ptr()
    }
}

impl Drop for OwnedContext {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_free(self.raw.as_ptr());
        }
    }
}

struct ComputeContext(OwnedContext);

impl ComputeContext {
    fn new_graph(max_nodes: usize) -> Result<Self, Qwen3TtsError> {
        Ok(Self(OwnedContext::new(graph_metadata_mem_size(max_nodes), true)?))
    }

    fn as_ptr(&self) -> *mut sys::ggml_context {
        self.0.as_ptr()
    }
}

#[derive(Clone)]
struct BackendSet(Rc<BackendSetInner>);

struct BackendSetInner {
    primary: OwnedBackend,
    cpu_fallback: Option<OwnedBackend>,
    primary_galloc: RefCell<OwnedGallocr>,
}

impl BackendSet {
    fn new() -> Result<Self, Qwen3TtsError> {
        unsafe {
            sys::ggml_backend_load_all();
            sys::ggml_cpu_init();
        }

        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        {
            if let Some(primary) = OwnedBackend::init_by_name(b"Metal\0")? {
                let primary_galloc = RefCell::new(OwnedGallocr::new(primary.as_ptr())?);
                return Ok(Self(Rc::new(BackendSetInner {
                    primary,
                    cpu_fallback: Some(OwnedBackend::cpu()?),
                    primary_galloc,
                })));
            }
        }

        let primary = OwnedBackend::cpu()?;
        let primary_galloc = RefCell::new(OwnedGallocr::new(primary.as_ptr())?);
        Ok(Self(Rc::new(BackendSetInner {
            primary,
            cpu_fallback: None,
            primary_galloc,
        })))
    }

    fn primary_ptr(&self) -> sys::ggml_backend_t {
        self.0.primary.as_ptr()
    }

    fn configure_threads(&self, thread_count: usize) {
        self.0.primary.set_threads(thread_count);
        if let Some(cpu_fallback) = &self.0.cpu_fallback {
            cpu_fallback.set_threads(thread_count);
        }
    }

    fn primary_galloc(&self) -> RefMut<'_, OwnedGallocr> {
        self.0.primary_galloc.borrow_mut()
    }
}

struct OwnedBackend {
    raw: NonNull<sys::ggml_backend>,
    is_cpu: bool,
}

impl OwnedBackend {
    fn cpu() -> Result<Self, Qwen3TtsError> {
        let raw = unsafe { sys::ggml_backend_cpu_init() };
        let raw = NonNull::new(raw)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to initialize ggml CPU backend".into()))?;
        Ok(Self { raw, is_cpu: true })
    }

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    fn init_by_name(name: &[u8]) -> Result<Option<Self>, Qwen3TtsError> {
        let raw = unsafe { sys::ggml_backend_init_by_name(name.as_ptr().cast(), std::ptr::null()) };
        Ok(NonNull::new(raw).map(|raw| Self { raw, is_cpu: false }))
    }

    fn as_ptr(&self) -> sys::ggml_backend_t {
        self.raw.as_ptr()
    }

    fn set_threads(&self, thread_count: usize) {
        if self.is_cpu {
            unsafe {
                sys::ggml_backend_cpu_set_n_threads(self.raw.as_ptr(), normalize_threads(thread_count));
            }
        }
    }
}

impl Drop for OwnedBackend {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_free(self.raw.as_ptr());
        }
    }
}

struct OwnedGallocr {
    raw: NonNull<sys::ggml_gallocr>,
}

impl OwnedGallocr {
    fn new(backend: sys::ggml_backend_t) -> Result<Self, Qwen3TtsError> {
        let raw = unsafe { sys::ggml_gallocr_new(sys::ggml_backend_get_default_buffer_type(backend)) };
        let raw = NonNull::new(raw)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to initialize ggml graph allocator".into()))?;
        Ok(Self { raw })
    }
}

impl Drop for OwnedGallocr {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_gallocr_free(self.raw.as_ptr());
        }
    }
}

struct TensorUpload<'a> {
    tensor: *mut sys::ggml_tensor,
    bytes: &'a [u8],
}

struct TensorDownload<'a> {
    tensor: *mut sys::ggml_tensor,
    bytes: &'a mut [u8],
}

struct OwnedBuffer {
    raw: NonNull<sys::ggml_backend_buffer>,
}

impl OwnedBuffer {
    fn alloc(ctx: *mut sys::ggml_context, backend: sys::ggml_backend_t) -> Result<Self, Qwen3TtsError> {
        let raw = unsafe { sys::ggml_backend_alloc_ctx_tensors(ctx, backend) };
        let raw = NonNull::new(raw)
            .ok_or_else(|| Qwen3TtsError::InvalidInput("failed to allocate ggml backend tensor buffer".into()))?;
        Ok(Self { raw })
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_backend_buffer_free(self.raw.as_ptr());
        }
    }
}

fn graph_metadata_mem_size(max_nodes: usize) -> usize {
    let tensor_overhead = unsafe { sys::ggml_tensor_overhead() };
    let graph_overhead = unsafe { sys::ggml_graph_overhead_custom(max_nodes, false) };
    max(1024 * 1024, graph_overhead + tensor_overhead * max_nodes * 16)
}

fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

fn slice_as_bytes_mut<T>(slice: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

fn execute_graph(
    backends: &BackendSet,
    graph: NonNull<sys::ggml_cgraph>,
    _max_nodes: usize,
    uploads: &[TensorUpload<'_>],
    downloads: &mut [TensorDownload<'_>],
    thread_count: usize,
    error_message: &str,
) -> Result<(), Qwen3TtsError> {
    maybe_log_backend_support(backends, graph, error_message);
    backends.configure_threads(thread_count);
    let galloc = backends.primary_galloc();
    let allocated = unsafe { sys::ggml_gallocr_alloc_graph(galloc.raw.as_ptr(), graph.as_ptr()) };
    if !allocated {
        return Err(Qwen3TtsError::InvalidInput(format!(
            "failed to allocate backend graph for {error_message}"
        )));
    }
    for upload in uploads {
        unsafe {
            sys::ggml_backend_tensor_set(upload.tensor, upload.bytes.as_ptr().cast(), 0, upload.bytes.len());
        }
    }
    let status = unsafe { sys::ggml_backend_graph_compute(backends.primary_ptr(), graph.as_ptr()) };
    if status != sys::ggml_status_GGML_STATUS_SUCCESS {
        return Err(Qwen3TtsError::InvalidInput(error_message.into()));
    }
    for download in downloads {
        unsafe {
            sys::ggml_backend_tensor_get(
                download.tensor,
                download.bytes.as_mut_ptr().cast(),
                0,
                download.bytes.len(),
            );
        }
    }
    Ok(())
}

fn maybe_log_backend_support(backends: &BackendSet, graph: NonNull<sys::ggml_cgraph>, label: &str) {
    if std::env::var_os("QWEN3_TTS_DEBUG_BACKEND").is_none() {
        return;
    }

    let n_nodes = unsafe { sys::ggml_graph_n_nodes(graph.as_ptr()) };
    let mut supported = 0usize;
    let mut offloaded = 0usize;
    let mut unsupported_ops = BTreeMap::<String, usize>::new();
    for idx in 0..n_nodes {
        let node = unsafe { sys::ggml_graph_node(graph.as_ptr(), idx) };
        if node.is_null() {
            continue;
        }
        let is_supported = unsafe { sys::ggml_backend_supports_op(backends.primary_ptr(), node) };
        let is_offloaded = unsafe { sys::ggml_backend_offload_op(backends.primary_ptr(), node) };
        if is_supported {
            supported += 1;
        } else {
            let op = unsafe {
                let desc = sys::ggml_op_desc(node);
                if desc.is_null() {
                    "<unknown>".to_string()
                } else {
                    std::ffi::CStr::from_ptr(desc).to_string_lossy().into_owned()
                }
            };
            *unsupported_ops.entry(op).or_default() += 1;
        }
        if is_offloaded {
            offloaded += 1;
        }
    }

    eprintln!(
        "[backend-debug] {label}: nodes={n_nodes} supported={supported} offloaded={offloaded}"
    );
    for (op, count) in unsupported_ops.into_iter().take(12) {
        eprintln!("[backend-debug] unsupported {op}: {count}");
    }
}

fn load_tensor_into_context(
    file: &GgufFile,
    ctx: *mut sys::ggml_context,
    name: &str,
) -> Result<NonNull<sys::ggml_tensor>, Qwen3TtsError> {
    let info = file.tensor_info(name)?;
    let mut ne = [1i64; 4];
    for (idx, dim) in info.dims.iter().copied().enumerate() {
        ne[idx] = dim as i64;
    }
    let tensor = unsafe { sys::ggml_new_tensor(ctx, info.ty, info.dims.len() as i32, ne.as_ptr()) };
    NonNull::new(tensor).ok_or_else(|| Qwen3TtsError::InvalidTensor(name.into()))
}

fn load_optional_tensor_into_context(
    file: &GgufFile,
    ctx: *mut sys::ggml_context,
    name: &str,
) -> Result<Option<NonNull<sys::ggml_tensor>>, Qwen3TtsError> {
    match file.tensor_info(name) {
        Ok(_) => load_tensor_into_context(file, ctx, name).map(Some),
        Err(Qwen3TtsError::MissingTensor(_)) => Ok(None),
        Err(err) => Err(err),
    }
}

fn upload_tensor(
    file: &GgufFile,
    name: &str,
    tensor: NonNull<sys::ggml_tensor>,
) -> Result<(), Qwen3TtsError> {
    let (_, raw) = file.read_tensor_bytes(name)?;
    unsafe {
        sys::ggml_backend_tensor_set(tensor.as_ptr(), raw.as_ptr().cast(), 0, raw.len());
    }
    Ok(())
}

fn upload_optional_tensor(
    file: &GgufFile,
    name: &str,
    tensor: Option<NonNull<sys::ggml_tensor>>,
) -> Result<(), Qwen3TtsError> {
    if let Some(tensor) = tensor {
        upload_tensor(file, name, tensor)?;
    }
    Ok(())
}

fn upload_named_tensor(
    file: &GgufFile,
    name: &str,
    tensors: &[(&str, Option<NonNull<sys::ggml_tensor>>)],
) -> Result<(), Qwen3TtsError> {
    if let Some((_, Some(tensor))) = tensors.iter().find(|(candidate, _)| *candidate == name) {
        upload_tensor(file, name, *tensor)?;
    }
    Ok(())
}

fn normalize_codebook(
    codebook: NonNull<sys::ggml_tensor>,
    usage: Option<NonNull<sys::ggml_tensor>>,
) -> Result<(), Qwen3TtsError> {
    let epsilon = 1e-5f32;
    let Some(usage) = usage else {
        return Ok(());
    };
    let codebook_dim = unsafe { (*codebook.as_ptr()).ne[0] as usize };
    let codebook_size = unsafe { (*codebook.as_ptr()).ne[1] as usize };
    let usage_data = unsafe { (*usage.as_ptr()).data.cast::<f32>() };
    if usage_data.is_null() {
        return Err(Qwen3TtsError::InvalidTensor("codebook usage has no data".into()));
    }

    unsafe {
        match (*codebook.as_ptr()).type_ {
            sys::ggml_type_GGML_TYPE_F16 => {
                let data = (*codebook.as_ptr()).data.cast::<sys::ggml_fp16_t>();
                if data.is_null() {
                    return Err(Qwen3TtsError::InvalidTensor("f16 codebook has no data".into()));
                }
                for emb_idx in 0..codebook_size {
                    let mut u = *usage_data.add(emb_idx);
                    if u < epsilon {
                        u = epsilon;
                    }
                    let inv_u = 1.0 / u;
                    for dim_idx in 0..codebook_dim {
                        let mem_idx = dim_idx + emb_idx * codebook_dim;
                        let val = sys::ggml_fp16_to_fp32(*data.add(mem_idx));
                        *data.add(mem_idx) = sys::ggml_fp32_to_fp16(val * inv_u);
                    }
                }
            }
            sys::ggml_type_GGML_TYPE_F32 => {
                let data = (*codebook.as_ptr()).data.cast::<f32>();
                if data.is_null() {
                    return Err(Qwen3TtsError::InvalidTensor("f32 codebook has no data".into()));
                }
                for emb_idx in 0..codebook_size {
                    let mut u = *usage_data.add(emb_idx);
                    if u < epsilon {
                        u = epsilon;
                    }
                    let inv_u = 1.0 / u;
                    for dim_idx in 0..codebook_dim {
                        let mem_idx = dim_idx + emb_idx * codebook_dim;
                        *data.add(mem_idx) *= inv_u;
                    }
                }
            }
            _ => {
                return Err(Qwen3TtsError::UnsupportedTensorType(
                    "tok_dec codebook normalization".into(),
                ))
            }
        }
    }

    Ok(())
}

fn normalize_threads(thread_count: usize) -> i32 {
    max(1, thread_count) as i32
}
