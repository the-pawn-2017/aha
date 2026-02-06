use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, pickle::read_all_with_key};
use candle_nn::{
    Activation, Conv1d, Embedding, GroupNorm, Init, LayerNorm, Linear, Module, RmsNorm, VarBuilder,
    embedding, group_norm, linear, linear_b, ops::sigmoid, rms_norm,
};

use crate::{
    models::{
        campplus::CAMPPlus,
        common::{
            GateUpDownMLP, QKVCatAttention, TwoLinearMLP, WNConv1d, WNLinear, get_conv1d,
            get_layer_norm, get_layer_norm_without_weight, mish,
        },
        feature_extractor::seamless_m4t_feature_extractor::SeamlessM4TFeatureExtractor,
        index_tts2::config::{DiTModelArgs, IndexTTS2Config, PreprocessParams, S2MelConfig},
        mask_gct::model::RepCodec,
        w2v_bert_2_0::model::W2VBert2_0Model,
    },
    position_embed::rope::RoPE,
    utils::{
        audio_utils::{
            create_hann_window, get_waveform_and_window_properties, kaldi_fbank,
            kaldi_get_mel_banks, mel_filter_bank, torch_stft,
        },
        get_vb_model_path, read_pth_tensor_info_cycle,
        tensor_utils::{
            interpolate_nearest_1d, pad_reflect_last_dim, sequence_mask, split_tensor_with_size,
        },
    },
};
pub struct AdaptiveLayerNorm {
    project_layer: Linear,
    norm: RmsNorm,
    d_model: usize,
}

impl AdaptiveLayerNorm {
    pub fn new(vb: VarBuilder, d_model: usize, eps: f64) -> Result<Self> {
        let project_layer = linear(d_model, d_model * 2, vb.pp("project_layer"))?;
        let norm = rms_norm(d_model, eps, vb.pp("norm"))?;
        Ok(Self {
            project_layer,
            norm,
            d_model,
        })
    }

    pub fn forward(&self, xs: &Tensor, embedding: Option<&Tensor>) -> Result<Tensor> {
        if let Some(embedding) = embedding {
            let emb = self.project_layer.forward(embedding)?;
            let emb_split = split_tensor_with_size(&emb, 2, D::Minus1)?;
            let weight = &emb_split[0];
            let bias = &emb_split[1];
            Ok(self
                .norm
                .forward(xs)?
                .broadcast_mul(weight)?
                .broadcast_add(bias)?)
        } else {
            Ok(self.norm.forward(xs)?)
        }
    }
}

pub struct DiTTransformerBlock {
    attention: QKVCatAttention,
    feed_forward: GateUpDownMLP,
    ffn_norm: AdaptiveLayerNorm,
    attention_norm: AdaptiveLayerNorm,
    skip_in_linear: Option<Linear>,
    uvit_skip_connection: bool,
    time_as_token: bool,
}

impl DiTTransformerBlock {
    pub fn new(vb: VarBuilder, config: &DiTModelArgs) -> Result<Self> {
        let attention = QKVCatAttention::new(
            vb.pp("attention"),
            config.dim,
            config.n_head,
            Some(config.head_dim),
            false,
            Some("wqkv"),
            Some("wo"),
        )?;

        let feed_forward = GateUpDownMLP::new(
            vb.pp("feed_forward"),
            config.dim,
            config.intermediate_size,
            candle_nn::Activation::Silu,
            false,
            Some("w1"),
            Some("w3"),
            Some("w2"),
        )?;
        let ffn_norm = AdaptiveLayerNorm::new(vb.pp("ffn_norm"), config.dim, config.norm_eps)?;
        let attention_norm =
            AdaptiveLayerNorm::new(vb.pp("attention_norm"), config.dim, config.norm_eps)?;
        let (skip_in_linear, uvit_skip_connection) = if config.uvit_skip_connection {
            let skip_in_linear = linear(config.dim * 2, config.dim, vb.pp("skip_in_linear"))?;
            (Some(skip_in_linear), config.uvit_skip_connection)
        } else {
            (None, false)
        };
        Ok(Self {
            attention,
            feed_forward,
            ffn_norm,
            attention_norm,
            skip_in_linear,
            uvit_skip_connection,
            time_as_token: config.time_as_token,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        c: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        mask: Option<&Tensor>,
        skip_in_x: Option<&Tensor>,
    ) -> Result<Tensor> {
        let c = if self.time_as_token { None } else { Some(c) };
        let mut xs = xs.clone();
        if self.uvit_skip_connection
            && let Some(skip_in_x) = skip_in_x
            && let Some(skip_in_linear) = &self.skip_in_linear
        {
            let cat = Tensor::cat(&[&xs, skip_in_x], D::Minus1)?;
            xs = skip_in_linear.forward(&cat)?;
        }
        let xs = self
            .attention
            .forward(
                &self.attention_norm.forward(&xs, c)?,
                cos,
                sin,
                mask,
                false,
                true,
            )?
            .add(&xs)?;
        let out = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&xs, c)?)?
            .add(&xs)?;
        Ok(out)
    }
}

pub struct DiTTransformer {
    layers: Vec<DiTTransformerBlock>,
    norm: AdaptiveLayerNorm,
    rope: RoPE,
    uvit_skip_connection: bool,
    layers_emit_skip: Vec<usize>,
    layers_receive_skip: Vec<usize>,
}

impl DiTTransformer {
    pub fn new(vb: VarBuilder, config: &DiTModelArgs) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        let mut layers = vec![];
        for i in 0..config.n_layer {
            let layer = DiTTransformerBlock::new(vb_layers.pp(i), config)?;
            layers.push(layer);
        }
        let norm = AdaptiveLayerNorm::new(vb.pp("norm"), config.dim, config.norm_eps)?;
        let rope = RoPE::new(config.dim, 10000.0, vb.device())?;
        let mut layers_emit_skip: Vec<usize> = vec![];
        let mut layers_receive_skip: Vec<usize> = vec![];
        if config.uvit_skip_connection {
            layers_emit_skip = (0..config.n_layer)
                .filter(|&x| x < config.n_layer / 2)
                .collect();
            layers_receive_skip = (0..config.n_layer)
                .filter(|&x| x > config.n_layer / 2)
                .collect();
        }
        Ok(Self {
            layers,
            norm,
            rope,
            uvit_skip_connection: config.uvit_skip_connection,
            layers_emit_skip,
            layers_receive_skip,
        })
    }
    pub fn forward(&self, xs: &Tensor, c: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;
        let (cos, sin) = self.rope.forward(0, seq_len, xs.device())?;
        let mut skip_in_x_list = vec![];
        let mut xs = xs.clone();
        for (i, layer) in (&self.layers).iter().enumerate() {
            let skip_in_x = if self.uvit_skip_connection && self.layers_receive_skip.contains(&i) {
                skip_in_x_list.pop()
            } else {
                None
            };
            xs = layer.forward(&xs, c, Some(&cos), Some(&sin), mask, skip_in_x.as_ref())?;
            if self.uvit_skip_connection && self.layers_emit_skip.contains(&i) {
                skip_in_x_list.push(xs.clone());
            }
        }
        xs = self.norm.forward(&xs, Some(c))?;
        Ok(xs)
    }
}

pub struct TimestepEmbedder {
    mlp: TwoLinearMLP,
    freqs: Tensor,
    scale: f64,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        frequency_embedding_size: usize,
    ) -> Result<Self> {
        let mlp = TwoLinearMLP::new(
            vb.pp("mlp"),
            frequency_embedding_size,
            hidden_size,
            hidden_size,
            candle_nn::Activation::Silu,
            true,
            "0",
            "2",
        )?;
        let scale = 1000.0;
        let half = frequency_embedding_size / 2;
        let freqs = Tensor::arange(0f32, half as f32, vb.device())?
            .affine(-(10000.0f64.ln()), 0.0)?
            .exp()?;
        Ok(Self {
            mlp,
            freqs,
            scale,
            frequency_embedding_size,
        })
    }
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let args = t
            .affine(self.scale, 0.0)?
            .unsqueeze(D::Minus1)?
            .broadcast_matmul(&self.freqs.unsqueeze(0)?)?;
        let mut embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        if self.frequency_embedding_size % 2 > 0 {
            embedding = embedding.pad_with_zeros(D::Minus1, 0, 1)?;
        }
        embedding = self.mlp.forward(&embedding)?;
        Ok(embedding)
    }
}

pub struct SConv1d {
    conv: WNConv1d,
    ks: usize,
    stride: usize,
    dilation: usize,
}

impl SConv1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        ks: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv = WNConv1d::new(
            vb.pp("conv.conv"),
            in_c,
            out_c,
            ks,
            dilation,
            0,
            groups,
            stride,
            bias,
        )?;
        Ok(Self {
            conv,
            ks,
            stride,
            dilation,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let length = xs.dim(D::Minus1)?;
        let ks = (self.ks - 1) * self.dilation + 1;
        let padding_total = ks - self.stride;
        let n_frames = (length - ks + padding_total) as f32 / self.stride as f32 + 1.0;
        let idea_length = (n_frames.ceil() as usize - 1) * self.stride + (ks - padding_total);
        let extra_padding = idea_length - length;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let xs = pad_reflect_last_dim(xs, (padding_left, padding_right + extra_padding))?;
        let xs = self.conv.forward(&xs)?;
        Ok(xs)
    }
}

pub struct Wavenet {
    cond_layer: Option<SConv1d>,
    in_layers: Vec<SConv1d>,
    res_skip_layers: Vec<SConv1d>,
    hidden_c: usize,
    n_layers: usize,
}

impl Wavenet {
    pub fn new(
        vb: VarBuilder,
        hidden_c: usize,
        ks: usize,
        dilation_rate: usize,
        n_layers: usize,
        gin_channels: usize,
    ) -> Result<Self> {
        let cond_layer = if gin_channels != 0 {
            Some(SConv1d::new(
                vb.pp("cond_layer"),
                gin_channels,
                2 * hidden_c * n_layers,
                1,
                1,
                1,
                1,
                true,
            )?)
        } else {
            None
        };
        let mut in_layers = vec![];
        let vb_layers = vb.pp("in_layers");
        let mut res_skip_layers = vec![];
        let vb_res_skip_layers = vb.pp("res_skip_layers");
        for i in 0..n_layers {
            let dilation = dilation_rate.pow(i as u32);
            let in_layer = SConv1d::new(
                vb_layers.pp(i),
                hidden_c,
                2 * hidden_c,
                ks,
                1,
                dilation,
                1,
                true,
            )?;
            in_layers.push(in_layer);
            let res_skip_c = if i < n_layers - 1 {
                2 * hidden_c
            } else {
                hidden_c
            };
            let res_skip_layer = SConv1d::new(
                vb_res_skip_layers.pp(i),
                hidden_c,
                res_skip_c,
                1,
                1,
                1,
                1,
                true,
            )?;
            res_skip_layers.push(res_skip_layer);
        }
        Ok(Self {
            cond_layer,
            in_layers,
            res_skip_layers,
            hidden_c,
            n_layers,
        })
    }

    pub fn fused_add_tanh_sigmoid_multiply(
        &self,
        input_a: &Tensor,
        input_b: &Tensor,
    ) -> Result<Tensor> {
        let in_act = input_a.add(&input_b)?;
        let parts = split_tensor_with_size(&in_act, 2, 1)?;
        let t_act = (&parts[0]).tanh()?;
        let s_act = sigmoid(&parts[1])?;
        let acts = t_act.mul(&s_act)?;
        Ok(acts)
    }

    pub fn forward(&self, xs: &Tensor, x_mask: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let mut output = xs.zeros_like()?;
        let g = if let Some(g) = g
            && let Some(cond_layer) = &self.cond_layer
        {
            Some(cond_layer.forward(g)?)
        } else {
            None
        };
        let mut xs = xs.clone();
        for i in 0..self.n_layers {
            let xs_in = &self.in_layers[i].forward(&xs)?;
            let g_l = if let Some(g) = &g {
                let cond_offset = i * 2 * self.hidden_c;
                g.narrow(1, cond_offset, 2 * self.hidden_c)?
            } else {
                xs_in.zeros_like()?
            };
            let acts = self.fused_add_tanh_sigmoid_multiply(&xs_in, &g_l)?;
            let res_skip_act = &self.res_skip_layers[i].forward(&acts)?;
            if i < self.n_layers - 1 {
                let res_acts = res_skip_act.narrow(1, 0, self.hidden_c)?;
                let out_acts = res_skip_act.narrow(1, self.hidden_c, self.hidden_c)?;
                xs = xs.add(&res_acts)?.mul(x_mask)?;
                output = output.add(&out_acts)?;
            } else {
                output = output.add(&res_skip_act)?;
            }
        }
        output = output.mul(x_mask)?;
        Ok(output)
    }
}

pub struct FinalLayer {
    norm_final: LayerNorm,
    linear: WNLinear,
    ada_ln_modulation: Linear, // (silu+linear)
}

impl FinalLayer {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        patch_size: usize,
        out_c: usize,
    ) -> Result<Self> {
        let norm_final = get_layer_norm_without_weight(vb.pp("norm_final"), 1e-6, hidden_size)?;
        let linear = WNLinear::new(
            vb.pp("linear"),
            hidden_size,
            patch_size * patch_size * out_c,
            true,
        )?;
        let ada_ln_modulation = linear_b(
            hidden_size,
            2 * hidden_size,
            true,
            vb.pp("adaLN_modulation.1"),
        )?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    pub fn forward(&self, xs: &Tensor, c: &Tensor) -> Result<Tensor> {
        let linear_c = self.ada_ln_modulation.forward(c)?.chunk(2, 1)?;
        let xs = self.norm_final.forward(xs)?;
        let xs = linear_c[1]
            .unsqueeze(1)?
            .affine(1.0, 1.0)?
            .broadcast_mul(&xs)?
            .add(&linear_c[0].unsqueeze(1)?)?;
        let xs = self.linear.forward(&xs)?;
        Ok(xs)
    }
}

pub struct DiT {
    transformer: DiTTransformer,
    x_embedder: WNLinear,
    cond_embedder: Embedding,
    cond_projection: Linear,
    t_embedder: TimestepEmbedder,
    input_pos: Tensor,
    t_embedder2: TimestepEmbedder,
    conv1: Linear,
    conv2: Conv1d,
    wavenet: Wavenet,
    final_layer: FinalLayer,
    res_projection: Linear,
    content_mask_embedder: Embedding,
    skip_linear: Linear,
    cond_x_merge_linear: Linear,
    style_in: Option<Linear>,
    time_as_token: bool,
    style_as_token: bool,
    uvit_skip_connection: bool,
    transformer_style_condition: bool,
    long_skip_connection: bool,
}

impl DiT {
    pub fn new(vb: VarBuilder, config: &S2MelConfig) -> Result<Self> {
        let time_as_token = config.di_t.time_as_token;
        let style_as_token = config.di_t.style_as_token;
        let uvit_skip_connection = config.di_t.uvit_skip_connection;
        let transformer_config = DiTModelArgs::new_from_dit_config(&config.di_t);
        let transformer = DiTTransformer::new(vb.pp("transformer"), &transformer_config)?;
        let x_embedder = WNLinear::new(
            vb.pp("x_embedder"),
            config.di_t.in_channels,
            config.di_t.hidden_dim,
            true,
        )?;
        let cond_embedder = embedding(
            config.di_t.content_codebook_size,
            config.di_t.hidden_dim,
            vb.pp("cond_embedder"),
        )?;
        let cond_projection = linear_b(
            config.di_t.content_dim,
            config.di_t.hidden_dim,
            true,
            vb.pp("cond_projection"),
        )?;
        let t_embedder = TimestepEmbedder::new(vb.pp("t_embedder"), config.di_t.hidden_dim, 256)?;
        let input_pos = Tensor::arange(0u32, 16384, vb.device())?;
        let t_embedder2 =
            TimestepEmbedder::new(vb.pp("t_embedder2"), config.wavenet.hidden_dim, 256)?;
        let conv1 = linear_b(
            config.di_t.hidden_dim,
            config.wavenet.hidden_dim,
            true,
            vb.pp("conv1"),
        )?;
        let conv2 = get_conv1d(
            vb.pp("conv2"),
            config.wavenet.hidden_dim,
            config.di_t.in_channels,
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let wavenet = Wavenet::new(
            vb.pp("wavenet"),
            config.wavenet.hidden_dim,
            config.wavenet.kernel_size,
            config.wavenet.dilation_rate,
            config.wavenet.num_layers,
            config.wavenet.hidden_dim,
        )?;
        let final_layer = FinalLayer::new(
            vb.pp("final_layer"),
            config.wavenet.hidden_dim,
            1,
            config.wavenet.hidden_dim,
        )?;
        let res_projection = linear(
            config.di_t.hidden_dim,
            config.wavenet.hidden_dim,
            vb.pp("res_projection"),
        )?;
        let content_mask_embedder =
            embedding(1, config.di_t.hidden_dim, vb.pp("content_mask_embedder"))?;
        let skip_linear = linear(
            config.di_t.hidden_dim + config.di_t.in_channels,
            config.di_t.hidden_dim,
            vb.pp("skip_linear"),
        )?;
        let in_dim = if config.di_t.style_condition && !config.di_t.style_as_token {
            config.di_t.hidden_dim + config.di_t.in_channels * 2 + config.style_encoder.dim
        } else {
            config.di_t.hidden_dim + config.di_t.in_channels * 2
        };
        let cond_x_merge_linear =
            linear(in_dim, config.di_t.hidden_dim, vb.pp("cond_x_merge_linear"))?;
        let style_in = if config.di_t.style_as_token {
            Some(linear(
                config.style_encoder.dim,
                config.di_t.hidden_dim,
                vb.pp("style_in"),
            )?)
        } else {
            None
        };
        Ok(Self {
            transformer,
            x_embedder,
            cond_embedder,
            cond_projection,
            t_embedder,
            input_pos,
            t_embedder2,
            conv1,
            conv2,
            wavenet,
            final_layer,
            res_projection,
            content_mask_embedder,
            skip_linear,
            cond_x_merge_linear,
            style_in,
            time_as_token,
            style_as_token,
            uvit_skip_connection,
            transformer_style_condition: config.di_t.style_condition,
            long_skip_connection: config.di_t.long_skip_connection,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        prompt_xs: &Tensor,
        x_lens: &Tensor,
        t: &Tensor,
        style: Option<&Tensor>,
        cond: &Tensor,
    ) -> Result<Tensor> {
        let (_, _, t_dim) = xs.dims3()?;
        let t1 = self.t_embedder.forward(t)?;
        let cond = self.cond_projection.forward(cond)?;
        let xs = xs.transpose(1, 2)?;
        let prompt_xs = prompt_xs.transpose(1, 2)?;
        let mut x_in = Tensor::cat(&[&xs, &prompt_xs, &cond], D::Minus1)?;
        if self.transformer_style_condition
            && !self.style_as_token
            && let Some(style) = style
        {
            let style = style.unsqueeze(1)?.repeat((1, t_dim, 1))?;
            x_in = Tensor::cat(&[&x_in, &style], D::Minus1)?;
        }
        x_in = self.cond_x_merge_linear.forward(&x_in)?;
        // if self.style_as_token
        //     && let Some(style_in) = self.style_in.as_ref()
        // {
        //     let style = style_in.forward(style)?.unsqueeze(1)?;
        //     x_in = Tensor::cat(&[&style, &x_in], 1)?;
        // }
        // if self.time_as_token {
        //     let t1 = t1.unsqueeze(1)?;
        //     x_in = Tensor::cat(&[&t1, &x_in], 1)?;
        // }
        // let mut x_lens = x_lens.clone();
        // if self.style_as_token {
        //     x_lens = x_lens.affine(1.0, 1.0)?;
        // }
        // if self.time_as_token {
        //     x_lens = x_lens.affine(1.0, 0.0)?;
        // }
        let x_mask = sequence_mask(&x_lens, Some(x_in.dim(1)? as u32))?
            .to_device(xs.device())?
            .unsqueeze(1)?;
        let mut x_res = self.transformer.forward(&x_in, &t1.unsqueeze(1)?, None)?;
        // if self.time_as_token {
        //     let last_dim = x_res.dim(D::Minus1)?;
        //     x_res = x_res.narrow(D::Minus1, 1, last_dim - 1)?;
        // }
        // if self.style_as_token {
        //     let last_dim = x_res.dim(D::Minus1)?;
        //     x_res = x_res.narrow(D::Minus1, 1, last_dim - 1)?;
        // }
        if self.long_skip_connection {
            x_res = self
                .skip_linear
                .forward(&Tensor::cat(&[&x_res, &xs], D::Minus1)?)?;
        }
        let xs = self.conv1.forward(&x_res)?;
        let xs = xs.transpose(1, 2)?;
        let t2 = self.t_embedder2.forward(t)?;
        let xs = self
            .wavenet
            .forward(&xs, &x_mask, Some(&t2.unsqueeze(2)?))?
            .transpose(1, 2)?
            .broadcast_add(&self.res_projection.forward(&x_res)?)?;
        let xs = self.final_layer.forward(&xs, &t1)?.transpose(1, 2)?;
        let xs = self.conv2.forward(&xs)?;
        Ok(xs)
    }
}

pub struct CFM {
    in_channels: usize,
    estimator: DiT,
    // criterion: l1Loss
    sigma_min: f32,
}

impl CFM {
    pub fn new(vb: VarBuilder, config: &S2MelConfig) -> Result<Self> {
        let in_channels = config.di_t.in_channels;
        let sigma_min = 1e-6;
        let estimator = DiT::new(vb.pp("estimator"), config)?;
        Ok(Self {
            in_channels,
            estimator,
            sigma_min,
        })
    }
}

pub struct InterpolateModule {
    conv1d: Conv1d,
    norm: GroupNorm,
}

impl InterpolateModule {
    pub fn new(vb: &VarBuilder, index: usize, channels: usize, groups: usize) -> Result<Self> {
        let start_index = index * 3;
        let conv1d = get_conv1d(vb.pp(start_index), channels, channels, 3, 1, 1, 1, 1, true)?;
        let norm = group_norm(groups, channels, 1e-5, vb.pp(start_index + 1))?;
        Ok(Self { conv1d, norm })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv1d.forward(xs)?;
        let xs = self.norm.forward(&xs)?;
        let xs = mish(&xs)?;
        Ok(xs)
    }
}

pub struct InterpolateRegulator {
    sampling_ratios: Vec<usize>,
    out_channels: usize,
    model0_11: Vec<InterpolateModule>,
    model_12: Conv1d,
    embedding: Embedding,
    mask_token: Tensor,
    quantizer_dropout: f32,
    content_in_proj: Linear,
    n_codebooks: usize,
    interpolate: bool,
}

impl InterpolateRegulator {
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        sampling_ratios: Vec<usize>,
        is_discrete: bool,
        in_channels: usize,
        vector_quantize: bool,
        codebook_size: usize,
        out_channels: Option<usize>,
        groups: usize,
        n_codebooks: usize,
        quantizer_dropout: f32,
        f0_condition: bool,
        n_f0_bins: usize,
    ) -> Result<Self> {
        let out_channels = out_channels.unwrap_or(channels);
        let vb_model = vb.pp("model");
        let interpolate = true;
        let mut model0_11 = vec![];
        for (index, _) in sampling_ratios.iter().enumerate() {
            let inter = InterpolateModule::new(&vb_model, index, channels, groups)?;
            model0_11.push(inter);
        }
        let model_12 = get_conv1d(
            vb_model.pp("12"),
            channels,
            out_channels,
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let embedding = embedding(codebook_size, channels, vb.pp("embedding"))?;
        let mask_token = vb.get_with_hints((1, channels), "mask_token", Init::Const(0.0))?;
        let content_in_proj = linear(in_channels, channels, vb.pp("content_in_proj"))?;
        Ok(Self {
            sampling_ratios,
            out_channels,
            model0_11,
            model_12,
            embedding,
            mask_token,
            quantizer_dropout,
            content_in_proj,
            n_codebooks,
            interpolate,
        })
    }
    pub fn forward(&self, x: &Tensor, y_lens: &Tensor) -> Result<Tensor> {
        let mut xs = self.content_in_proj.forward(x)?;
        xs = xs.transpose(1, 2)?.contiguous()?;
        if self.interpolate {
            let size = y_lens.max_all()?.to_scalar::<u32>()? as usize;
            xs = interpolate_nearest_1d(&xs, size)?;
        }
        for model_i in self.model0_11.iter() {
            xs = model_i.forward(&xs)?;
        }
        xs = self.model_12.forward(&xs)?.transpose(1, 2)?.contiguous()?;
        Ok(xs)
    }
}

pub struct MyModel {
    cfm: CFM,
    length_regulator: InterpolateRegulator,
}

impl MyModel {
    pub fn new(
        model_path: &str,
        config: &S2MelConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let s2mel_path = model_path.to_string() + "/s2mel.pth";
        let length_regulator_dict =
            read_pth_tensor_info_cycle(s2mel_path.clone(), Some("net.length_regulator"))?;
        let length_regulator_vb = VarBuilder::from_tensors(length_regulator_dict, dtype, device);
        let length_regulator = InterpolateRegulator::new(
            length_regulator_vb,
            config.length_regulator.channels,
            config.length_regulator.sampling_ratios.clone(),
            config.length_regulator.is_discrete,
            config.length_regulator.in_channels,
            config.length_regulator.vector_quantize,
            config.length_regulator.content_codebook_size,
            None,
            1,
            config.length_regulator.n_codebooks,
            config.length_regulator.quantizer_dropout,
            config.length_regulator.f0_condition,
            config.length_regulator.n_f0_bins,
        )?;
        let cfm_dict = read_pth_tensor_info_cycle(s2mel_path.clone(), Some("net.cfm"))?;
        let cfm_vb = VarBuilder::from_tensors(cfm_dict, dtype, device);
        let cfm = CFM::new(cfm_vb, config)?;
        Ok(Self {
            cfm,
            length_regulator,
        })
    }

    pub fn length_regulator_forward(
        &self,
        s_ori: &Tensor,
        target_lengths: &Tensor,
    ) -> Result<Tensor> {
        let xs = self.length_regulator.forward(s_ori, target_lengths)?;
        Ok(xs)
    }
}

pub struct IndexTTS2Cache {
    pub cache_spk_cond: Tensor,
    pub cache_s2mel_style: Tensor,
    pub cache_s2mel_prompt: Tensor,
    pub cache_mel: Tensor,
    // pub cache_emo_cond: Tensor,
    // pub cache_emo_audio_prompt: Tensor,
}

pub struct IndexTTS2Model {
    cache: Option<IndexTTS2Cache>,
    feature_extractor: SeamlessM4TFeatureExtractor,
    semantic_model: W2VBert2_0Model,
    semantic_mean: Tensor,
    semantic_std: Tensor,
    semantic_codec: RepCodec,
    s2mel_filters: Tensor,
    s2mel_windows: Tensor,
    s2mel_preprocess_params: PreprocessParams,
    window_shift: usize,
    window_size: usize,
    padded_window_size: usize,
    mel_energies: Tensor,
    campplus_model: CAMPPlus,
    s2mel: MyModel,
}

impl IndexTTS2Model {
    pub fn new(
        path: &str,
        save_dir: &str,
        config: &IndexTTS2Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let feature_extractor = SeamlessM4TFeatureExtractor::new(
            // 80,
            80,
            crate::utils::tensor_utils::PaddingSide::Right,
            1.0,
            16000,
            2,
            device,
        )?;
        let w2vbert2_path = save_dir.to_string() + "/facebook/w2v-bert-2.0";
        let semantic_model = W2VBert2_0Model::init(&w2vbert2_path, device, dtype)?;
        let semantic_mean_var_path = path.to_string() + "/" + &config.w2v_stat;
        let dict = read_all_with_key(semantic_mean_var_path, None)?;
        let mut semantic_mean = Tensor::new(0.0, device)?.to_dtype(dtype)?;
        let mut semantic_std = Tensor::new(1.0, device)?.to_dtype(dtype)?;
        for (k, v) in dict {
            if k.eq("mean") {
                semantic_mean = v.to_device(device)?.to_dtype(dtype)?;
            } else if k.eq("var") {
                semantic_std = v.to_device(device)?.to_dtype(dtype)?.sqrt()?;
            }
        }

        let semantic_codec_path =
            save_dir.to_string() + "/amphion/MaskGCT/semantic_codec/model.safetensors";
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[semantic_codec_path], dtype, &device)? };
        let semantic_codec = RepCodec::new(vb, &config.semantic_codec)?;
        let s2mel_filters = mel_filter_bank(
            config.s2mel.preprocess_params.spect_params.n_fft / 2 + 1,
            config.s2mel.preprocess_params.spect_params.n_mels,
            config.s2mel.preprocess_params.spect_params.fmin as f32,
            config
                .s2mel
                .preprocess_params
                .spect_params
                .fmax
                .unwrap_or(config.s2mel.preprocess_params.sr / 2) as f32,
            config.s2mel.preprocess_params.sr as f32,
            Some("slaney"),
            crate::utils::audio_utils::MelScale::Slaney,
            false,
            device,
        )?
        .t()?;
        let s2mel_windows = create_hann_window(
            config.s2mel.preprocess_params.spect_params.win_length,
            dtype,
            device,
        )?;
        let (window_shift, window_size, padded_window_size) =
            get_waveform_and_window_properties(16000, 10.0, 25.0, true)?;
        let (mel_energies, _) =
            kaldi_get_mel_banks(80, padded_window_size, 16000 as f32, 20.0, 0.0, device)?;
        let mel_energies = mel_energies.pad_with_zeros(D::Minus1, 0, 1)?.t()?;
        let campplus_model_path = save_dir.to_string()
            + "/iic/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin";
        let campplus_vb = get_vb_model_path(campplus_model_path, dtype, device.clone(), None)?;
        let campplus_model = CAMPPlus::new(campplus_vb, 80, 192, 32, 4, 128)?;
        let s2mel = MyModel::new(path, &config.s2mel, dtype, device)?;
        Ok(Self {
            cache: None,
            feature_extractor,
            semantic_model,
            semantic_mean,
            semantic_std,
            semantic_codec,
            s2mel_filters,
            s2mel_windows,
            s2mel_preprocess_params: config.s2mel.preprocess_params.clone(),
            window_shift,
            window_size,
            padded_window_size,
            mel_energies,
            campplus_model,
            s2mel,
        })
    }

    pub fn get_emb(
        &self,
        input_features: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let output =
            self.semantic_model
                .forward(input_features, attention_mask, Some(17), false)?;
        let feature = &output.specify_layer_id_hidden_state.unwrap();
        let feature = feature
            .broadcast_sub(&self.semantic_mean)?
            .broadcast_div(&self.semantic_std)?;
        Ok(feature)
    }

    pub fn s2mel_spectrogram(&self, waveform: &Tensor) -> Result<Tensor> {
        let pad = (self.s2mel_preprocess_params.spect_params.n_fft
            - self.s2mel_preprocess_params.spect_params.hop_length)
            / 2;
        let pad_audio_22k = pad_reflect_last_dim(&waveform, (pad, pad))?;
        let spec = torch_stft(
            &pad_audio_22k,
            self.s2mel_preprocess_params.spect_params.n_fft,
            self.s2mel_preprocess_params.spect_params.hop_length,
            &self.s2mel_windows,
        )?
        .transpose(1, 2)?;
        let spec = self.s2mel_filters.broadcast_matmul(&spec)?;
        let spec = spec.clamp(1e-5, f64::INFINITY)?.log()?;
        Ok(spec)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        audio_22k: Option<&Tensor>,
        audio_16k: Option<&Tensor>,
    ) -> Result<()> {
        if (audio_22k.is_none() || audio_16k.is_none()) && self.cache.is_none() {
            return Err(anyhow!(
                "Missing required audio input: must provide either audio_22k, audio_16k, or have cached prompt data available"
            ));
        }
        let (spk_cond_emb, style, prompt_condition, ref_mel) = if let Some(audio_22k) = audio_22k
            && let Some(audio_16k) = audio_16k
        {
            let (audio_16k_features, audio_16k_mask) =
                self.feature_extractor.call(&audio_16k, 16000, true, true)?;
            let spk_cond_emb = self.get_emb(&audio_16k_features, audio_16k_mask.as_ref())?;
            let (_, s_ref) = self.semantic_codec.quantize(&spk_cond_emb)?;
            let ref_mel = self.s2mel_spectrogram(&audio_22k)?;
            let ref_target_lengths = Tensor::new(ref_mel.dim(2)? as u32, ref_mel.device())?;
            let feat = kaldi_fbank(
                &audio_16k,
                &self.mel_energies,
                self.window_shift,
                self.window_size,
                self.padded_window_size,
                0.0,
            )?;
            let feat = feat.broadcast_sub(&feat.mean_keepdim(1)?)?;
            let style = self.campplus_model.forward(&feat)?;
            let prompt_condition = self
                .s2mel
                .length_regulator
                .forward(&s_ref, &ref_target_lengths)?;
            let cache = IndexTTS2Cache {
                cache_spk_cond: spk_cond_emb.clone(),
                cache_s2mel_style: style.clone(),
                cache_s2mel_prompt: prompt_condition.clone(),
                cache_mel: ref_mel.clone(),
            };
            self.cache = Some(cache);
            (spk_cond_emb, style, prompt_condition, ref_mel)
        } else {
            let cache = self.cache.as_ref().unwrap();
            (
                cache.cache_spk_cond.clone(),
                cache.cache_s2mel_style.clone(),
                cache.cache_s2mel_prompt.clone(),
                cache.cache_mel.clone(),
            )
        };
        
        
        Ok(())
    }
}
