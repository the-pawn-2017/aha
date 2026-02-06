use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use sentencepiece::SentencePieceProcessor;

use crate::{
    models::index_tts2::{
        config::IndexTTS2Config,
        model::IndexTTS2Model,
        processor::IndexTTS2Processor,
        utils::{TextNormalizer, tokenize_by_cjk_char},
    },
    tokenizer::sentencepiece_encode,
    utils::{
        audio_utils::extract_audio_url, extract_user_text, get_default_save_dir, get_device,
        get_dtype,
    },
};

pub struct IndexTTS2Generate {
    processor: IndexTTS2Processor,
    tokenizer: SentencePieceProcessor,
    config: IndexTTS2Config,
    cache_spk_audio_prompt: Option<String>,
    model: IndexTTS2Model,
    device: Device,
}

impl IndexTTS2Generate {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let config_path = path.to_string() + "/config.yaml";
        let save_dir = get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
        let config: IndexTTS2Config = serde_yaml::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, "bf16");
        let processor = IndexTTS2Processor::new(&device)?;
        let bpe_path = path.to_string() + "/bpe.model";
        let tokenizer = SentencePieceProcessor::open(bpe_path)
            .map_err(|e| anyhow!(format!("load bpe,model file error:{}", e)))?;
        let model = IndexTTS2Model::new(path, &save_dir, &config, &device, dtype)?;
        Ok(Self {
            processor,
            tokenizer,
            config,
            cache_spk_audio_prompt: None,
            model,
            device,
        })
    }

    pub fn use_prompt(&self, mes: &ChatCompletionParameters) -> bool {
        if let Some(cache) = &self.cache_spk_audio_prompt {
            let audio_vec = extract_audio_url(mes);
            if audio_vec.len() == 0 {
                true
            } else {
                if cache.eq(&audio_vec[0]) { true } else { false }
            }
        } else {
            false
        }
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters) -> Result<()> {
        let text = extract_user_text(&mes)?;
        let text = tokenize_by_cjk_char(&text, true);
        let input_ids = sentencepiece_encode(&text, &self.tokenizer, &self.device)?;
        let (audio_22k, audio_16k) = if self.use_prompt(&mes) {
            (None, None)
        } else {
            let (audio_22k, audio_16k, prompt) = self.processor.process_info(&mes)?;
            self.cache_spk_audio_prompt = Some(prompt);
            (Some(audio_22k), Some(audio_16k))
        };
        let _ = self.model.forward(&input_ids, audio_22k.as_ref(), audio_16k.as_ref())?;
        Ok(())
    }
}
