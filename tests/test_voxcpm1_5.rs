use std::time::Instant;

use aha::{
    models::voxcpm::{generate::VoxCPMGenerate, tokenizer::SingleChineseTokenizer},
    utils::audio_utils::save_wav,
};
use anyhow::{Ok, Result};

#[test]
fn voxcpm1_5_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda voxcpm1_5_generate -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/OpenBMB/VoxCPM1.5/";

    let i_start = Instant::now();
    let mut voxcpm_generate = VoxCPMGenerate::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    // let generate = voxcpm_generate.generate_simple("太阳当空照，花儿对我笑，小鸟说早早早".to_string())?;
    let generate = voxcpm_generate.generate(
        "太阳当空照，花儿对我笑，小鸟说早早早".to_string(),
        // Some("啥子小师叔，打狗还要看主人，你再要继续，我，就是你的对手".to_string()),
        // Some("./assets/audio/voice_01.wav".to_string()),
        Some("一定被灰太狼给吃了，我已经为他准备好了花圈了".to_string()),
        Some("./assets/audio/voice_05.wav".to_string()),
        2,
        100,
        10,
        2.0,
        false,
        6.0,
    )?;

    // 创建prompt_cache
    // let _ = voxcpm_generate.build_prompt_cache(
    //     "啥子小师叔，打狗还要看主人，你再要继续，我，就是你的对手".to_string(),
    //     "./assets/audio/voice_01.wav".to_string(),
    // )?;
    // // 使用prompt_cache生成语音
    // let generate = voxcpm_generate.generate_use_prompt_cache(
    //     "太阳当空照，花儿对我笑，小鸟说早早早".to_string(),
    //     2,
    //     100,
    //     10,
    //     2.0,
    //     false,
    //     6.0,
    // )?;

    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    save_wav(&generate, "voxcpm.wav")?;
    Ok(())
}

#[test]
fn voxcpm1_5_tokenizer() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda voxcpm1_5_tokenizer -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/OpenBMB/VoxCPM1.5/";
    let tokenizer = SingleChineseTokenizer::new(model_path)?;
    let ids = tokenizer.encode("你好啊，你吃饭了吗".to_string())?;
    println!("ids: {:?}", ids);
    Ok(())
}
