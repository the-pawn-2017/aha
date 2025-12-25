use std::io::Cursor;

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::Result;
use base64::{Engine, prelude::BASE64_STANDARD};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::{Rgba, RgbaImage};
use rocket::futures::{Stream, stream};

use crate::{
    models::{GenerateModel, rmbg2_0::model::BiRefNet},
    utils::{
        build_img_completion_response, find_type_files, get_device, get_dtype,
        img_utils::{extract_images, float_tensor_to_dynamic_image, img_transform_with_resize},
    },
};

pub struct RMBG2_0Model {
    model: BiRefNet,
    h: u32,
    w: u32,
    img_mean: Tensor,
    img_std: Tensor,
    device: Device,
    dtype: DType,
    model_name: String,
}

impl RMBG2_0Model {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let device = get_device(device);
        let dtype = get_dtype(dtype, "float32");
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = BiRefNet::new(vb)?;
        let img_mean =
            Tensor::from_slice(&[0.485, 0.456, 0.406], (3, 1, 1), &device)?.to_dtype(dtype)?;
        let img_std =
            Tensor::from_slice(&[0.229, 0.224, 0.225], (3, 1, 1), &device)?.to_dtype(dtype)?;
        Ok(Self {
            model,
            h: 1024,
            w: 1024,
            img_mean,
            img_std,
            device,
            dtype,
            model_name: "RMBG2.0".to_string(),
        })
    }

    pub fn inference(&self, mes: ChatCompletionParameters) -> Result<Vec<RgbaImage>> {
        let imgs = extract_images(&mes)?;
        let mut rmbg_png = vec![];
        for img in imgs {
            let height = img.height();
            let width = img.width();
            let img_tensor = img_transform_with_resize(
                &img,
                self.h,
                self.w,
                &self.img_mean,
                &self.img_std,
                &self.device,
                self.dtype,
            )?
            .unsqueeze(0)?;
            let rmbg_img = self.model.forward(&img_tensor)?.squeeze(0)?;
            let alpha_img = float_tensor_to_dynamic_image(&rmbg_img)?;
            let alpha_img =
                alpha_img.resize_exact(width, height, image::imageops::FilterType::CatmullRom);
            let alpha_gray = alpha_img.to_luma8();
            let mut rgba_img = RgbaImage::new(width, height);

            // 遍历像素并组合
            for (x, y, pixel) in img.to_rgb8().enumerate_pixels() {
                let alpha_value = alpha_gray.get_pixel(x, y).0[0];
                let rgba_pixel = Rgba([pixel.0[0], pixel.0[1], pixel.0[2], alpha_value]);
                rgba_img.put_pixel(x, y, rgba_pixel);
            }
            rmbg_png.push(rgba_img);
        }
        Ok(rmbg_png)
    }
}

impl GenerateModel for RMBG2_0Model {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let rmbg_png = self.inference(mes)?;
        let mut base64_vec = vec![];
        for img in rmbg_png {
            let mut png_bytes = Vec::new();
            img.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)?;
            let base64_string = BASE64_STANDARD.encode(png_bytes);
            base64_vec.push(base64_string);
        }
        let response = build_img_completion_response(&base64_vec, &self.model_name);
        Ok(response)
    }
    #[allow(unused_variables)]
    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let error_stream = stream::once(async {
            Err(anyhow::anyhow!(format!(
                "{} model not support stream",
                self.model_name
            ))) as Result<ChatCompletionChunkResponse, anyhow::Error>
        });

        Ok(Box::new(Box::pin(error_stream)))
    }
}
