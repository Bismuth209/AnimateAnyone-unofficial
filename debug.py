from train2 import *
from pipelines.pipeline_aa2 import AnimateAnyonePipeline
from utils.show_video import denormalize_image

# config = 'configs/prompts/my_stage_1_pipeline.yaml'
config = "configs/prompts/my_stage_2_pipeline.yaml"
config = OmegaConf.load(config)

(
    noise_scheduler,
    unet,
    clip_image_encoder,
    text_encoder,
    tokenizer,
    vae,
    reference_control_reader,
    reference_control_writer,
    poseguider,
    referencenet,
) = load_all_models(**config)


pipeline = AnimateAnyonePipeline(
    vae,
    clip_image_encoder.model,
    unet,
    referencenet,
    poseguider,
    noise_scheduler,
    clip_image_encoder.processor,
)
train_dataset = TikTok(**config.train_data, is_image=config.image_finetune)

data = train_dataset[0]
pixel_values_pose = [
    Image.fromarray(denormalize_image(i.permute(1, 2, 0).numpy()))
    for i in data["pixel_values_pose"]
]
pixel_values_ref_img = Image.fromarray(
    denormalize_image(data["pixel_values_ref_img"].permute(1, 2, 0).numpy())
)

if __name__ == "__main__":
    o = pipeline(pixel_values_ref_img, pixel_values_pose, max_guidance_scale=0)
