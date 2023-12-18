from torch_snippets import *
from torch_snippets.markup2 import AD
from omegaconf import OmegaConf
from models.PoseGuider import PoseGuider
from models.ReferenceNet import ReferenceNet
from models.unet import UNet3DConditionModel
from diffusers.models import UNet2DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor
from models.ReferenceEncoder import ReferenceEncoder
from models.ReferenceNet_attention import ReferenceNetAttention


def load_unet(config, state_dict=None):
    if isinstance(config, (P,str)):
        config = read_json(config)
    strict = True
    classes = {'UNet3DConditionModel': UNet3DConditionModel, 'UNet2DConditionModel': UNet2DConditionModel}
    unet = classes[config['_class_name']].from_config(config)
    if config['_class_name'] == 'UNet2DConditionModel': strict=False
    unet.load_state_dict(state_dict, strict=strict)
    return unet


def load_models_stage_2(config):
    """
    config = AD(
        pretrained_model_path="checkpoints/stable-diffusion-v1-5/",
        checkpoint_folder="outputs/train_stage_2_v1-2023-12-17T18-31-50/",
        checkpoint_file="outputs/train_stage_2_v1-2023-12-17T18-31-50/checkpoints/checkpoint-epoch-88.ckpt",
        clip_model_path="checkpoints/clip-vit-base-patch32/"
    )
    models = load_models_stage_2(config)
    models.print_summary()
    """
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
    state_dict = torch.load(config.checkpoint_file)
    unet = load_unet(f'{config.checkpoint_folder}/unet.config.json', state_dict['unet'])
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")

    poseguider = PoseGuider.from_pretrained(pretrained_model_path=config.checkpoint_file)
    clip_image_encoder = ReferenceEncoder(model_path=config.clip_model_path)
    clip_image_processor = CLIPProcessor.from_pretrained(config.clip_model_path,local_files_only=True)
    # referencenet = ReferenceNet.load_referencenet(pretrained_model_path=config.pretrained_referencenet_path)
    referencenet = load_unet(f'{config.checkpoint_folder}/referencenet.config.json', state_dict['referencenet_state_dict'])
    
    reference_control_writer = None
    reference_control_reader = None

    unet.enable_xformers_memory_efficient_attention()
    referencenet.enable_xformers_memory_efficient_attention()

    vae.to(torch.float32)
    unet.to(torch.float32)
    text_encoder.to(torch.float32)
    referencenet.to(torch.float32).to(device)
    poseguider.to(torch.float32).to(device)
    clip_image_encoder.to(torch.float32).to(device)
    return AD(
        config, device, 
        reference_control_writer, reference_control_reader, 
        referencenet, poseguider, clip_image_processor, 
        clip_image_encoder, tokenizer, vae, unet, text_encoder
    )
