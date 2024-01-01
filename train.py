from torch_snippets import *
from pipelines.animation_stage_1 import main as animation_stage_1
import os
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

# from diffusers.models import UNet2DConditionModel
from models.hack_unet2d import Hack_UNet2DConditionModel as UNet2DConditionModel

# from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

# from data.dataset import WebVid10M
from data.dataset import TikTok, collate_fn, UBC_Fashion

# from models.unet import UNet3DConditionModel
from models.hack_unet3d import Hack_UNet3DConditionModel as UNet3DConditionModel

# from animatediff.pipelines.pipeline_animation import AnimationPipeline
from utils.util import save_videos_grid, zero_rank_print
from models.ReferenceEncoder import ReferenceEncoder

from models.PoseGuider import PoseGuider
from models.ReferenceNet import ReferenceNet
from models.ReferenceNet_attention import ReferenceNetAttention

import glob
import os


def save_first_stage_weights(ckpt_path, stage=1):
    checkpoint_pattern = os.path.join(ckpt_path, "checkpoint*")
    checkpoint_files = glob.glob(checkpoint_pattern)
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"The latest checkpoint is: {latest_checkpoint}")
    else:
        print("No checkpoint files found in the specified folder.")
    ckpt_path = latest_checkpoint
    fldr_name = P(ckpt_path).parent.parent
    to = str(fldr_name).split("/")[-1]
    to = f"checkpoints/{to}/"
    makedir(to)
    Info(f"Saving weights to {to}")

    full_state_dict = torch.load(ckpt_path, map_location="cpu")

    poseguider_state_dict = full_state_dict["poseguider_state_dict"]
    referencenet_state_dict = full_state_dict["referencenet_state_dict"]
    unet_state_dict = full_state_dict["unet_state_dict"]

    poseguider_ckpt_path = f"{to}/poseguider_stage_{stage}.ckpt"
    referencenet_ckpt_path = f"{to}/referencenet_stage_{stage}.ckpt"
    unet_ckpt_path = f"{to}/unet_stage_{stage}.ckpt"

    torch.save(poseguider_state_dict, poseguider_ckpt_path)
    torch.save(referencenet_state_dict, referencenet_ckpt_path)
    torch.save(unet_state_dict, unet_ckpt_path)


def init_dist(launcher="slurm", backend="nccl", port=28888, **kwargs):
    """Initializes distributed environment."""
    if launcher == "pytorch":
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(
            f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}"
        )

    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")

    return local_rank


def get_parameters_without_gradients(model):
    """
    Returns a list of names of the model parameters that have no gradients.

    Args:
    model (torch.nn.Module): The model to check.

    Returns:
    List[str]: A list of parameter names without gradients.
    """
    no_grad_params = []
    for name, param in model.named_parameters():
        print(f"{name} : {param.grad}")
        if param.grad is None:
            no_grad_params.append(name)
    return no_grad_params


def main(
    image_finetune: bool,
    name: str,
    use_wandb: bool,
    launcher: str,
    output_dir: str,
    pretrained_model_path: str,
    clip_model_path: str,
    description: str,
    fusion_blocks: str,
    poseguider_checkpoint_path: str,
    referencenet_checkpoint_path: str,
    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs=None,
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",
    trainable_modules: Tuple[str] = (None,),
    num_workers: int = 8,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,
    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    global_seed: int = 42,
    is_debug: bool = False,
    folder_name=None,
    pretrained_unet_path=None,
):
    check_min_version("0.21.4")

    # Initialize distributed training
    local_rank = init_dist(launcher=launcher, port=28888)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    # num_processes   = 0
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    # folder_name = (
    #     "debug"
    #     if is_debug
    #     else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    # )
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(
            project="AnimateAnyone train stage 1", name=folder_name, config=config
        )

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))

        print(description)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    # tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    clip_image_encoder = ReferenceEncoder(model_path=clip_model_path)
    poseguider = PoseGuider(noise_latent_channels=320)
    referencenet = ReferenceNet.from_pretrained(pretrained_model_path, subfolder="unet")
    try:
        if poseguider_checkpoint_path != "" and P(poseguider_checkpoint_path).exists():
            poseguider_state_dict = torch.load(
                poseguider_checkpoint_path, map_location="cpu"
            )
            poseguider.load_state_dict(poseguider_state_dict, strict=True)
            Info("Loaded poseguider weights properly!")

        if (
            referencenet_checkpoint_path != ""
            and P(referencenet_checkpoint_path).exists()
        ):
            referencenet_state_dict = torch.load(
                referencenet_checkpoint_path, map_location="cpu"
            )
            referencenet.load_state_dict(referencenet_state_dict, strict=True)
            Info("Loaded referencenet weights properly!")

    except Exception as e:
        Warn(f"Error while loading poseguider and referencnet weights...\n{e}")

    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
        )
        # unet = UNet3DConditionModel.from_pretrained_2d(
        #     'checkpoints/train_stage_2_UBC_768-2023-12-26T04-55-42',
        #     unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
        #     specific_model='unet_stage_2.ckpt'
        # )
    else:
        if pretrained_unet_path != "" and P(pretrained_unet_path).exists():
            unet_config = UNet2DConditionModel.load_config(
                pretrained_model_path, subfolder="unet"
            )
            unet = UNet2DConditionModel.from_config(unet_config)
            unet_state_dict = torch.load(pretrained_unet_path, map_location="cpu")
            unet.load_state_dict(unet_state_dict, strict=False)
            Info("Loaded weights for UNET!")
        else:
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_path, subfolder="unet"
            )

    reference_control_writer = ReferenceNetAttention(
        referencenet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks=fusion_blocks,
        batch_size=train_batch_size,
        is_image=image_finetune,
    )
    reference_control_reader = ReferenceNetAttention(
        unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks=fusion_blocks,
        batch_size=train_batch_size,
        is_image=image_finetune,
    )

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path:
            zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = (
            unet_checkpoint_path["state_dict"]
            if "state_dict" in unet_checkpoint_path
            else unet_checkpoint_path
        )

        m, u = unet.load_state_dict(state_dict, strict=True)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        del state_dict
        assert len(u) == 0

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)

    # Set unet trainable parameters
    unet.requires_grad_(False)
    # unet.requires_grad_(True)
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                # print(trainable_module_name)
                param.requires_grad = True
                break

    if image_finetune:
        poseguider.requires_grad_(True)
        referencenet.requires_grad_(True)
    else:
        poseguider.requires_grad_(False)
        referencenet.requires_grad_(False)

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if image_finetune:
        trainable_params += list(
            filter(lambda p: p.requires_grad, poseguider.parameters())
        ) + list(filter(lambda p: p.requires_grad, referencenet.parameters()))

    # print(len(trainable_params))
    # exit(0)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(
            f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M"
        )

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            referencenet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        referencenet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    # text_encoder.to(local_rank)
    clip_image_encoder.to(local_rank)
    poseguider.to(local_rank)
    referencenet.to(local_rank)

    # Get the training dataset
    # train_dataset = WebVid10M(**train_data, is_image=image_finetune)
    # train_dataset = TikTok(**train_data, is_image=image_finetune)
    train_dataset = UBC_Fashion(**train_data, is_image=image_finetune)

    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * num_processes
        )

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    if image_finetune:
        poseguider = DDP(poseguider, device_ids=[local_rank], output_device=local_rank)
        referencenet = DDP(
            referencenet, device_ids=[local_rank], output_device=local_rank
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps), disable=not is_main_process
    )
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        poseguider.train()
        referencenet.train()

        for step, batch in enumerate(train_dataloader):
            ### >>>> Training >>>> ###

            # Convert videos to latent space
            pixel_values = batch["pixel_values"].to(local_rank)
            pixel_values_pose = batch["pixel_values_pose"].to(local_rank)
            clip_ref_image = batch["clip_ref_image"].to(local_rank)
            pixel_values_ref_img = batch["pixel_values_ref_img"].to(local_rank)
            drop_image_embeds = batch["drop_image_embeds"].to(
                local_rank
            )  # torch.Size([bs])
            video_length = pixel_values.shape[1]

            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                latents = latents * 0.18215

                latents_ref_img = vae.encode(pixel_values_ref_img).latent_dist
                latents_ref_img = latents_ref_img.sample()
                latents_ref_img = latents_ref_img * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if not image_finetune:
                pixel_values_pose = rearrange(
                    pixel_values_pose, "b f c h w -> (b f) c h w"
                )
                latents_pose = poseguider(pixel_values_pose)
                latents_pose = rearrange(
                    latents_pose, "(b f) c h w -> b c f h w", f=video_length
                )
            else:
                latents_pose = poseguider(pixel_values_pose)

            # noisy_latents = noisy_latents + latents_pose

            # Get the text embedding for conditioning
            with torch.no_grad():
                # prompt_ids = tokenizer(
                #     batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                # ).input_ids.to(latents.device)
                # encoder_hidden_states = text_encoder(prompt_ids)[0]
                encoder_hidden_states = clip_image_encoder(clip_ref_image).unsqueeze(
                    1
                )  # [bs,1,768]

            # support cfg train
            mask = drop_image_embeds > 0
            mask = mask.unsqueeze(1).unsqueeze(2).expand_as(encoder_hidden_states)
            encoder_hidden_states[mask] = 0

            # pdb.set_trace()

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                ref_timesteps = torch.zeros_like(timesteps)

                # pdb.set_trace()

                referencenet(latents_ref_img, ref_timesteps, encoder_hidden_states)
                reference_control_reader.update(reference_control_writer)

                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    latent_pose=latents_pose,
                ).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # no_grad_params_poseguider = get_parameters_without_gradients(poseguider)
                # no_grad_params_referencenet = get_parameters_without_gradients(referencenet)
                # if len(no_grad_params_poseguider) != 0:
                #     print("PoseGuider no grad params:", no_grad_params_poseguider)
                # if len(no_grad_params_referencenet) != 0:
                #     print("ReferenceNet no grad params:", no_grad_params_referencenet)

                """ >>> gradient clipping >>> """
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)

            reference_control_reader.clear()
            reference_control_writer.clear()
            global_step += 1

            ### <<<< Training <<<< ###

            # Save checkpoint
            if is_main_process and (
                global_step == 1
                or global_step % checkpointing_steps == 0
                or (
                    epoch % 5 == 0 and step == len(train_dataloader) - 1
                )  # save every 5th epoch
            ):
                save_path = os.path.join(output_dir, f"checkpoints")
                if image_finetune:
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "unet_state_dict": unet.module.state_dict(),
                        "poseguider_state_dict": poseguider.module.state_dict(),
                        "referencenet_state_dict": referencenet.module.state_dict(),
                    }
                else:
                    state_dict = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "unet_state_dict": unet.module.state_dict(),
                    }

                if step == len(train_dataloader) - 1:
                    torch.save(
                        state_dict,
                        os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"),
                    )
                    try:
                        os.remove(
                            os.path.join(save_path, f"checkpoint-epoch-{epoch}.ckpt")
                        )
                    except Exception as e:
                        Warn(f"Warning: {e}")
                else:
                    torch.save(
                        state_dict,
                        os.path.join(
                            save_path, f"checkpoint-global_step-{global_step}.ckpt"
                        ),
                    )
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                save_first_stage_weights(save_path)
                args = AD(
                    dist=False,
                    world_size=1,
                    rank=0,
                    config="configs/prompts/v3/v3.1.yaml",
                )
                animation_results = animation_stage_1(args)
                images = [read(im, 1)[None] for im in animation_results.images]
                images = (
                    torch.Tensor(np.concatenate(images).astype(np.uint8))
                    .permute(0, 3, 1, 2)
                    .long()
                )
                from torchvision.utils import make_grid

                all_images = make_grid(images, nrow=1)
                wandb.log(
                    {f"images": wandb.Image(all_images / 255.0)}, step=global_step
                )
                import shutil

                shutil.rmtree(animation_results.savedir)

            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch"
    )
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)

    # CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_oneshot.yaml
    # CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes=1 --nproc_per_node=2 --master_port 28888 train.py --config configs/training/train_stage_1.yaml
    # CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=6 --master_port 28889 train.py --config configs/training/train_stage_1.yaml
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port 28887 train.py --config configs/training/train_stage_1.yaml

    # CUDA_VISIBLE_DEVICES=7 torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_2.yaml
