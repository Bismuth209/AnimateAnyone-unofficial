from torch_snippets import *
import glob
from pipelines.animation_stage_1 import main as animation_stage_1


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


save_path = os.path.join("outputs/v3.1/", f"checkpoints")
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
    torch.Tensor(np.concatenate(images).astype(np.uint8)).permute(0, 3, 1, 2).long()
)
from torchvision.utils import make_grid

all_images = make_grid(images, nrow=1)
all_images2 = Image.fromarray((all_images.permute(1, 2, 0).numpy()).astype(np.uint8))
all_images3 = resize(all_images2, 0.5)
all_images3.save("/tmp/tmp.png")
