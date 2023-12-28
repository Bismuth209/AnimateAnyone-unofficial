from torch_snippets import *
import glob
import os


def save_first_stage_weights(ckpt_path, stage):
    checkpoint_pattern = os.path.join(ckpt_path, "checkpoint*")
    checkpoint_files = glob.glob(checkpoint_pattern)
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"The latest checkpoint is: {latest_checkpoint}")
    else:
        print("No checkpoint files found in the specified folder.")
    ckpt_path = latest_checkpoint
    to = f'checkpoints/{stem(parent(parent(ckpt_path)))}/'
    makedir(to)
    print(to)

    full_state_dict = torch.load(ckpt_path, map_location='cpu')

    poseguider_state_dict = full_state_dict['poseguider_state_dict']
    referencenet_state_dict = full_state_dict['referencenet_state_dict']
    unet_state_dict = full_state_dict['unet_state_dict']

    poseguider_ckpt_path = f'{to}/poseguider_stage_{stage}.ckpt'
    referencenet_ckpt_path = f'{to}/referencenet_stage_{stage}.ckpt'
    unet_ckpt_path = f'{to}/unet_stage_{stage}.ckpt'

    torch.save(poseguider_state_dict, poseguider_ckpt_path)
    torch.save(referencenet_state_dict, referencenet_ckpt_path)
    torch.save(unet_state_dict, unet_ckpt_path)

if __name__ == '__main__':
    # ckpt_path = './outputs/train_stage_1_UBC_768-2023-12-25T07-30-58/checkpoints/'
    ckpt_path = sys.argv[1]
    stage = 1
    save_first_stage_weights(ckpt_path, stage)