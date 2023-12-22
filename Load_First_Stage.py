from torch_snippets import *

ckpt_path = P('outputs/train_stage_2_UBC_micro_v2_longer-2023-12-21T10-46-28/checkpoints/checkpoint-epoch-500.ckpt')
ckpt_name = ckpt_path.parent.parent.stem
save_to = P(f'checkpoints/{ckpt_name}/')
stage = 2

makedir(save_to)

full_state_dict = torch.load(ckpt_path,map_location='cpu')

poseguider_state_dict = full_state_dict['poseguider_state_dict']
referencenet_state_dict = full_state_dict['referencenet_state_dict']
unet_state_dict = full_state_dict['unet_state_dict']

poseguider_ckpt_path = f'{save_to}/poseguider_stage_{stage}.ckpt'
referencenet_ckpt_path = f'{save_to}/referencenet_stage_{stage}.ckpt'
unet_ckpt_path = f'{save_to}/unet_stage_{stage}.ckpt'

torch.save(poseguider_state_dict, poseguider_ckpt_path)
torch.save(referencenet_state_dict, referencenet_ckpt_path)
torch.save(unet_state_dict, unet_ckpt_path)
