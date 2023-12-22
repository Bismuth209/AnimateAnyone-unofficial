
if true; then
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_micro.yaml
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_micro_v2_longer.yaml
    torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_2_UBC_micro_v2_longer.yaml
fi

# Random helper functions

switch-sizhky() {
    mv .git .git-og
    mv .git-sizhky .git
}

switch-og() {
    mv .git .git-sizhky
    mv .git-og .git
}