
if true; then
    ## training stage 1
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_micro.yaml
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC.yaml
    torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_768.yaml

    ## training stage 2
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_micro_v2_longer.yaml
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_2_UBC_micro_v2_longer.yaml

    ## inference stage 1
    # python3 -m pipelines.animation_stage_1 --config configs/prompts/my_animation_stage_1_202312231519.yaml
    # python3 -m pipelines.animation_stage_1 --config configs/prompts/my_animation_stage_1_202312232123.yaml

    ## inference stage 2
    # python3 -m pipelines.animation_stage_2 --config configs/prompts/my_animation_stage_2_202312211801.yaml
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