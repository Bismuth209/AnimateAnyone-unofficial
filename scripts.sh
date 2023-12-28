
if true; then
    ## training stage 1
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_micro.yaml
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC.yaml
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_768.yaml
    # torchrun --rdzv_endpoint=localhost:29299 --master-port=29292 --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_768_micro.yaml

    ## training stage 2
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_micro_v2_longer.yaml
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_2_UBC_micro_v2_longer.yaml
    # torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_2_UBC_768.yaml

    ## inference stage 1
    python3 -m pipelines.animation_stage_1 --config configs/prompts/my_animation_stage_1_202312241057.yaml

    ## inference stage 2
    # python3 -m pipelines.animation_stage_2 --config configs/prompts/my_animation_stage_2_202312262032.yaml
fi

# Random helper functions

switch-sizhky() {
    mv .git ../gits/.git-og
    mv ../gits/.git-sizhky .git
}

switch-og() {
    mv .git ../gits/.git-sizhky
    mv ../gits/.git-og .git
}