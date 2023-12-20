
if false; then
    torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/train_stage_1_UBC_micro.yaml
fi


switch-sizhky() {
    mv .git .git-og
    mv .git-sizhky .git
}

switch-og() {
    mv .git .git-sizhky
    mv .git-og .git
}