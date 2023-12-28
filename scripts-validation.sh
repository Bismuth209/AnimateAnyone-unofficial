#! /bin/bash
if true; then
    ## inference stage 1
    # /home/ubuntu/miniconda3/envs/manimate/bin/python Load_First_Stage.py outputs/v6
    /home/ubuntu/miniconda3/envs/manimate/bin/python -m pipelines.animation_stage_1 --config configs/prompts/my_animation_stage_1_202312241057.yaml

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