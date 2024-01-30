#!/bin/bash


train() {
  poetry run python -m src.run.train --config-path="$PWD/conf/" "$@"
}


# Example training:
train --config-name="example_pcc_singletask" ++criterion.lmbda.cls=100


################################################################################
#
# DEFINE MODEL HYPERPARAMETERS USED IN THE PAPER
#
################################################################################


MICRO_HPARAMS=(
  ++hp.num_channels.g_a='[3,16]'
  ++hp.num_channels.task_backend='[16,512,256,40]'
  ++hp.groups.g_a='[1]'
)

LITE_HPARAMS=(
  ++hp.num_channels.g_a='[3,8,8,16,16,32]'
  ++hp.num_channels.task_backend='[32,512,256,40]'
  ++hp.groups.g_a='[1,1,1,2,4]'
)

FULL_HPARAMS=(
  ++hp.num_channels.g_a='[3,64,64,64,128,1024]'
  ++hp.num_channels.task_backend='[1024,512,256,40]'
  ++hp.groups.g_a='[1,1,1,1,1]'
)


################################################################################
#
# TRAIN ALL RA CURVES
#
#
# NOTE: Potentially, this can be sped up via:
#
# - Parallelization
# - Pre-training a single high-rate model, then finetuning it for various lambdas/num_points
#
################################################################################


train_ra_curves() {
  for num_points in "${NUM_POINTS[@]}"; do
    for lmbda in "${LMBDAS[@]}"; do
      train "$@" ++criterion.lmbda.cls="${lmbda}" ++hp.num_points="${num_points}"
    done
  done
}


NUM_POINTS=(1024 512 256 128 64 32 16 8)
LMBDAS=(16000 8000 4000 1000 320 160 80 40 28 20 14 10)

train_ra_curves --config-name="example_pcc_singletask" ++model.name="um-pcc-cls-only-pointnet" "${MICRO_HPARAMS[@]}"
train_ra_curves --config-name="example_pcc_singletask" ++model.name="um-pcc-cls-only-pointnet" "${LITE_HPARAMS[@]}"
train_ra_curves --config-name="example_pcc_singletask" ++model.name="um-pcc-cls-only-pointnet" "${FULL_HPARAMS[@]}"


################################################################################
#
# TRAIN MULTI-TASK (INPUT RECONSTRUCTION) VARIANTS
#
################################################################################


COMMON_MULTI_TASK_ARGS=(
  --config-name="example_pcc_multitask"
  ++model.name="um-pcc-multitask-cls-pointnet"
  ++hp.detach_y1_hat=True
  ++criterion.lmbda.rec=16000
)

NUM_POINTS=(1024)
LMBDAS=(1000 160 40 28 14)

train_ra_curves "${COMMON_MULTI_TASK_ARGS[@]}" "${MICRO_HPARAMS[@]}" ++hp.num_channels.g_s='[16,64,64,256,3072]'
train_ra_curves "${COMMON_MULTI_TASK_ARGS[@]}" "${LITE_HPARAMS[@]}"  ++hp.num_channels.g_s='[32,64,64,256,3072]'
train_ra_curves "${COMMON_MULTI_TASK_ARGS[@]}" "${FULL_HPARAMS[@]}"  ++hp.num_channels.g_s='[1024,256,256,512,3072]'


################################################################################
#
# TRAIN POINTNET VANILLA (NO COMPRESSION)
#
################################################################################


NUM_POINTS=(1024 512 256 128 64 32 16 8)

for num_points in "${NUM_POINTS[@]}"; do
  train --config-name="example_pc_task" ++model.name="um-pc-cls-pointnet" ++criterion.lmbda.cls=1.0 ++hp.num_points="${num_points}"
done


