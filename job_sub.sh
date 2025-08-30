#!/bin/bash
#
#SBATCH --job-name=MILC_CNN
#SBATCH --account=psy53c17
#SBATCH -o /data/users4/ziqbal5/ILR/training_output/%j.out # STDOUT
#SBATCH --partition=qTRDGPUH
#SBATCH  --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --array=0-99

eval "$(conda shell.bash hook)"
conda activate z4_env
config=/data/users4/ziqbal5/ILR/config.txt


Data=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
Encoder=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
Seed_value=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
ws=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
nw=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
wsize=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)
convsize=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)
ep=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)
tp=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $10}' $config)
samples=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $11}' $config)
l_ptr=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $12}' $config)
fold_v=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $13}' $config)

b_file='output.txt'
echo " ${SLURM_JOBID}" >> $b_file
#sort -n -o "$b_file" "$b_file"


#python -m Main --jobid $SLURM_JOBID --fold_v $fold_v --daata $Data --encoder $Encoder --seeds $Seed_value --ws $ws --nw $nw --wsize $wsize --convsize $convsize --epp $ep --tp $tp --samples $samples --l_ptr $l_ptr
python -m Main2 --jobid $SLURM_JOBID --fold_v $fold_v --daata $Data --encoder $Encoder --seeds $Seed_value --ws $ws --nw $nw --wsize $wsize --convsize $convsize --epp $ep --tp $tp --samples $samples --l_ptr $l_ptr
#python -m Main --fold_v 0 --daata FBIRN --encoder LSTM --seeds 1 --ws 140 --nw 1 --wsize 140 --convsize 2400 --epp 2 --tp 140 --samples 311 --l_ptr F
#python -m Main --fold_v 0 --daata FBIRN --encoder LSTM --seeds 1 --ws 20 --nw 7 --wsize 20 --convsize 0 --epp 2 --tp 140 --samples 311 --l_ptr F