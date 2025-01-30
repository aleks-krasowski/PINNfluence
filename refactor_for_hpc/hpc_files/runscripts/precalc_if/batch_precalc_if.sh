#!/bin/bash

#SBATCH --job-name=burgers_IF
#SBATCH --mail-type=ALL

#         Output (stdout and stderr) of this job will go into a file named with SLURM_JOB_ID (%j) and the job_name (%x)
#SBATCH --output=./runs/%j_%x.out

#SBATCH --partition=amd1,amd2

#         Tell slurm to run the job on a single node. Always use this statement.
#SBATCH --nodes=1

#         Ask slurm to run at most 1 task (slurm task == OS process). A task (process) might launch subprocesses/threads.
#SBATCH --ntasks=1

#         Max number of cpus per process (threads/subprocesses) is 16. Seems reasonable with 4 GPUs on a 64 core machine.
#SBATCH --cpus-per-task=8

#         Request RAM for your job
#SBATCH --mem=12G

#SBATCH --array=10

#####################################################################################

# This included file contains the definition for $LOCAL_JOB_DIR to be used locally on the node.
source "/etc/slurm/local_job_dir.sh"

ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" batch_precalc_if.txt)
echo ${ARGS} > ${LOCAL_JOB_DIR}/args.txt

# Launch the apptainer image with --nv for nvidia support. Two bind mounts are used: 
# - One for the ImageNet dataset and 
# - One for the results (e.g. checkpoint data that you may store in $LOCAL_JOB_DIR on the node
apptainer run --bind ${LOCAL_JOB_DIR}:/opt/model_zoo --bind ${HOME}/pinnfluence_finetuning/pinnfluence:/opt/code --bind ${HOME}/pinnfluence_finetuning/model_zoo_src/:/opt/model_zoo_src /home/fe/krasowski/pinnfluence_finetuning/deepxde_generic_code.sif  /opt/code/burgers/precompute_influences.py $ARGS


# This command copies all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cp -r ${LOCAL_JOB_DIR} ${SLURM_SUBMIT_DIR}/${SLURM_JOB_ID}
