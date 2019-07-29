#!/bin/sh
###########################################
#                                         #
#     This job loops over saved models    #
#     and outputs the most accurate       #
#                                         #
###########################################

# Grid Engine Options
#$ -N Accuracy
#$ -cwd
#$ -l h_rt=02:00:00
#$ -pe sharedmem 16
#$ -l h_vmem=8G

# Initialise the modules framework
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v
WORKING_DIR="/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/Results/2542531"
# User specified commands go below here
module load anaconda/5.0.1
source activate mypytorch
# Run the program
python ${HOME}/python/accuracy.py --out_dir ${WORKING_DIR} --model_path ${WORKING_DIR}
