#!/bin/sh
###########################################
#                                         #
# This job predicts and plots where the   #
# buildings are on an RGB image.          #
#                                         #
###########################################

# Grid Engine Options
#$ -N Predict
#$ -cwd
#$ -l h_rt=01:00:00
#$ -pe sharedmem 16
#$ -l h_vmem=8G

# Initialise the modules framework
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v

# User specified commands go below here
module load anaconda/5.0.1
source activate mypytorch
# Run the program

for FILE in /exports/eddie/scratch/s1217815/AerialImageDataset/train/images/*.tif;
do
    fname=$(basename $FILE)
    python ${HOME}/python/predict_compare.py -model /exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/saved_models/model_inria_batch2_lr0.01_arch16_epochs100.pt -inpfile /exports/eddie/scratch/s1217815/AerialImageDataset/train/images/"$fname" -mask /exports/eddie/scratch/s1217815/AerialImageDataset/train/gt/"$fname"
done
#./test_model.py -model model_inria.pt
