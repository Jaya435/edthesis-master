#!/bin/sh
###########################################
#                                         #
# This job predicts and plots where the   #
# buildings are on an RGB image.          #
#                                         #
###########################################

# Grid Engine Options
#$ -N pred_rst
#$ -cwd
#$ -l h_rt=06:00:00
#$ -pe sharedmem 16
#$ -l h_vmem=8G

# Initialise the modules framework
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v
wdir = 
# User specified commands go below here
module load anaconda/5.0.1
source activate gdal
# Run the program
dir='/exports/eddie/scratch/s1217815/AerialImageDataset/train/images'
find $dir -type f > filename.txt
input='filename.txt'
readarray myArray < ${input}
python ${HOME}/python/raster_predict.py -model /exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/saved_models/model_inria_batch2_lr0.01_arch16_epochs100.pt -inpfile ${myArray[$SGE_TASK_ID-1]}
