#!/bin/sh

# Grid Engine options
#
#$ -N train-titan
#$ -cwd
# -pe gpu 2 
# -l h_vmem=32G
#$ -pe gpu-titanx 2
#$ -l h_vmem=25G
#$ -l h_rt=03:00:00



WORKING_DIR="/exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/Results/$JOB_ID"

# Initialise the modules framework
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v

# User specified commands go below here
module load anaconda/5.0.1
source activate mypytorch2
mkdir ${WORKING_DIR}
# Read a text file, containing a list of possible combinations#
input='grid_search_96.txt'
readarray myArray < "$input"
set -- ${myArray[$SGE_TASK_ID]}
# submits batch job to SGE engine
echo arch _size is "$1"
echo learning rate is "$2"
echo batch size is "$3"
python ${HOME}/python/ConvNet.py --out_dir ${WORKING_DIR} --arch_size "$1" --lr "$2" --batch_size "$3"
#python ${HOME}/python/ConvNet.py --out_dir /exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/Results/2542531 --arch_size 2 --lr 0.1 --batch_size 16
#python ${HOME}/python/ConvNet.py --out_dir /exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/Results/2542531 --arch_size 2 --lr 0.1 --batch_size 32
#python ${HOME}/python/ConvNet.py --out_dir /exports/csce/eddie/geos/groups/geos_cnn_imgclass/data/Results/2542531 --arch_size 2 --lr 0.01 --batch_size 128
