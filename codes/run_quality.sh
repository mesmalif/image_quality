#!/bin/bash

#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=20   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=200G        # memory per node
#SBATCH --time=01-20:00      # time (DD-HH:MM)
#SBATCH --output=cnn_terminal.out  # %N for node name, %j for jobID

echo "started"
SECONDS=0
#module load cuda cudnn 
source "../venv_quality/bin/activate"
python local_quality.py 
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


