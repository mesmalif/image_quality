#!/bin/bash

#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=92000M        # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=cnn_terminal.out  # %N for node name, %j for jobID

echo "started"
SECONDS=0
#module load cuda cudnn 
source "./venv_clf/bin/activate"
python create_cat_dog_dataset.py 
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


