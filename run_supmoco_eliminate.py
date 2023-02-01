
import os
import sys
import warnings
import wandb
import torch

from configs.task_configs import SupMoCoEliminateConfig
from run_moco import main


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    config = SupMoCoEliminateConfig.parse_arguments()

    try:
        main(config)
    except KeyboardInterrupt:
        wandb.finish()
        os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
        sys.exit(0)
