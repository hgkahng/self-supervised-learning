
import os
import sys
import wandb
import torch

from configs.task_configs import CLAPv2Config
from run_clap import main


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    config = CLAPv2Config.parse_arguments()

    try:
        main(config)
    except KeyboardInterrupt:
        wandb.finish()
        os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
        sys.exit(0)
