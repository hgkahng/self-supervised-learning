# -*- coding: utf-8 -*-

import os
import sys
import shutil
import logging

from rich.console import Console
from rich.progress import Progress
from rich.logging import RichHandler


def make_epoch_description(history: dict, current: int, total: int, best: int, exclude: list = []):
    """Create description string for logging progress."""
    pfmt = f">{len(str(total))}d"
    desc = f" Epoch: [{current:{pfmt}}/{total:{pfmt}}] ({best:{pfmt}}) |"
    for metric_name, metric_dict  in history.items():
        if not isinstance(metric_dict, dict):
            raise TypeError("`history` must be a nested dictionary.")
        if metric_name in exclude:
            continue
        for k, v in metric_dict.items():
            desc += f" {k}_{metric_name}: {v:.3f} |"
    return desc


def get_rich_pbar(transient: bool = True, auto_refresh: bool = False):
    """A colorful progress bar based on the `rich` python library."""
    console = Console(color_system='256', force_terminal=True, width=160)
    return Progress(
        console=console,
        auto_refresh=auto_refresh,
        transient=transient
    )


def get_rich_logger(logfile: str = None, level=logging.INFO):
    """A colorful logger based on the `rich` python library."""

    myLogger = logging.getLogger()

    # File handler
    if logfile is not None:
        touch(logfile)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s"))
        myLogger.addHandler(fileHandler)

    # Rich handler
    width, _ = shutil.get_terminal_size()
    console = Console(color_system='256', width=width)
    richHandler = RichHandler(console=console)
    richHandler.setFormatter(logging.Formatter("%(message)s"))
    myLogger.addHandler(richHandler)

    # Set level
    myLogger.setLevel(level)

    return myLogger


def get_logger(stream=False, logfile=None, level=logging.INFO):
    """
    Arguments:
        stream: bool, default False.
        logfile: str, path.
    """
    _format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logFormatter = logging.Formatter(_format)

    rootLogger = logging.getLogger()

    if logfile:
        touch(logfile)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    if stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(logFormatter)
        rootLogger.addHandler(streamHandler)

    rootLogger.setLevel(level)

    return rootLogger


def touch(filepath: str, mode: str='w'):
    assert mode in ['a', 'w']
    directory, _ = os.path.split(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    open(filepath, mode).close()
