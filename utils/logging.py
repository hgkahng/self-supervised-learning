# -*- coding: utf-8 -*-

import os
import math
import shutil
import logging

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.logging import RichHandler


def make_epoch_message(history: dict,
                           current: int,
                           total: int,
                           best: int,
                           exclude: list = []) -> str:
    """Create description string for logging progress."""
    fmt = f">0{int(math.log10(total))+1}d"
    msg = list();
    for k, v in history.items():
        if k in exclude:
            continue
        msg.append(f"{k}: {v:>2.4f}")
    msg = " | ".join(msg)
    msg = f"Epoch: [{current:{fmt}}/{total:{fmt}}] ({best:{fmt}}) - " + msg
    return msg


def get_rich_pbar(console: Console = None,
                  transient: bool = True,
                  auto_refresh: bool = False,
                  disable: bool = False,
                  **kwargs) -> Progress:
    """A colorful progress bar based on the `rich` python library."""
    if console is None:
        console = Console(color_system='256', width=240)  # 160

    columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

    return Progress(
        *columns,
        console=console,
        auto_refresh=auto_refresh,
        transient=transient,
        disable=disable
    )


def configure_logger(logfile: str = None, level = logging.INFO) -> logging.Logger:
    """..."""

    # Logger instance
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # File handler
    if logfile is not None:
        touch(logfile)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s"))
        logger.addHandler(fileHandler)
    
    # Rich handler
    width, _ = shutil.get_terminal_size()
    console = Console(color_system='256', width=width)
    richHandler = RichHandler(console=console)
    richHandler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(richHandler)

    # Set logging level
    logger.setLevel(level)
    logger.propagate = False

    return logger


def get_rich_logger(logfile: str = None, level=logging.INFO) -> logging.Logger:  # TODO: remove
    """A colorful logger based on the `rich` python library."""
    myLogger = logging.getLogger()
    if myLogger.hasHandlers():
        myLogger.handlers.clear()

    # File handler (created if logfile is provided)
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
    myLogger.propagate = False

    return myLogger


def get_logger(logfile: str = None, level=logging.INFO) -> logging.Logger:
    """A basic logger."""
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    myLogger = logging.getLogger()

    # File handler (created if logfile is provided)
    if logfile:
        touch(logfile)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logFormatter)
        myLogger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(logFormatter)
    myLogger.addHandler(streamHandler)
    myLogger.setLevel(level)

    return myLogger


def touch(path: str, mode: str = 'w'):
    """Creates file. Ignored when file already exists."""
    assert mode in ['a', 'w']
    directory, _ = os.path.split(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    open(path, mode).close()
