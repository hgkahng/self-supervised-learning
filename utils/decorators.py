
import time
import functools
import logging


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        if value is not None:
            return value, elapsed_time
        else:
            return elapsed_time
    return wrapper_timer


def suppress_logging_info(func):
    def wrapped(*args, **kwargs):
        logging.getLogger().setLevel(logging.WARNING)
        output = func(*args, **kwargs)
        logging.getLogger().setLevel(logging.INFO)
        if output is not None:
            return output

    return wrapped
