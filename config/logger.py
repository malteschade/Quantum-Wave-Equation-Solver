import logging

import numpy as np

def setup_logger(log_file, verbose=6):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose < 6:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(round(verbose)*10)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

def handle_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj