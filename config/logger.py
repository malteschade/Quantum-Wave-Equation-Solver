#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Description}
{License_info}
"""

# -------- IMPORTS --------
# Built-in modules
from typing import Any
import logging

# Other modules
import numpy as np

# -------- CLASSES --------
class Logger:
    """
    A utility class for setting up a custom logger.

    This class is responsible for creating and configuring a logger that outputs to both 
    the console and a specified log file.

    Attributes:
        _logger (logging.Logger, optional): A static instance of the logging.Logger class.
        _file_handler (logging.FileHandler, optional): File handler for logging to a file.
        _formatter (logging.Formatter): Formatter for log messages.
    """
    _logger = None
    _file_handler = None
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    @staticmethod
    def setup_logger(log_file: str, verbose: int = 6) -> logging.Logger:
        """
        Sets up a logger with both stream and file handlers.

        Creates a new logger or configures the existing one to log messages both to the console 
        and to a specified file. The verbosity of the console logger can be adjusted.

        Args:
            log_file (str): The path to the log file.
            verbose (int): The verbosity level for the stream (console) logger,
                           from 1 (least verbose) to 6 (most verbose). Default is 6.

        Returns:
            logging.Logger: Configured logger instance.
        """
        if Logger._logger is None:
            # Create a new logger
            Logger._logger = logging.getLogger('logger')
            Logger._logger.setLevel(logging.DEBUG)

            # Stream handler
            if verbose < 6:
                stream_handler = logging.StreamHandler()
                stream_handler.setLevel(round(verbose) * 10)
                stream_handler.setFormatter(Logger._formatter)
                Logger._logger.addHandler(stream_handler)

        # Remove the previous file handler if it exists
        if Logger._file_handler is not None:
            Logger._logger.removeHandler(Logger._file_handler)

        # Add or replace the file handler
        Logger._file_handler = logging.FileHandler(log_file)
        Logger._file_handler.setLevel(logging.DEBUG)
        Logger._file_handler.setFormatter(Logger._formatter)
        Logger._logger.addHandler(Logger._file_handler)

        return Logger._logger

# -------- FUNCTIONS --------
def handle_ndarray(obj: Any) -> Any:
    """
    Converts a numpy ndarray to a list if the object is an ndarray,
    otherwise returns the object as-is.

    This function is useful for handling numpy ndarrays in contexts where JSON serialization or
    similar operations are performed, as ndarrays are not directly serializable.

    Args:
        obj (Any): The object to be converted or passed through.

    Returns:
        Any: A list if the input was an ndarray, or the original object otherwise.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
