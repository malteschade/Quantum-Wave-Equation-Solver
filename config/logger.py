#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Custom Data Logger Class}
{    
    Copyright (C) [2023]  [Malte Schade]

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    }
"""

# -------- IMPORTS --------
# Built-in modules
import logging

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
