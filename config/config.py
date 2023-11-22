from typing import Any, Union
from pathlib import Path
import sys
import functools
import logging
import atexit
import signal
import time
import datetime
import json
import pickle
import csv


class Config:
    """Class to manage application configuration and data persistence."""
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, settings_path: str = "settings.json", enable_logging: bool = True):
        if self._initialized:
            return
        
        self.start_time = time.time()
        
        self.base_path: Path = Path.cwd()
        self.settings_path: Path = self.base_path / settings_path
        self.settings: dict = self._load_settings()
        self.enable_logging = enable_logging
        
        self.id: Union[str, None] = self.settings.get("id", "")
        self.paths: dict = self.settings.get("paths", {})
        self.backend: Union[str, None] = self.settings.get("backend", {})
        self.simulation: Union[str, None] = self.settings.get("simulation", {})
        self.visualisation: Union[str, None] = self.settings.get("visualisation", {})
        
        self.folder_data: Path = self.paths.get("folder_data", "data")
        
        if self.id == "":
            self.folder_run: Path = self._create_folder(self.folder_data)
        elif (self.base_path / self.folder_data / self.id) in (self.base_path / self.folder_data).iterdir():
            self.folder_run: Path = self.base_path / self.folder_data / self.id
        else:
            raise ValueError(f"Invalid ID: {self.id}")
        
        self.id = self.folder_run.name
        self.file_log: str = self.paths.get("file_log", "log.log")
        self.log_path: Path = self.base_path / self.folder_run / self.file_log
        
        self.logger_instance = logging.getLogger(str(self.log_path))
        self._setup_logger()
        atexit.register(self._final_log)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self._initialized = True


    def __repr__(self) -> str:
        return f"Config({self.settings_path}, {self.settings})"
    
    def __str__(self) -> str:
        return f"Config({self.settings_path})"
        
    def _load_settings(self) -> dict:
        """Load settings from the provided path."""
        with self.settings_path.open('r') as file:
            return json.load(file)

    def _create_folder(self, folder_name: str) -> Path:
        """Create a new output folder and return its path."""
        dt = datetime.datetime.now()
        folder_path = self.base_path / folder_name / f"{dt.strftime('%Y%m%dT%H%M%S')}"
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    
    def save_data(self, data: Any, filename: str) -> None:
        """
        Save provided data to a file.
        
        Args:
            data (Any): Data to be saved.
            filename (str): Name of the file to save the data to.
        """
        ext = Path(filename).suffix
        filepath = self.folder_run / filename
        
        if ext == ".pkl":
            with filepath.open('wb') as file:
                pickle.dump(data, file)
        elif ext == ".json":
            with filepath.open('w') as file:
                json.dump(data, file)
        elif ext == ".csv":
            with filepath.open('w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in data:
                    writer.writerow(row)
        else:
            raise ValueError(f"Unknown or unsupported file extension: {filename}")

    def load_data(self, filename: str) -> Any:
        """
        Load data from a file.
        
        Args:
            filename (str): Name of the file to load data from.
            
        Returns:
            Any: Loaded data.
        """
        ext = Path(filename).suffix
        filepath = self.folder_run / filename
        
        if ext == ".pkl":
            with filepath.open('rb') as file:
                return pickle.load(file)
        elif ext == ".json":
            with filepath.open('r') as file:
                return json.load(file)
        elif ext == ".csv":
            with filepath.open('r') as csvfile:
                reader = csv.reader(csvfile)
                return list(reader)
        else:
            raise ValueError(f"Unknown or unsupported file extension: {filename}")

    def _setup_logger(self) -> None:
        """Setup the logger."""
        if self.enable_logging:
            if self.logger_instance.handlers:
                return

            handler = logging.FileHandler(self.log_path)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger_instance.addHandler(handler)
            self.logger_instance.setLevel(logging.INFO)
            self.logger_instance.info(f"Logging enabled. ({self.id})")
        else:
            logging.disable(logging.CRITICAL)

    
    def logger(self, func: Any) -> Any:
        """Logger decorator for function calls."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.enable_logging:
                self.logger_instance.info(f"Running {func.__name__} ...")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if self.enable_logging:
                    self.logger_instance.info(f"Finished {func.__name__} in {elapsed_time:.2f} seconds.")
                return result
        
            except Exception as e:
                if self.enable_logging:
                    self.logger_instance.error(f"Error in {func.__name__}: {e}")
                raise e
        return wrapper
    
    def log(self, message: str) -> None:
        """Log a custom message."""
        if self.enable_logging:
            self.logger_instance.info(message)
        else:
            print(message)

    def _final_log(self, status="completion") -> None:
        """Log a final message when the program stops."""
        stop_time = time.time()
        self.logger_instance.info(f"Logging stopped through {status} after {stop_time - self.start_time:.2f} seconds.")
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle received signals and perform cleanup."""
        self._final_log("termination")
        atexit.unregister(self._final_log)
        sys.exit(1)
        
