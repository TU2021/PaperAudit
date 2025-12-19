import logging
import os
import sys
from datetime import datetime

def _setup_logger(name: str, filename_prefix: str, console_output: bool = True):
    logger = logging.getLogger(name)
    
    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return logger
        
    # Check environment variable for logging level
    # If LOGGING_LEVEL is not set or empty, logging is disabled.
    log_level_env = os.getenv("LOGGING_LEVEL", "DEBUG").strip()
    
    if not log_level_env:
        # If logging is not enabled via environment variable, we disable it
        # by setting a level higher than CRITICAL or using NullHandler.
        logger.setLevel(logging.CRITICAL + 1)
        logger.addHandler(logging.NullHandler())
        return logger

    # If enabled, set level
    level_str = log_level_env.upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handlers_added = False

    # Console Handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        handlers_added = True

    # File Handler
    # Check if file logging is enabled via SAVE_LOG
    save_log = os.getenv("SAVE_LOG", "False").lower() in ('true', '1', 'yes', 'on')
    
    if save_log:
        # Project root is one level up from agents/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(project_root, "logs")
        
        if not os.path.exists(logs_dir):
            try:
                os.makedirs(logs_dir)
            except OSError:
                pass

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"{filename_prefix}_{timestamp}.log")
        
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            handlers_added = True
        except Exception as e:
            # If we can't write to file, we just rely on console
            print(f"Failed to setup file logging: {e}")

    # If no handlers were added (e.g. console_output=False and SAVE_LOG=False),
    # add NullHandler to prevent "No handlers could be found" warning
    if not handlers_added:
        logger.addHandler(logging.NullHandler())

    return logger

def get_logger(name: str):
    return _setup_logger(name, "s1_review", console_output=True)

def get_artifact_logger(name: str):
    return _setup_logger(name, "s1_artifacts", console_output=False)
