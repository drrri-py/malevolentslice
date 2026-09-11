import os
import logging
import sys
from typing import Optional

def get_logger(
    name: str = "malevolentslice",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """Configures and returns a logger instance for malevolentslice."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Adjust console handlers based on console_output flag
    has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    if console_output and not has_stream_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    elif not console_output and has_stream_handler:
        # Remove stream handlers when console output is suppressed
        logger.handlers = [h for h in logger.handlers if not (isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler))]

    if log_file:
        has_file_handler = any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(log_file) for h in logger.handlers)
        if not has_file_handler:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass

    return logger

