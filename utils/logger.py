from __future__ import annotations
import logging
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)


class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        message = super().format(record)
        return f"{color}{timestamp} | {record.levelname:<8} | {message}{Style.RESET_ALL}"


def get_logger(name: str = "bot") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(message)s"))
    logger.addHandler(handler)
    # Also log to file (unbuffered) for reliable log capture
    import os, sys
    log_file = os.environ.get("BOT_LOG_FILE")
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    logger.propagate = False
    return logger
