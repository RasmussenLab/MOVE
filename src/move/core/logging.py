import logging
from pathlib import Path


def get_logger(name: str):
    """Return a logger with the specified name. The logger writes messages to
    a log file and the console.

    Args:
        name:
            Name of the logger. If it contains a dot, only the succeeding
            substring is used (e.g., `foo.bar` => `bar`).

    Returns:
        Logger
    """

    if "." in name:
        name = name.split(".")[-1]

    logging.captureWarnings(True)
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        path = Path.cwd() / "logs"
        path.mkdir(exist_ok=True)
        fmt = "[%(asctime)s] [%(levelname)-5s - %(name)s]: %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(path / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        fmt = "[%(levelname)-5s - %(name)s]: %(message)s"
        formatter = logging.Formatter(fmt)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
