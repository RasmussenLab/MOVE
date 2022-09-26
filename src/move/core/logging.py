import logging
from pathlib import Path


def get_logger(name):
    if "." in name:
        name = name.split(".")[-1]

    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        path = Path.cwd() / "logs"
        path.mkdir(exist_ok=True)
        format = "[%(asctime)s] [%(levelname)-5s - %(name)s]: %(message)s"
        formatter = logging.Formatter(format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(path / f"{name}.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        format = "[%(levelname)-5s - %(name)s]: %(message)s"
        formatter = logging.Formatter(format)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
