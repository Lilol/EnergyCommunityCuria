from datetime import datetime
from logging import getLogger, INFO, StreamHandler, Formatter, FileHandler, DEBUG
from os import makedirs
from os.path import join

from utility import configuration


def get_logger_path(path=configuration.config.get("path", "log")):
    return join(path, f"log_{datetime.now().date():%Y%m%d}.txt")


def init_logger(mode="w"):
    try:
        # create root logger
        logger = getLogger()
        logger.setLevel(DEBUG)

        # Remove already existing handlers
        while len(logger.handlers):
            logger.handlers[0].close()
            logger.removeHandler(logger.handlers[0])

        # create formatter and add it to the handlers
        # create file handler which logs even debug messages
        logger_path = configuration.config.get("path", "log")
        makedirs(logger_path, exist_ok=True)
        filename = join(logger_path, f"{datetime.now():%Y%m%d}.txt")
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        try:
            fh = FileHandler(filename, mode)
            fh.setLevel(DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except (FileNotFoundError, PermissionError):
            print(f"Unable to init log file '{filename}'")

        # create console handler with a higher log level
        ch = StreamHandler()
        ch.setLevel(INFO)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(ch)
        return logger
    except Exception as e:
        print(f"Unable to initialize logger: '{e}'")
        raise e
