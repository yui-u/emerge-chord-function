import logging
import logging.config


def create_logger(dest_log):
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    FORMAT = '%(asctime)s %(levelname)s %(message)s'

    dest_log.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    level = logging.INFO
    logging.basicConfig(level=level)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename=str(dest_log), mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger
