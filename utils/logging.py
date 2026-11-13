import logging
import sys
import os

def setup_logger(name='GlaucomaScreening', log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            f_handler = logging.FileHandler(log_file, encoding='utf-8')
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)
    return logger

logger = setup_logger()
