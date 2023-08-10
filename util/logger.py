import logging
import time

logger = logging.getLogger('flakefinder')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh = logging.FileHandler('flakefinder.log')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s - %(module)s %(levelname)s]: %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)
