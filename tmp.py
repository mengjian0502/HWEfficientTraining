import logging
logger = logging.getLogger('test')
logger.addHandler(logging.FileHandler('test.log'))
logger.addHandler(logging.StreamHandler())
logger.info("Hello")
logger.info("How are you?")
logger.warning("Warning1")
