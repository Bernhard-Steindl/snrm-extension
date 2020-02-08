"""
Lists config parameters and system info

Authors: Bernhard Steindl
"""

from app_logger import logger
logger = logger(__file__)

from config import config
import torch
import sys
import platform

logger.info('=====================')
logger.info('Printing config values')
for key, value in sorted(config.items()):
    logger.info('{} = {}'.format(key.upper(), value))
logger.info('=====================')

logger.info('Printing system info')
logger.info('Platform info = {}'.format(platform.platform()))
logger.info('Python system info = {}'.format(sys.version.replace('\n', ' ')))
logger.info('PyTorch version = {}'.format(torch.__version__))
logger.info('CUDA available = {}'.format(torch.cuda.is_available()))
logger.info('CUDA version = {}'.format(torch.version.cuda))
logger.info('cuDNN version = {}'.format(torch.backends.cudnn.version()))
logger.info('cuDNN enabled = {}'.format(torch.backends.cudnn.enabled))
logger.info('CUDA device count = {}'.format(torch.cuda.device_count()))
logger.info('=====================')

if not config.contains('run_name') or config.get('run_name').strip() == '':
    raise ValueError('The config value "run_name" should be given!')