"""
Provides config values from config file

Authors: Bernhard Steindl
"""
from app_logger import logger

import json
import os.path
from typing import Union, Dict, ItemsView

class Config(object):

    def __init__ (self):
        self._logger = logger(__file__)
        self._config_file_name:str = os.path.join('config', 'config.json')
        self._config_data: Dict = {}
        self._logger.debug('Loading config file')
        try:
            with open(self._config_file_name, 'r') as c:
                self._config_data = json.load(c)
        except Exception as e:
            self._logger.exception('Could not load config')
            raise e

    def get(self, key:str ='') -> Union[str, int, float]:
        if key != '' and key in self._config_data:
            return self._config_data[key]
        else:
            error_msg = 'Could not find config value for key "{}"'.format(key)
            self._logger.error(error_msg)
            raise ValueError(error_msg)

    def contains(self, key:str ='') -> bool:
        return key in self._config_data

    def items(self) -> ItemsView:
        return self._config_data.items()

config = Config()