# -*- coding: utf-8 -*-
import logging
from tornado.log import enable_pretty_logging

class LogService():

    initFlag = False

    def __init__(self):
        self.logger = logging.getLogger('logger')
        enable_pretty_logging(logger=self.logger)
        LogService.initFlag = True

    def get_logger(self):
        return self.logger

    def info(self, log):
        self.logger.info(log)

    def error(self, log):
        self.logger.error(log)

