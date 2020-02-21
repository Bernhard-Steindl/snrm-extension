"""
Sets up application logging

Authors: Bernhard Steindl
"""
import logging
import logging.handlers
import sys
import os.path

# Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
logging.getLogger().setLevel(logging.NOTSET)

# Add stdout handler, with level INFO
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)-16s %(levelname)-8s %(name)-10s %(funcName)-15s %(message)s')
console.setFormatter(console_formatter)
logging.getLogger().addHandler(console)


# Add file rotating handler, with level DEBUG
log_file_name = os.path.join('logs', 'application.log')
#rotatingHandler = logging.handlers.RotatingFileHandler(filename=log_file_name, maxBytes=(1048576*5), backupCount=5)
#rotatingHandler.setLevel(logging.DEBUG)
#rotatingHandler.setFormatter(formatter)
#logging.getLogger().addHandler(rotatingHandler)

# Adds a timed file rotating handler which writes rotates files on midnight and adds suffix "%Y-%m-%d" to log file
rotating_handler = logging.handlers.TimedRotatingFileHandler(filename=log_file_name,when='midnight')
file_formatter = logging.Formatter('%(asctime)-16s %(levelname)-8s %(name)-10s %(funcName)-15s %(message)s')
rotating_handler.setFormatter(file_formatter)
rotating_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(rotating_handler)


def logger(filename):
    return logging.getLogger(os.path.basename(filename))