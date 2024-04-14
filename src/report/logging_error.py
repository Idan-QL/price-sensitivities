"""Module of class ErrorCounter."""
import logging

class ErrorCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.error_count = 0

    def emit(self, record):
        if record.levelname == 'ERROR':
            self.error_count += 1