"""Module of class ErrorCounter."""

import logging


class ErrorCounter(logging.Handler):
    """ErrorCounter class to count errors."""

    def __init__(self) -> None:
        """Initializes the ErrorCounter class."""
        super().__init__()
        self.error_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Emit the log record."""
        if record.levelname == "ERROR":
            self.error_count += 1
