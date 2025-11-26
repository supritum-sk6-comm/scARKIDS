import logging


class Logger:
    """Project-wide logger helper."""

    _configured = set()

    @classmethod
    def get_logger(cls, name: str = __name__) -> logging.Logger:
        """Return a configured logger with the given name."""
        logger = logging.getLogger(name)
        if name not in cls._configured:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            cls._configured.add(name)
        return logger

    # Optional instance-style wrapper if you ever need it
    def __init__(self, name: str = __name__):
        self.logger = self.get_logger(name)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg, exc_info=True)
