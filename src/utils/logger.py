import logging
import os
from pathlib import Path

class Logger:
    """Project-wide logger helper with file and console output."""
    
    _configured = set()
    _file_loggers = {}
    
    @classmethod
    def configure_file_logging(
        cls,
        session_name: str,
        log_dir: str = "scARKIDS/logs"
    ) -> str:
        """
        Configure file logging for a specific session.
        Creates one log file per session.
        
        Args:
            session_name: Name of the session (e.g., "unsupervised_dataset_1")
            log_dir: Directory to save logs (default: scARKIDS/logs)
        
        Returns:
            Path to the log file
        """
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Log file path (one per session)
        log_file = os.path.join(log_dir, f"{session_name}_training.log")
        
        # Root logger setup
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # File handler (DEBUG level to capture everything)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        cls._file_loggers[session_name] = log_file
        
        return log_file
    
    @classmethod
    def get_logger(cls, name: str = __name__) -> logging.Logger:
        """
        Return a configured logger with console output.
        
        Args:
            name: Logger name (typically __name__)
        
        Returns:
            Configured logging.Logger instance
        """
        logger = logging.getLogger(name)
        
        if name not in cls._configured:
            # Console handler (INFO level for clarity)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            
            cls._configured.add(name)
        
        return logger
    
    @classmethod
    def get_log_file(cls, session_name: str) -> str:
        """Get the log file path for a session."""
        return cls._file_loggers.get(session_name, None)
