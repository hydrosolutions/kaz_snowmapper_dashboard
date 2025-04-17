import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict

class LoggerSetup:
    """Handles logger initialization for the application."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config['logging']
        self.root_logger_name = 'snowmapper'

    def setup(self) -> logging.Logger:
        """Set up the root logger for the application."""
        # Create or get the root logger
        logger = logging.getLogger(self.root_logger_name)

        # Clear any existing handlers
        logger.handlers.clear()

        # Set base logging level
        log_level = getattr(logging, self.config['level'].upper())
        logger.setLevel(log_level)

        # Add handlers
        logger.addHandler(self._create_console_handler(log_level))
        logger.addHandler(self._create_file_handler(log_level))

        # Prevent propagation to root logger
        logger.propagate = False

        return logger

    def _create_console_handler(self, level: int) -> logging.Handler:
        """Create console handler."""
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(self._create_formatter())
        return handler

    def _create_file_handler(self, level: int) -> logging.Handler:
        """Create rotating file handler."""
        # Ensure log directory exists
        log_file = Path(self.config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=self.config['max_size_mb'] * 1024 * 1024,
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )
        handler.setLevel(level)
        handler.setFormatter(self._create_formatter())
        return handler

    def _create_formatter(self) -> logging.Formatter:
        """Create log formatter."""
        return logging.Formatter(self.config['format'])