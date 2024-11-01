import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Union
from pathlib import Path
import json
from functools import lru_cache
import torch.cuda


class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels."""

    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}"
                    f"{levelname}"
                    f"{self.COLORS['RESET']}"
                )
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno
        }

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


@lru_cache(maxsize=None)
def get_logger(
        name: str,
        log_dir: Optional[str] = "logs",
        log_level: Optional[str] = None,
        use_json: bool = False
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        use_json: Whether to use JSON formatting

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Set log level
        level = getattr(logging, log_level or 'INFO')
        logger.setLevel(level)

        # Create formatters
        console_formatter = (
            JSONFormatter() if use_json
            else ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        )

        file_formatter = (
            JSONFormatter() if use_json
            else logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(pathname)s:%(lineno)d - %(message)s'
            )
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"hallucination_detector_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

    return logger


class ContextLogger:
    """Logger for tracking contextual information."""

    def __init__(self, name: str):
        self.logger = get_logger(f"Context_{name}")
        self.context_stack = []

    def push_context(self, context: str) -> None:
        """Add context to the stack."""
        self.context_stack.append(context)

    def pop_context(self) -> None:
        """Remove the last context from stack."""
        if self.context_stack:
            self.context_stack.pop()

    def log(
            self,
            message: str,
            level: str = 'INFO',
            extra: Optional[Dict] = None
    ) -> None:
        """Log message with current context."""
        context = '/'.join(self.context_stack)
        full_message = f"[{context}] {message}" if context else message

        log_func = getattr(self.logger, level.lower())
        if extra:
            log_func(full_message, extra=extra)
        else:
            log_func(full_message)


class PerformanceLogger:
    """Logger for tracking performance metrics."""

    def __init__(self, name: str):
        self.logger = get_logger(f"Performance_{name}")
        self.start_time = None
        self.metrics = []

    def start_timing(self) -> None:
        """Start timing a code block."""
        self.start_time = datetime.now()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_timing(
            self,
            operation: str,
            extra_metrics: Optional[Dict] = None
    ) -> None:
        """End timing and log results."""
        if self.start_time is None:
            self.logger.warning("end_timing called without start_timing")
            return

        duration = (datetime.now() - self.start_time).total_seconds()

        metrics = {
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }

        if torch.cuda.is_available():
            metrics.update({
                'cuda_memory_allocated': torch.cuda.memory_allocated() / 1024 ** 2,
                'cuda_max_memory': torch.cuda.max_memory_allocated() / 1024 ** 2
            })

        if extra_metrics:
            metrics.update(extra_metrics)

        self.metrics.append(metrics)

        self.logger.info(
            f"Operation '{operation}' completed in {duration:.2f}s",
            extra=metrics
        )


class DebugLogger:
    """Logger for debugging with detailed context."""

    def __init__(self, name: str):
        self.logger = get_logger(f"Debug_{name}")
        self.debug_context = {}

    def set_context(self, **kwargs) -> None:
        """Set debug context."""
        self.debug_context.update(kwargs)

    def clear_context(self) -> None:
        """Clear debug context."""
        self.debug_context.clear()

    def log(
            self,
            message: str,
            level: str = 'DEBUG',
            **kwargs
    ) -> None:
        """Log debug message with context."""
        context = {**self.debug_context, **kwargs}
        log_func = getattr(self.logger, level.lower())

        if context:
            log_func(f"{message} | Context: {context}")
        else:
            log_func(message)


def setup_global_logging(
        log_dir: str = "logs",
        log_level: str = "INFO",
        use_json: bool = False
) -> None:
    """Setup global logging configuration."""
    root_logger = get_logger(
        "hallucination_detector",
        log_dir=log_dir,
        log_level=log_level,
        use_json=use_json
    )

    # Set as root logger
    logging.root = root_logger
    logging.Logger.root = root_logger
    logging.Logger.manager.root = root_logger