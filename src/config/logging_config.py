import logging
import os
import yaml
from datetime import datetime
import sys

# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# config/logging_config.py
import logging
import colorlog
import os
from datetime import datetime
from src.config.config_settings import settings

class CustomLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, settings.LOG_LEVEL))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_filename = f"{settings.LOGS_DIR}/rag_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_formatter = logging.Formatter(settings.LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_filename = f"{settings.LOGS_DIR}/errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_filename, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def get_logger(self):
        return self.logger

# Factory function
def get_logger(name: str) -> logging.Logger:
    return CustomLogger(name).get_logger()

# Performance logging decorator
def log_execution_time(logger):
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


## Custom logger for track chaneges 
class CustomLoggerTracker:
    def __init__(self, config_path='logging_config.yaml'):
        """Initialize the custom logger with configuration."""
        self.config = self._load_config(config_path)
        self.loggers = {}
        self.base_log_dir = self.config.get('base_log_dir', 'logs')
        self._setup_base_directory()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Default configuration if file not found
            return {
                'base_log_dir': 'logs',
                'default_level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console_output': True,
                'modules': {
                    'main': {'level': 'INFO'},
                    'detection': {'level': 'INFO'},
                    'language_classification': {'level': 'INFO'},
                    'recognition': {'level': 'INFO'},
                    'registration': {'level': 'INFO'},
                    'metadata': {'level': 'INFO'}
                }
            }

    def _setup_base_directory(self):
        """Setup the base directory structure for logs."""
        if not os.path.exists(self.base_log_dir):
            os.makedirs(self.base_log_dir)

    def _get_log_path(self, module_name):
        """Generate the hierarchical path for log files."""
        now = datetime.now()
        year_dir = os.path.join(self.base_log_dir, str(now.year))
        month_dir = os.path.join(year_dir, f"{now.month:02d}")
        day_dir = os.path.join(month_dir, f"{now.day:02d}")
        os.makedirs(day_dir, exist_ok=True)
        return os.path.join(day_dir, f"{module_name}.log")

    def get_logger(self, module_name):
        """Get or create a logger for a specific module."""
        if module_name in self.loggers:
            return self.loggers[module_name]

        # Create new logger & Models Spceific Config
        logger = logging.getLogger(module_name)
        module_config = self.config['modules'].get(module_name, {})
        level = getattr(logging, module_config.get('level', self.config['default_level']))
        logger.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(self.config.get('format'))

        # Create file handler with the hierarchical path
        log_path = self._get_log_path(module_name)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optionally add console handler
        if self.config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        self.loggers[module_name] = logger
        return logger

    def update_config(self, new_config):
        """Update logger configuration."""
        self.config.update(new_config)
        # Reset all loggers to apply new configuration
        for module_name in self.loggers:
            logger = self.loggers[module_name]
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        self.loggers = {}
        
if __name__ == "__main__":
    # Create an instance first
    logger_tracker = CustomLoggerTracker()
    
    # Get a logger for the registration module
    logger = logger_tracker.get_logger("registration")
    
    # Test the logger
    logger.info("This is a test message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test another module
    detection_logger = logger_tracker.get_logger("detection")
    detection_logger.info("Detection module test message")