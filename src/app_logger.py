import sys 
from loguru import logger

logger.add(sys.stdout, level="WARNING", format="{time} {level} {message}") # Add console logging with WARNING level
logger.add("log.log", level="DEBUG", format="{time} {level} {message}") # Add file logging with DEBUG level

def log_application_start_end(image_path, model_name, num_classes):
    logger.info("Application started.")
    logger.info(f"Image path: {image_path}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info("Application ended.")

# Export only the logger
__all__ = ["logger"]


