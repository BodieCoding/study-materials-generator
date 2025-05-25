import logging
import os
import streamlit as st

def get_logger(module_name):
    """
    Returns a logger object that logs to a file and to the Streamlit app.
    """
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    logger_name = f"streamlit_logger.{module_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    # Create a file handler
    log_file = os.path.join("logs", f"{module_name}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='w')
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Corrected log format
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a Streamlit handler
    streamlit_handler = StreamlitHandler(st.empty())
    streamlit_handler.setLevel(numeric_level)
    streamlit_handler.setFormatter(formatter)
    logger.addHandler(streamlit_handler)

    return logger

class StreamlitHandler(logging.Handler):
    def __init__(self, streamlit_container):
        logging.Handler.__init__(self)
        self.streamlit_container = streamlit_container

    def emit(self, record):
        msg = self.format(record)
        self.streamlit_container.write(msg)

if __name__ == '__main__':
    logger = get_logger(__name__)
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")