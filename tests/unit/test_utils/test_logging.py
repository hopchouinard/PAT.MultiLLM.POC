# tests/unit/test_utils/test_logging.py

import pytest
import logging
import os
from unittest.mock import patch, Mock, call
from utils.logging import setup_logger, get_logger, set_log_level, logger

@pytest.fixture
def mock_logging():
    with patch('utils.logging.logging') as mock:
        yield mock

@pytest.fixture
def mock_config():
    with patch('utils.logging.get_config') as mock:
        mock.return_value.get.return_value = 'INFO'
        yield mock

def test_setup_logger(mock_logging, mock_config):
    logger = setup_logger('test_logger')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'test_logger'
    assert logger.level == logging.INFO
    mock_logging.StreamHandler.assert_called_once()
    mock_logging.Formatter.assert_called_once()

def test_setup_logger_with_file(mock_logging, mock_config):
    with patch('os.makedirs') as mock_makedirs:
        logger = setup_logger('test_file_logger', log_file='test.log')
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'test_file_logger'
    mock_logging.StreamHandler.assert_called_once()
    mock_logging.RotatingFileHandler.assert_called_once()
    mock_makedirs.assert_called_once()

def test_get_logger_new(mock_logging):
    with patch('utils.logging.setup_logger') as mock_setup:
        logger = get_logger('new_logger')
    
    mock_setup.assert_called_once_with('new_logger', log_file='logs/new_logger.log')
    assert isinstance(logger, logging.Logger)

def test_get_logger_existing(mock_logging):
    existing_logger = logging.getLogger('existing_logger')
    mock_logging.Logger.manager.loggerDict = {'existing_logger': existing_logger}
    
    logger = get_logger('existing_logger')
    
    assert logger is existing_logger

def test_set_log_level(mock_logging, mock_config):
    mock_handler1 = Mock()
    mock_handler2 = Mock()
    mock_logging.getLogger.return_value.handlers = [mock_handler1, mock_handler2]
    
    set_log_level('DEBUG')
    
    mock_logging.getLogger.return_value.setLevel.assert_called_once_with(logging.DEBUG)
    mock_handler1.setLevel.assert_called_once_with(logging.DEBUG)
    mock_handler2.setLevel.assert_called_once_with(logging.DEBUG)
    mock_config.return_value.set.assert_called_once_with('log_level', 'DEBUG')

def test_set_log_level_invalid():
    with pytest.raises(ValueError):
        set_log_level('INVALID_LEVEL')

def test_logger_debug(mock_logging):
    logger.debug("Debug message")
    mock_logging.getLogger.return_value.debug.assert_called_once_with("Debug message")

def test_logger_info(mock_logging):
    logger.info("Info message")
    mock_logging.getLogger.return_value.info.assert_called_once_with("Info message")

def test_logger_warning(mock_logging):
    logger.warning("Warning message")
    mock_logging.getLogger.return_value.warning.assert_called_once_with("Warning message")

def test_logger_error(mock_logging):
    logger.error("Error message")
    mock_logging.getLogger.return_value.error.assert_called_once_with("Error message")

def test_logger_critical(mock_logging):
    logger.critical("Critical message")
    mock_logging.getLogger.return_value.critical.assert_called_once_with("Critical message")

def test_logger_exception(mock_logging):
    try:
        raise ValueError("Test exception")
    except ValueError:
        logger.exception("Exception occurred")
    
    mock_logging.getLogger.return_value.exception.assert_called_once_with("Exception occurred")

@patch('utils.logging.RotatingFileHandler')
def test_rotating_file_handler(mock_rotating_handler, mock_logging, mock_config):
    setup_logger('rotating_logger', log_file='test_rotating.log')
    
    mock_rotating_handler.assert_called_once_with(
        'test_rotating.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )

@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_log_directory_creation(mock_makedirs, mock_exists, mock_logging, mock_config):
    setup_logger('dir_creation_logger', log_file='logs/test.log')
    
    mock_makedirs.assert_called_once_with('logs', exist_ok=True)

def test_multiple_loggers(mock_logging):
    logger1 = get_logger('logger1')
    logger2 = get_logger('logger2')
    
    assert logger1 != logger2
    assert logger1.name == 'logger1'
    assert logger2.name == 'logger2'

@patch('utils.logging.get_config')
def test_log_level_from_config(mock_get_config, mock_logging):
    mock_get_config.return_value.get.return_value = 'DEBUG'
    logger = setup_logger('config_logger')
    
    assert logger.level == logging.DEBUG

def test_logger_formatting(mock_logging):
    setup_logger('format_logger')
    formatter = mock_logging.Formatter.call_args[0][0]
    assert '%(asctime)s' in formatter
    assert '%(name)s' in formatter
    assert '%(levelname)s' in formatter
    assert '%(message)s' in formatter

@patch('utils.logging.logging.Logger.addHandler')
def test_multiple_handlers(mock_add_handler, mock_logging, mock_config):
    setup_logger('multi_handler_logger', log_file='test_multi.log')
    
    assert mock_add_handler.call_count == 2  # StreamHandler and FileHandler

@patch('utils.logging.logging.Logger.removeHandler')
@patch('utils.logging.logging.Logger.addHandler')
def test_set_log_level_handler_update(mock_add_handler, mock_remove_handler, mock_logging, mock_config):
    logger = setup_logger('level_update_logger')
    set_log_level('DEBUG')
    
    assert mock_remove_handler.call_count == 2  # Remove both handlers
    assert mock_add_handler.call_count == 2  # Add both handlers back

def test_logger_inheritance(mock_logging):
    parent_logger = get_logger('parent')
    child_logger = get_logger('parent.child')
    
    assert child_logger.parent == parent_logger

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])