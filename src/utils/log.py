import logging
import sys
import threading

_logger = logging.getLogger()


# TODO: Add a way to release log handlers to allow unmounting GDrive on Colab


class ErrorsStats:
    COUNT = 0  # Tracks how many errors occurred

    @classmethod
    def register_error(cls):
        cls.COUNT += 1

    @classmethod
    def has_occurred(cls):
        return cls.COUNT > 0


def register_special_error_logger():
    global _logger
    error_handler = logging.FileHandler('log/ERRORS.log')
    error_handler.setFormatter(
        logging.Formatter('%(asctime)s %(message)s'))
    error_handler.setLevel(logging.ERROR)
    _logger.addHandler(error_handler)


def log(msg, log_level=None):
    global _logger
    if log_level is None:
        log_level = logging.INFO
    if log_level >= logging.ERROR:
        # add new logger if first error
        if not ErrorsStats.has_occurred():
            register_special_error_logger()
        ErrorsStats.register_error()
    _logger.log(log_level, msg)


def setup_log(filename=None, *, only_std_out=False):
    """
    Setups up logging handlers. Only needs to be called once
    """
    global _logger

    if _logger.hasHandlers():
        log('\n<<<<<<<<<<<<<<<<<<< CLOSING HANDLERS >>>>>>>>>>>>>>>>>>')
        handlers = _logger.handlers[:]  # Copy handlers list
        for handler in handlers:  # Iterate over copy to close and remove
            handler.close()
            _logger.removeHandler(handler)

    if filename is None:
        filename = 'run.log'

    if not only_std_out:
        # Set up a file handler
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s %(message)s'))
        file_handler.setLevel(logging.DEBUG)
        _logger.addHandler(file_handler)

    # Set up standard output
    std_stream_handler = logging.StreamHandler(sys.stdout)
    std_stream_handler.setFormatter(
        logging.Formatter('%(message)s'))
    std_stream_handler.setLevel(logging.DEBUG)
    _logger.addHandler(std_stream_handler)

    # Set up error output
    err_stream_handler = logging.StreamHandler(sys.stderr)
    err_stream_handler.setFormatter(
        logging.Formatter('%(levelname)s: %(message)s'))
    err_stream_handler.setLevel(logging.ERROR)
    _logger.addHandler(err_stream_handler)

    # Set default log level accepted
    _logger.setLevel(logging.INFO)  # Increased to stop matplotlib

    # Expected that it will show in file but not screen because each thread
    # creates this and many thread are created during training
    log(
        f'\n<<<<<<<<<<<<<<< Thread {threading.get_ident()} CONNECTED '
        f'to log  >>>>>>>>>>>>>>',
        logging.DEBUG)


def set_log_level(level):
    global _logger
    _logger.setLevel(level)


def disable_logging():
    set_log_level(logging.CRITICAL + 1)


def check_error():
    """
    See if any errors occurred and add warning to end of output
    NB: Expected to be called at or near end of execution to give user easy
    location to if an error occurred
    :return:
    """
    if ErrorsStats.has_occurred():
        log('!!! ERRORS OCCURRED !!!', logging.ERROR)
        # TODO make a warning sound
