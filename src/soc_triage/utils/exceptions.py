import sys


class SocTriageError(Exception):
    """Base class for all project errors."""


class ConfigError(SocTriageError):
    """Raised when a config file is missing or invalid."""


class DataValidationError(SocTriageError):
    """Raised when input data fails a quality check."""


def error_message_detail(error: Exception, error_detail=sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error in [{file_name}] line [{exc_tb.tb_lineno}]: {error}"


class PipelineError(SocTriageError):
    def __init__(self, error: Exception, error_detail=sys):
        self.message = error_message_detail(error, error_detail)
        super().__init__(self.message)
