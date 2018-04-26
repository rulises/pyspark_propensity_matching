import warnings
import logging
import sys
from io import StringIO


log_stream = StringIO()
logging.basicConfig(stream=log_stream)
logging.warn("root_test")
log_stream.getvalue()


exception_logger = logging.getLogger(__name__)
exception_logger.error("excpetion_log_test")
log_stream.getvalue()


logging.captureWarnings(capture=True)
warning_logger = logging.getLogger('py.warnings')
warnings.warn('capture_warnings_test')
log_stream.getvalue()

original_hook = sys.excepthook
def exception_handler(exc_type, exc_value, exc_traceback):
    exception_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    original_hook(exc_type, exc_value, exc_traceback)
    return

sys.excepthook = exception_handler

assert 1==0, 'monkeypatch_test'
print(log_stream.getvalue())

out_str = log_stream.getvalue()
filename = "logs"

with open(filename, 'w+') as f:
    f.write(out_str)
    f.close()

sys.excepthook = exception_handler
sys.excepthook is exception_handler
