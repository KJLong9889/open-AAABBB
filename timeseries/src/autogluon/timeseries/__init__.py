from autogluon.common.utils.log_utils import _add_stream_handler
__version__ = "1.3.0"
try:
    from .version import __version__
except ImportError:
    pass

from .dataset import TimeSeriesDataFrame
from .predictor import TimeSeriesPredictor

_add_stream_handler()


__all__ = ["TimeSeriesDataFrame", "TimeSeriesPredictor"]
