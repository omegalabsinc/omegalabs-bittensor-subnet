class ChainTransactionError(Exception):
    """Error for any chain transaction related errors."""


class NetworkError(BaseException):
    """Base for any network related errors."""


class NetworkQueryError(NetworkError):
    """Network query related error."""


class NetworkTimeoutError(NetworkError):
    """Timeout error"""
