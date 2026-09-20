import httpx2

HTTPX_ERRORS = (
    httpx2.ConnectError,
    httpx2.ConnectTimeout,
    httpx2.RemoteProtocolError,
    httpx2.ReadError,
    httpx2.ReadTimeout,
    httpx2.HTTPStatusError,
)
