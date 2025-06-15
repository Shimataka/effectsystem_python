import urllib.request as request

from pyeffects.effect import Effect


class NetworkEffect(Effect[str]):
    """Effect that is a network request"""

    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.method = method
        self.headers = headers or {}

    def unwrap(self) -> str:
        """Execute the effect and return the result"""
        req = request.Request(self.url, headers=self.headers)  # noqa: S310
        with request.urlopen(req) as response:  # noqa: S310
            return response.read().decode("utf-8")  # type: ignore[no-any-return]


class DatabaseEffect(Effect[list[dict[str, str]]]):
    """Effect that is a database request"""

    def __init__(
        self,
        query: str,
        params: dict[str, str] | None = None,
    ) -> None:
        self.query = query
        self.params = params or {}

    def unwrap(self) -> list[dict[str, str]]:
        """Execute the effect and return the result"""
        return [{"id": "1", "name": "John"}, {"id": "2", "name": "Jane"}]
