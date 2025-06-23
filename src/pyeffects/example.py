import urllib.request as request

from pyeffects.base import Effect


class NetworkEffect(Effect[str]):
    """Effect that is a network request

    Args:
        url (str): The URL to request.
        method (str): The HTTP method to use.
        headers (dict[str, str] | None): The headers to send with the request.
    """

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
        """Execute the effect and return the result

        Returns:
            str: The result of the network request.
        """
        req = request.Request(self.url, headers=self.headers)  # noqa: S310
        with request.urlopen(req) as response:  # noqa: S310
            return response.read().decode("utf-8")  # type: ignore[no-any-return]


class DatabaseEffect(Effect[list[dict[str, str]]]):
    """Effect that is a database request

    Args:
        query (str): The SQL query to execute.
        params (dict[str, str] | None): The parameters to pass to the query.
    """

    def __init__(
        self,
        query: str,
        params: dict[str, str] | None = None,
    ) -> None:
        self.query = query
        self.params = params or {}

    def unwrap(self) -> list[dict[str, str]]:
        """Execute the effect and return the result

        Returns:
            list[dict[str, str]]: The result of the database request.
        """
        return [{"id": "1", "name": "John"}, {"id": "2", "name": "Jane"}]
