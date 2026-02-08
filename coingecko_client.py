class CoinGeckoClient:
    def __init__(self, timeout_s: float = 30.0):
        self._timeout = timeout_s
        self._client = httpx.AsyncClient(timeout=self._timeout)

    def _headers(self) -> Dict[str, str]:
        if not API_KEY:
            return {}
        return {DEFAULT_KEY_HEADER: API_KEY}

    async def market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: str = "max",
        interval: Optional[str] = "daily",
        precision: Optional[str] = "full",
    ) -> Dict[str, Any]:
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        params: Dict[str, Any] = {"vs_currency": vs_currency, "days": days}
        if interval:
            params["interval"] = interval
        if precision:
            params["precision"] = precision

        r = await self._client.get(url, params=params, headers=self._headers())
        r.raise_for_status()
        return r.json()

    async def simple_price(self, ids: str, vs_currencies: str = "usd") -> Dict[str, Any]:
        url = f"{COINGECKO_BASE_URL}/simple/price"
        params = {"ids": ids, "vs_currencies": vs_currencies}

        r = await self._client.get(url, params=params, headers=self._headers())
        r.raise_for_status()
        return r.json()

    async def close(self):
        await self._client.aclose()
