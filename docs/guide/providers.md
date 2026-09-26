# Data providers & terms of service

!!! note
    Stub page (E0-T04). Filled in by E2-T01…E2-T09 (provider adapters); completed in E8-T04.

regimekit fetches data from third-party providers (yfinance, stooq, FRED, CBOE, tiingo). Provider data is
never redistributed or committed to this repository (ADR-0002); you are responsible for complying with each
provider's own terms of service when you fetch and cache data locally.

## yfinance (default price provider)

`regimekit.finance.providers.yfinance.YFinanceProvider` downloads indices, ETFs and FX from Yahoo! Finance.
It needs the optional extra: `uv sync --extra providers` (`yfinance >= 1.7` and `curl_cffi`, used to
impersonate a browser session).

- Intervals `1h`, `1d`, `1wk`, `1mo` map one-to-one. `adjusted=True` requests split/dividend-adjusted
  prices (`auto_adjust=True`); `adjusted=False` requests the raw `Close`.
- `start`/`end` are both inclusive (yfinance's own `end` is exclusive; the adapter compensates).
- Intraday history is capped (`1h`: 730 days, table `INTRADAY_CAP_DAYS`). Older `start` dates are clipped with
  a `UserWarning` and a log line.
- Downloads are retried (`with_retries`, 5 attempts) and spaced at least 0.5 s apart.
- Multiple symbols are aligned on the union of bar times (NaN where missing); alignment to the master grid
  happens later, in the data-preparation step.
- Every fetch logs `provider="yfinance"` together with the terms-of-service note at INFO level.

!!! warning "Terms of service"
    Yahoo! data is licensed for personal, non-commercial use. regimekit never commits or redistributes it
    (ADR-0002); keep your local cache private and check Yahoo's current terms before any other use.

## stooq (daily fallback, EU coverage)

`regimekit.finance.providers.stooq.StooqProvider` is an end-of-day fallback for when Yahoo is unavailable.

- **API key.** Set the environment variable `STOOQ_API_KEY`. Without it the adapter raises a `ProviderError`
  with setup instructions *before* any network call. The key is never logged and never appears in errors.
- **Intervals.** Daily only (`supports_intraday = False`; `1h` raises). `1wk` and `1mo` are resampled from
  daily data as the last close of each period, labelled with the last actual trading date; `meta` records
  the resampling. The `[start, end]` window is applied to the daily data first, so a period cut by the
  window edge yields a partial first or last bar.
- **Adjustment.** Stooq closes are split-adjusted only, so `adjusted` is echoed but does not change prices.
- **Quota.** A client-side counter (default 100 calls per UTC day, `max_calls_per_day`) is persisted in
  `~/.cache/regimekit/stooq_quota.json` and shared by every adapter instance. One call is charged per
  symbol; retries are free. Exceeding it raises `ProviderError("stooq quota ...")`. Cache aggressively.
- **Symbols.** Yahoo-style names are mapped (`^GDAXI` -> `^dax`, `^GSPC` -> `^spx`, see `STOOQ_SYMBOLS`);
  anything else is passed through unchanged. Results keep the names you asked for.

!!! warning "Terms of service"
    Stooq data is for personal use under stooq.com's terms; do not redistribute it.
