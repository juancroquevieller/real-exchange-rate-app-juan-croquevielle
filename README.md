# Parity · Currency lens

A responsive Streamlit app comparing bilateral nominal and real exchange rates using IMF data, with an explicitly conditional historical valuation view.

## Run and publish

Python 3.12 recommended. `app.py`, `requirements.txt`, and `README.md` are at repository root.

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

On [Streamlit Community Cloud](https://share.streamlit.io), choose this repository, branch `main`, file `app.py`, and Python 3.12. No API key or Secrets is required. Keep `data/` for snapshot operation during API outages. `.streamlit/config.toml` sets the light theme.

## Use

Choose Country 1 (currency evaluated), Country 2 (comparison/home), a time frame, dates, and Real/Nominal/Both. Monthly is the default; choices are ordered Daily, Monthly, Annual. English year/month/day selectors adapt to the frequency. Type a year to jump directly to it, or use 1Y/5Y/10Y/All. The first chart has two automatic summaries. The second adds a historical RER reference and its nominal implication when supported. Research, trader, foreign debt/payments, and foreign assets/receipts perspectives tailor the interpretation. Download all observations and calculations as CSV; annual mode also offers its plotted annual aggregates.

## Economics

Numbers use English comma thousands separators and a decimal point throughout the display. RER usually has one decimal, or two when the central 80% of the plotted variation spans less than one unit. Small raw RER levels retain at least two significant digits. Nominal quotes with typical magnitude at least 100 use whole units; ordinary quotes use two decimals; quotes below 0.1 gain decimal places to retain at least two significant digits (for example, 0.0011). These are presentation rules only. Charts use unrounded coordinates, all calculations use full precision, and CSV files retain unrounded numeric values without grouping commas.

IMF FX inputs are local-currency units per USD:

```
S(t) = FX2(t) / FX1(t)             # Country 2 currency / Country 1 currency
q(t) = S(t) * CPI1(t) / CPI2(t)    # raw RER
q* = arithmetic mean of valid q(t) in selected sample
I(t) = 100 * q(t) / q*            # first-chart real index
```

UK/US means USD per GBP. Changing dates recalculates q*. The valid monthly real index averages 100. An index above 100 means above the selected sample average, **not** necessarily overvalued. Separate CPI base years are arbitrary; benchmark ratios remove fixed scale differences, not structural comparability problems. Nominal rates retain their currency units. Both uses separate axes.

Annual observations average monthly S, q and I; they do not multiply annual-average inputs. Partial-year coverage is displayed. The observations CSV remains monthly in annual mode for reproducibility. Annual index points need not average 100 without weighting for different monthly coverage.

Daily mode is a labeled extension: BIS daily reference FX (ECB fallback), combined with CPI from two calendar months earlier. There is no daily CPI invention, interpolation, weekend FX filling, or indefinite CPI carry-forward. The fixed lag does not verify each country's actual release dates; revised CPI means this is **not a point-in-time trading backtest**. Unsupported daily pairs receive an explicit explanation. This is why Monthly remains the default even with broader daily FX coverage.

The date controls use available nominal history, keeping dates accessible when CPI is missing. The download starts in 1976 where available. Daily GBP/USD reaches 1976, CLP/USD August 1982, and PEN/USD August 1990. The euro itself starts in January 1999; no predecessor-currency backcast is presented as traded EUR. BIS may restate older observations in current currency units: historical levels should not be interpreted as executable quotes in the currency units circulating at that time.

Missing/nonpositive inputs are excluded from RER without filling across gaps. Nominal observations remain where available. National euro-member comparisons use EUR FX starting at euro adoption to avoid mixing national units with EUR. Other currencies may have redenominations, multiple rates or measurement breaks requiring further research.

## Historical reference

For fixed RER benchmark B:

```
Reference on first chart's scale = 100 * B / q*
PPP-implied S(t) = B * CPI2(t) / CPI1(t)
Valuation gap = 100 * (q(t)/B - 1)
Movement from observed S to implied S = 100 * (B/q(t) - 1)
```

The two percentages have different denominators. A low RER suggests Country 1 is cheap **conditional on the benchmark**. The RER reference stays horizontal; its nominal implication changes with relative CPI. Both charts retain the same q* scaling.

### Automatic screening

Automatic references screen every pair with sufficient data. The former country whitelist and pre-2008 cutoff were overly restrictive and have been removed. Documented regime starts still bound the search where available: the paper's histories, euro introduction in 1999, Chile's floating regime from September 1999, and Peru's inflation-targeting framework from January 2002. Unverified regime coverage is disclosed. Euro-area growth uses the IMF DataMapper aggregate **EURO**, not EUQ (Europe).

Search five complete calendar years, most recent first, ending at least two calendar years before the current year and no later than the displayed endpoint. Require:

- All 60 monthly CPI/FX observations plus prior-year CPI.
- Both year-over-year inflation means 0–6%; at least 90% of months within −1–8%; no absolute inflation above 12%; standard deviation ≤3 percentage points.
- Annual growth rates all −3–10%, nonnegative period means; mean absolute bilateral growth difference ≤3 percentage points and maximum yearly difference ≤6 points.
- RER coefficient of variation ≤15%; absolute fitted log trend ≤3% per year.

Use the median (default) or mean of the latest passing window. Current valuation never enters window ranking. Show all passing windows and the range of valuation gaps in the sensitivity panel. **These thresholds are app-design heuristics, not estimates or thresholds from the paper.** The two-year buffer excludes current/next-year WEO forecasts; historical observations can still be estimates and are revised. A passing period is an indicative anchor, not proof of equilibrium or current monetary-regime continuity. Failure means insufficient evidence in this app, not proof that no fair-value period exists. In the October 26 snapshot, Chile/US and Peru/US both qualify for 2015–2019.

Changing the start date does not affect the benchmark. The endpoint bounds eligible reference windows to avoid selecting a period completed after that endpoint. Historical plots remain retrospective comparisons, not simulated trading results.

### EUR/USD source selection and interpretation

For Euro area/US, select **IMF** or **FRED**. IMF supplies monthly CPI/HICP and period-average FX. FRED distributes BLS US CPI (CPIAUCNS), Eurostat 19-country euro-area HICP (CP0000EZ19M086NEST), and Federal Reserve noon New York USD/EUR observations (DEXUSEU), averaged by month. No FRED/IMF splicing occurs. Price-index definitions, geographic coverage, exchange-rate observation times, revisions and missing observations can affect the trend and gap. Neither provider establishes fair value.

Both sources default to a mean-100 index. Switching sources preserves scale, chart dates where coverage permits, reference approach and custom reference dates. Automatic remains the initial benchmark choice and independently screens each selected source. The explanation panel additionally compares sources using the latest common monthly observation, identical reference dates and the same statistic, without independently re-selecting the comparison source's benchmark.

Benchmark choice can matter much more than source choice. In the October 26 snapshot, the August 2026 valuation gap against a 2015–2019 median is −1.1% for IMF and −1.3% for FRED. Keeping FRED but changing the reference to January 1999 gives −10.9%. These are different economic assumptions about the reference level. Raw CPI base-year scaling changes chart levels, but cancels out of a consistently calculated benchmark ratio; a sample-mean index above 100 does not itself indicate overvaluation.

To explore a euro-launch assumption, select **FRED → Custom period → Use January 1999**. Use **All** for the full available history, and the sidebar to select raw RER if desired. IMF euro-area HICP starts in December 1999 in the bundled snapshot, so a January reference is unavailable for that source.

### Custom reference periods

Select **Custom period** and choose starting and ending year/month. Endpoints are inclusive and may be the same month. The reference is the median or mean of the raw monthly RER, independent of chart frequency and chart start date. The nominal implied path and valuation insights update from the unrounded reference. A custom period is a user assumption, not a successful automatic stability screen. Reversed dates, any missing monthly RER inputs, or a reference completed after the displayed endpoint produce a clear explanation and no valuation call. In daily mode a reference month must have ended by the displayed endpoint. The export records the approach, statistic and period.

As a reproducible calculation example, the FRED snapshot gives September 2026 observed FX of 1.150480952 USD/EUR and a January-1999-reference-implied rate of 1.300765911 USD/EUR: gap −11.5536%, required appreciation +13.0628%. This launch-date reference assumes fair pricing in January 1999; it is not independently established equilibrium.

### Automated insights

The neutral range combines the historical interquartile range and ±5% around the reference (adjustable). Historical shading is not a confidence interval. Labels are **Possibly undervalued/overvalued** or **Near reference range**. No statistical probability is assigned.

Trader suggestions are conditional multi-year buy/sell biases; below two years no trade signal is offered. They are not entry timing or guaranteed returns. Company perspectives assume the firm reports in Country 2 currency and owes/receives Country 1 currency. Payment/debt and receipt/asset exposures have opposite hedge directions. Shared-currency pairs get no bilateral FX trade or hedge suggestion. Insights are deterministic and AI-assisted in design; no live LLM or model API cost is involved.

The paper supports stronger medium/long-horizon predictability under inflation targeting and different adjustment under fixed exchange rates. It does not establish every historical mean as fair value. [Paper](https://www.kellogg.northwestern.edu/faculty/rebelo/htm/RER.pdf).

## Sources and updates

- [IMF SDMX](https://data.imf.org/en/Resource-Pages/IMF-API): CPI `ISO3.CPI._T.IX.M`; euro area uses `G163.HICP._T.IX.M`; period-average FX `ISO3.XDC_USD.PA_RT.M`; latest version `~`.
- [IMF DataMapper](https://www.imf.org/external/datamapper/api/help): countries and WEO historical `NGDP_RPCH` real GDP growth.
- [ECB](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html): daily reference FX where covered.
- [BIS bilateral exchange rates](https://data.bis.org/topics/XRU): daily currency units per USD, with source and break fields retained in the snapshot.
- [FRED US CPI](https://fred.stlouisfed.org/series/CPIAUCNS), [EA19 HICP](https://fred.stlouisfed.org/series/CP0000EZ19M086NEST), and [USD/EUR](https://fred.stlouisfed.org/series/DEXUSEU): alternative internally consistent EUR/USD source bundle.

Default mode uses a verified, dated public-source snapshot for reproducible loading, **not a live feed**. `data/sources.json` records source URLs and UTC retrieval timestamps. Enable **Refresh from live sources** to attempt downloads; failures explicitly fall back to snapshots. Live results cache six hours (growth one day). Run `python refresh_data.py` and `python refresh_extended_data.py` and commit results to refresh bundled data. No scheduled updater or trading automation is configured.

## Verification

`python -m unittest -v` checks quote direction, inverse pairs, average-100 rebasing, missing inputs, annual aggregation, daily lag, benchmark conversion/gap denominators, unsupported/look-ahead benchmarks, same-currency behavior and chart axes.

`PROMPTS.md` records AI assistance and development decisions. `verify_export.py` independently checks exported calculations using decimal arithmetic. Personal files remain outside this public repository.
