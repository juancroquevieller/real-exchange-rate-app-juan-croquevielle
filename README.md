# Parity · Currency lens

A responsive Streamlit app comparing bilateral nominal and real exchange rates using IMF data, with an explicitly conditional historical valuation view.

## Run and publish

Python 3.12 recommended. The assignment files `app.py`, `requirements.txt`, and `README.md` are at repository root.

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

On [Streamlit Community Cloud](https://share.streamlit.io), choose this repository, branch `main`, file `app.py`, and Python 3.12. No API key or Secrets is required. Keep `data/` for snapshot operation during API outages. `.streamlit/config.toml` sets the light theme.

## Use

Choose Country 1 (currency evaluated), Country 2 (comparison/home), a time frame, dates, and Real/Nominal/Both. The first chart has two automatic summaries. The second adds a historical RER reference and its nominal implication when supported. Research, trader, foreign debt/payments, and foreign assets/receipts perspectives tailor the interpretation. Download all observations and calculations as CSV; annual mode also offers its plotted annual aggregates.

## Economics

IMF FX inputs are local-currency units per USD:

```
S(t) = FX2(t) / FX1(t)             # Country 2 currency / Country 1 currency
q(t) = S(t) * CPI1(t) / CPI2(t)    # raw RER
q* = arithmetic mean of valid q(t) in selected sample
I(t) = 100 * q(t) / q*            # first-chart real index
```

UK/US means USD per GBP. Changing dates recalculates q*. The valid monthly real index averages 100. An index above 100 means above the selected sample average, **not** necessarily overvalued. Separate CPI base years are arbitrary; benchmark ratios remove fixed scale differences, not structural comparability problems. Nominal rates retain their currency units. Both uses separate axes.

Annual observations average monthly S, q and I; they do not multiply annual-average inputs. Partial-year coverage is displayed. The observations CSV remains monthly in annual mode for reproducibility. Annual index points need not average 100 without weighting for different monthly coverage.

Daily mode is a labeled extension: ECB business-day reference FX, crossed via EUR, combined with IMF CPI from two calendar months earlier. There is no daily CPI invention, interpolation, weekend FX filling, or indefinite CPI carry-forward. The fixed lag does not verify each country's actual release dates; revised CPI means this is **not a point-in-time trading backtest**. Unsupported daily pairs receive an explicit explanation.

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

Automatic references are restricted to pairs with documented pre-2008 monetary-regime coverage in Eichenbaum, Johannsen and Rebelo (2019), pp. 6–7: US, UK, Canada, Australia, New Zealand, Sweden and Germany. German national comparisons start at EUR adoption, further limiting eligible overlap. Other pairs get no invented regime history or automatic valuation label.

Search five complete calendar years, most recent first, ending by 2007 within both documented regimes. Require:

- All 60 monthly CPI/FX observations plus prior-year CPI.
- Both year-over-year inflation rates always 0–5%, each with standard deviation ≤2 percentage points.
- Annual growth rates all 0–8%; mean absolute bilateral growth difference ≤2 percentage points.
- RER coefficient of variation ≤10%; absolute fitted log trend ≤2% per year.

Use the median (default) or mean of the latest passing window. **These thresholds are app-design heuristics, not estimates or thresholds from the paper.** WEO GDP growth uses pre-2008 historical years only, avoiding projections. A passing period is an indicative anchor, not proof of equilibrium or current monetary-regime continuity. Failure means insufficient evidence in this app, not proof that no fair-value period exists.

The benchmark stays independent of viewing dates. If the selected endpoint predates completion of the reference, it is withheld to avoid look-ahead. Other historical plots are retrospective comparisons, not simulated trading results.

### EUR/USD course scenario

An explicit optional January 1999 reference implements the first homework's fair-pricing assumption when that month has joint data. The downloaded IMF euro-area HICP begins December 1999, so the current snapshot correctly reports the January anchor as unavailable. It does not silently substitute December or splice in another provider. If IMF supplies the missing month later, it uses those consistent series. Results can differ from the original FRED-based homework due to source series, CPI definitions and revisions. It is a **course assumption**, not a verified equilibrium observation.

### Automated insights

The neutral range combines the historical interquartile range and ±5% around the reference (adjustable). Historical shading is not a confidence interval. Labels are **Possibly undervalued/overvalued** or **Near reference range**. No statistical probability is assigned.

Trader suggestions are conditional multi-year buy/sell biases; below two years no trade signal is offered. They are not entry timing or guaranteed returns. Company perspectives assume the firm reports in Country 2 currency and owes/receives Country 1 currency. Payment/debt and receipt/asset exposures have opposite hedge directions. Shared-currency pairs get no bilateral FX trade or hedge suggestion. Insights are deterministic and AI-assisted in design; no live LLM or model API cost is involved.

The paper supports stronger medium/long-horizon predictability under inflation targeting and different adjustment under fixed exchange rates. It does not establish every historical mean as fair value. [Paper](https://www.kellogg.northwestern.edu/faculty/rebelo/htm/RER.pdf).

## Sources and updates

- [IMF SDMX](https://data.imf.org/en/Resource-Pages/IMF-API): CPI `ISO3.CPI._T.IX.M`; euro area uses `G163.HICP._T.IX.M`; period-average FX `ISO3.XDC_USD.PA_RT.M`; latest version `~`.
- [IMF DataMapper](https://www.imf.org/external/datamapper/api/help): countries and WEO historical `NGDP_RPCH` real GDP growth.
- [ECB](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html): daily reference FX where covered.

Default mode uses a verified, dated public-source snapshot for reproducible loading, **not a live feed**. `data/sources.json` records source URLs and UTC retrieval timestamps. Enable **Refresh from live sources** to attempt downloads; failures explicitly fall back to snapshots. Live results cache six hours (growth one day). Run `python refresh_data.py` and commit results to refresh bundled data. No scheduled updater or trading automation is configured.

## Verify and submit

`python -m unittest -v` checks quote direction, inverse pairs, average-100 rebasing, missing inputs, annual aggregation, daily lag, benchmark conversion/gap denominators, unsupported/look-ahead benchmarks, same-currency behavior and chart axes.

`PROMPTS.md` records AI assistance. The course also requires a submission document with deployed URL, GitHub URL, prompts/chat screenshots, model names and dates, a manual RER verification, plus an exported CSV. Course PDFs and personal files are intentionally outside this public repository.
