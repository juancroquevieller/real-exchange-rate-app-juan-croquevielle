"""Create the assignment check CSV and an independently recalculated example."""
from decimal import Decimal, getcontext
from pathlib import Path
import pandas as pd
import app

def main():
    output=Path(__file__).parent.parent/'Verification'
    output.mkdir(exist_ok=True)
    monthly,notes=app.monthly_pair('GBR','USA')
    selected,mean=app.normalize(monthly.loc['2016-10':'2026-09'])
    growth,_=app.growth_data()
    b=app.benchmark(monthly,'GBR','USA',growth)
    result=app.add_benchmark(selected,b,mean).reset_index()
    result.insert(1,'country1','GBR');result.insert(2,'country2','USA')
    result['quote_direction']='USD per GBP';result['frequency']='M'
    result['benchmark_start']=b.get('start','');result['benchmark_end']=b.get('end','')
    result['benchmark_method']=b.get('method','Unavailable');result['source_notes']=' | '.join(notes)
    csv=output/'GBP_USD_monthly_verification.csv'
    result.to_csv(csv,index=False)
    # Re-read the exported inputs as decimal strings, independently of app calculation functions.
    exported=pd.read_csv(csv,dtype=str).dropna(subset=['rer'])
    row=exported.iloc[-1]
    getcontext().prec=28
    p1,p2,x1,x2=[Decimal(row[x]) for x in ['cpi1','cpi2','fx1_per_usd','fx2_per_usd']]
    s=x2/x1; q=s*p1/p2
    qs=[Decimal(x) for x in exported.rer]
    avg=sum(qs)/Decimal(len(qs));index=100*q/avg
    exported_mean=sum(Decimal(x) for x in exported.real_index)/Decimal(len(qs))
    assert abs(q-Decimal(row.rer))<Decimal('1e-12')
    assert abs(index-Decimal(row.real_index))<Decimal('1e-10')
    assert abs(exported_mean-100)<Decimal('1e-10')
    report=f'''# RER app verification

Prepared October 25, 2026 (America/Chicago).

Repository: https://github.com/juancroquevieller/real-exchange-rate-app-juan-croquevielle

Deployment URL: pending Streamlit sign-in and deployment.

AI assistance: OpenAI Codex, GPT-6-based agent. Confirm the exact model variant in the app UI for submission. See PROMPTS.md in the project for the request and design decisions; keep screenshots of the original conversation as evidence.

## Manual reproduction from the exported CSV

Country 1: United Kingdom. Country 2: United States. Quote: **USD per GBP**.
Selection: October 2016–September 2026. Valid monthly RER observations: {len(exported)} of {len(result)}.
Example month: {row['date'][:7]} (monthly average FX, not a daily spot quote).

| Input | Value |
|---|---:|
| UK CPI | {p1} |
| US CPI | {p2} |
| GBP per USD | {x1} |
| USD per USD (identity) | {x2} |

1. S = FX2 / FX1 = {x2} / {x1} = **{s:.9f} USD per GBP**.
2. RER = S × UK CPI / US CPI = {s:.9f} × {p1} / {p2} = **{q:.9f}**.
3. Sample mean RER = **{avg:.9f}** across valid monthly observations.
4. Real index = 100 × RER / sample mean = **{index:.9f}**.
5. Mean of the exported real index = **{exported_mean:.12f}**, confirming 100 up to numeric precision.

Calculations above re-read the exported CSV and use decimal arithmetic independently of the app's calculation helpers. Missing observations remain blank and do not enter the mean.

## Reference check

Automatic reference: {b.get('label','Unavailable')}. Raw RER reference: {b.get('value',float('nan')):.9f}.
The reference is distinct from the selected-sample mean. It passes the disclosed historical screens, but does not establish true fair value or current regime continuity. Its implied nominal exchange rate is B × US CPI / UK CPI.

## Verification evidence

Economic tests cover quote direction, inverse pairs, average-100 rebasing, gaps, aggregation, daily lag, benchmark conversion, unsupported/look-ahead references and axes. Streamlit interaction tests exercise the three time frames, user profiles, Euro area, same-currency pairs, and unsupported daily data. Desktop, phone (390 px), and tablet (820 px) layouts were reviewed in Chrome.

The downloaded IMF euro-area HICP starts December 1999. The optional January 1999 course scenario therefore reports insufficient data instead of substituting a different period.

Sources: IMF CPI/ER/WEO and ECB daily FX. See the CSV's source notes and data/sources.json for exact retrieval timestamps. This file is preparation material; add the deployed Streamlit URL and your chat screenshots before submission.
'''
    (output/'Verification.md').write_text(report,encoding='utf-8')
    print('Created:',csv)
    print('Decimal-verified RER:',q,'; index:',index,'; mean index:',exported_mean)

if __name__=='__main__': main()
