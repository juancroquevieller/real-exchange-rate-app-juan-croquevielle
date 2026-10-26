"""Parity: bilateral exchange rates, transparent benchmarks, and conditional insights."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import io
import json
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from refresh_extended_data import BIS_URL, FRED_CPI_URL, FRED_FX_URL, parse_bis

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
IMF = 'https://api.imf.org/external/sdmx/3.0'
PAPER = 'https://www.kellogg.northwestern.edu/faculty/rebelo/htm/RER.pdf'
TEAL, ORANGE, PURPLE = '#167565', '#d07931', '#7764a5'
CURRENCIES = dict(USA='USD', GBR='GBP', G163='EUR', CAN='CAD', AUS='AUD', NZL='NZD',
 JPN='JPY', CHE='CHF', CHL='CLP', CHN='CNY', COL='COP', BRA='BRL', MEX='MXN',
 SWE='SEK', NOR='NOK', DNK='DKK', IND='INR', IDN='IDR', KOR='KRW', ZAF='ZAR',
 HKG='HKD', SGP='SGD', THA='THB', MYS='MYR', PHL='PHP', ISR='ILS', TUR='TRY',
 CZE='CZK', HUN='HUF', POL='PLN', ROU='RON', ISL='ISK', ARG='ARS', PER='PEN')
EURO_START = dict(DEU=1999, FRA=1999, ITA=1999, ESP=1999, PRT=1999, IRL=1999,
 AUT=1999, BEL=1999, FIN=1999, LUX=1999, NLD=1999, GRC=2001, SVN=2007,
 CYP=2008, MLT=2008, SVK=2009, EST=2011, LVA=2014, LTU=2015, HRV=2023)
CURRENCIES.update({c:'EUR' for c in EURO_START})
ECB_CURRENCIES = set('EUR USD GBP CAD AUD NZD JPY CHF CNY BRL MXN SEK NOK DKK INR IDR KRW ZAR HKD SGD THB MYR PHP ILS TRY CZK HUF PLN RON ISK'.split())
# Historical regime starts: paper pp. 6-7 plus ECB, Chile and Peru central-bank histories.
# Do not infer that these histories certify a country's current regime.
REGIME_START = dict(USA='1982-10', GBR='1992-10', CAN='1991-04', AUS='1993-07',
 NZL='1990-01', SWE='1996-01', DEU='1982-10', G163='1999-01', CHL='1999-09', PER='2002-01')
MONTH_NAMES=['January','February','March','April','May','June','July','August','September','October','November','December']


def currency(code: str) -> str:
    return CURRENCIES.get(code, f'{code} currency')


def manifest() -> dict:
    path = DATA / 'sources.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def snapshot_note(name: str) -> str:
    return f"Snapshot retrieved {manifest().get(name, {}).get('retrieved_at', 'at an unknown time')[:10]} UTC"


def http_get(url: str, accept='text/csv') -> requests.Response:
    response = requests.get(url, headers={'Accept': accept}, timeout=(8, 35))
    response.raise_for_status()
    return response


@st.cache_data(ttl=21600, show_spinner=False)
def countries() -> dict:
    path = DATA / 'countries.json'
    if path.exists():
        raw = json.loads(path.read_text(encoding='utf-8'))
    else:
        raw = http_get('https://www.imf.org/external/datamapper/api/v1/countries','application/json').json()
    result = {k:(v.get('label') or k) for k,v in raw['countries'].items()}
    result['G163'] = 'Euro area'
    return result


def parse_imf(raw: pd.DataFrame, codes: tuple[str, ...]) -> pd.DataFrame:
    required = {'COUNTRY','TIME_PERIOD','OBS_VALUE'}
    if not required.issubset(raw.columns):
        raise ValueError('IMF response did not contain observation columns.')
    frame = raw.loc[raw.COUNTRY.isin(codes)].copy()
    frame['date'] = pd.to_datetime(frame.TIME_PERIOD.str.replace('-M','-',regex=False),errors='coerce')
    frame['value'] = pd.to_numeric(frame.OBS_VALUE,errors='coerce')
    if 'SCALE' in frame:
        frame['value'] *= 10.0 ** pd.to_numeric(frame.SCALE,errors='coerce').fillna(0)
    frame.loc[~np.isfinite(frame['value']) | (frame['value'] <= 0), 'value'] = np.nan
    if frame.duplicated(['COUNTRY','date']).any():
        raise ValueError('Ambiguous duplicate IMF observations; no series was silently averaged.')
    frame = frame[frame.date.notna() & (frame.date >= '1976-01-01')]
    return frame.pivot(index='date',columns='COUNTRY',values='value').sort_index()


@st.cache_data(ttl=21600, show_spinner=False)
def imf_series(flow: str, codes: tuple[str,...], live: bool) -> tuple[pd.DataFrame,str]:
    if flow=='CPI' and 'G163' in codes and len(codes)>1:
        national,n1=imf_series(flow,tuple(c for c in codes if c!='G163'),live)
        euro,n2=imf_series(flow,('G163',),live)
        return national.join(euro,how='outer'),n1+'; '+n2
    filename = 'cpi.csv.gz' if flow == 'CPI' else 'fx.csv.gz'
    suffix = 'CPI._T.IX.M' if flow == 'CPI' else 'XDC_USD.PA_RT.M'
    if flow=='CPI' and codes==('G163',):
        filename='euro_cpi.csv.gz'
        suffix='HICP._T.IX.M'
    url = f"{IMF}/data/dataflow/IMF.STA/{flow}/~/{'+'.join(codes)}.{suffix}"
    error = None
    if live or not (DATA / filename).exists():
        try:
            raw = pd.read_csv(io.StringIO(http_get(url).text),low_memory=False)
            frame = parse_imf(raw, codes)
            if frame.empty:
                raise ValueError('No observations returned for this selection.')
            return frame, f'Live IMF {flow} · retrieved {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}'
        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
            error = str(exc)
    if (DATA / filename).exists():
        raw = pd.read_csv(DATA / filename,low_memory=False)
        note = f'IMF {flow} · {snapshot_note(filename)}'
        if error:
            note += f' · Live request failed: {error[:160]}'
        return parse_imf(raw,codes), note
    raise ValueError(f'IMF {flow} unavailable. {error} No cached observations exist.')


@st.cache_data(ttl=21600, show_spinner=False)
def monthly_pair(c1: str,c2: str,live=False) -> tuple[pd.DataFrame,list[str]]:
    codes=tuple(sorted({c1,c2}))
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(imf_series,flow,codes,live) for flow in ['CPI','ER']]
        (cpi,n1),(fx,n2)=[j.result() for j in jobs]
    # A US dollar is exactly one US dollar, even when IMF omits that identity series.
    if 'USA' in codes:
        fx=fx.reindex(fx.index.union(cpi.index)).sort_index()
        fx['USA']=1.0
    idx=cpi.index.union(fx.index)
    if idx.empty:
        raise ValueError('No monthly CPI or FX observations are available for these countries.')
    index=pd.date_range(max(pd.Timestamp('1976-01-01'),idx.min()),idx.max(),freq='MS')
    frame=pd.DataFrame(index=index)
    for code,n in [(c1,1),(c2,2)]:
        frame[f'cpi{n}']=cpi[code] if code in cpi else np.nan
        frame[f'fx{n}_per_usd']=fx[code] if code in fx else np.nan
        # National euro-member series have historical currency changes. Use the common
        # euro rate only after adoption; do not present old national units as euros.
        if code in EURO_START:
            euro,_=imf_series('ER',('G163',),live)
            frame[f'fx{n}_per_usd']=euro.get('G163',pd.Series(dtype=float))
            frame=frame.loc[frame.index>=f'{EURO_START[code]}-01-01']
    frame['nominal']=frame.fx2_per_usd / frame.fx1_per_usd
    frame['rer']=frame.nominal * frame.cpi1 / frame.cpi2
    frame['valid_rer']=frame[['cpi1','cpi2','nominal','rer']].notna().all(axis=1)
    frame['cpi_month']=frame.index.strftime('%Y-%m')
    frame.index.name='date'
    return frame,[n1,n2]


@st.cache_data(ttl=21600, show_spinner=False)
def ecb_daily_fx(cur1: str,cur2: str,live=False) -> tuple[pd.DataFrame,str]:
    if not {cur1,cur2}.issubset(ECB_CURRENCIES):
        raise ValueError('Daily ECB reference-rate coverage is unavailable for this currency pair. Monthly IMF data remain available where reported.')
    path=DATA/'ecb.csv.gz'
    error=None
    raw=None
    if live or not path.exists():
        currencies='+'.join(sorted({cur1,cur2}-{'EUR'}))
        if not currencies:
            currencies='USD'  # supplies the official ECB business-day calendar
        url=f'https://data-api.ecb.europa.eu/service/data/EXR/D.{currencies}.EUR.SP00.A?startPeriod=1999-01-01&format=csvdata'
        try:
            raw=pd.read_csv(io.StringIO(http_get(url).text),low_memory=False)
            note=f'Live ECB reference rates · retrieved {datetime.now(timezone.utc):%Y-%m-%d}'
        except (requests.RequestException,ValueError) as exc:
            error=str(exc)
    if raw is None and path.exists():
        raw=pd.read_csv(path,low_memory=False)
        note=f'ECB reference rates · {snapshot_note("ecb.csv.gz")}'
        if error: note+=f' · Live request failed: {error[:160]}'
    if raw is None:
        raise ValueError(f'Daily rates unavailable: {error}')
    raw['date']=pd.to_datetime(raw.TIME_PERIOD,errors='coerce')
    raw['OBS_VALUE']=pd.to_numeric(raw.OBS_VALUE,errors='coerce')
    pivot=raw.pivot(index='date',columns='CURRENCY',values='OBS_VALUE').sort_index()
    pivot['EUR']=1.0
    if cur1 not in pivot or cur2 not in pivot:
        raise ValueError('The daily source has no observations for one of these currencies.')
    frame=pd.DataFrame({'fx1_per_eur':pivot[cur1],'fx2_per_eur':pivot[cur2]})
    frame=frame[(frame>0).all(axis=1)].dropna()
    frame['nominal']=frame.fx2_per_eur/frame.fx1_per_eur
    frame.index.name='date'
    return frame,note


@st.cache_data(ttl=21600,show_spinner=False)
def extended_csv(name: str,url: str,live=False) -> tuple[pd.DataFrame,str]:
    error=None
    if live or not (DATA/name).exists():
        try:
            raw=pd.read_csv(io.StringIO(http_get(url).text),low_memory=False)
            if name=='bis_daily.csv.gz': raw=parse_bis(raw)
            return raw,f'Live {url} · retrieved {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}'
        except (requests.RequestException,ValueError,KeyError) as exc:
            error=str(exc)
    if not (DATA/name).exists(): raise ValueError(f'{name} unavailable: {error}')
    note=f'{url} · {snapshot_note(name)}'
    if error: note+=f' · Live request failed; using snapshot: {error[:120]}'
    return pd.read_csv(DATA/name,low_memory=False),note


@st.cache_data(ttl=21600,show_spinner=False)
def daily_fx(cur1: str,cur2: str,live=False) -> tuple[pd.DataFrame,str]:
    raw,note=extended_csv('bis_daily.csv.gz',BIS_URL,live)
    raw=raw.loc[raw.currency.isin({cur1,cur2}-{'USD'})].copy()
    if {cur1,cur2}-{'USD'} <= set(raw.currency):
        raw['date']=pd.to_datetime(raw.date)
        raw['value']=pd.to_numeric(raw.value,errors='coerce')
        pivot=raw.pivot(index='date',columns='currency',values='value').sort_index()
        pivot['USD']=1.0
        frame=pd.DataFrame({'fx1_per_usd':pivot[cur1],'fx2_per_usd':pivot[cur2]})
        frame=frame[(frame>0).all(axis=1)].dropna()
        frame['nominal']=frame.fx2_per_usd/frame.fx1_per_usd
        frame.index.name='date'
        return frame,'BIS daily FX · '+note
    frame,ecb_note=ecb_daily_fx(cur1,cur2,live)
    return frame,ecb_note+' · BIS has no complete currency coverage for this pair.'


@st.cache_data(ttl=21600,show_spinner=False)
def fred_pair(c1: str,c2: str,live=False) -> tuple[pd.DataFrame,pd.DataFrame,list[str]]:
    """A separate, internally consistent source basis; never splice FRED CPI into IMF."""
    if {c1,c2}!={'G163','USA'}: raise ValueError('The FRED source option is available for EUR/USD only.')
    cp,n1=extended_csv('fred_cpi.csv.gz',FRED_CPI_URL,live)
    fx,n2=extended_csv('fred_eurusd.csv.gz',FRED_FX_URL,live)
    cp=cp.set_index(pd.to_datetime(cp.observation_date))
    fx=fx.set_index(pd.to_datetime(fx.observation_date))
    eur=pd.to_numeric(fx.DEXUSEU,errors='coerce').dropna()
    eur=eur[eur>0]
    monthly=pd.DataFrame(index=pd.date_range('1996-01-01',max(cp.index.max(),eur.index.max()),freq='MS'))
    monthly['cpi1']=pd.to_numeric(cp['CP0000EZ19M086NEST' if c1=='G163' else 'CPIAUCNS'],errors='coerce')
    monthly['cpi2']=pd.to_numeric(cp['CPIAUCNS' if c2=='USA' else 'CP0000EZ19M086NEST'],errors='coerce')
    # Average daily USD/EUR first; invert that monthly quote for the reverse pair.
    avg=eur.resample('MS').mean()
    monthly['fx1_per_usd']=1/avg if c1=='G163' else 1.0
    monthly['fx2_per_usd']=1.0 if c2=='USA' else 1/avg
    monthly['nominal']=monthly.fx2_per_usd/monthly.fx1_per_usd
    monthly['rer']=monthly.nominal*monthly.cpi1/monthly.cpi2
    monthly['valid_rer']=monthly.rer.notna()
    monthly['cpi_month']=monthly.index.strftime('%Y-%m')
    monthly.index.name='date'
    daily=pd.DataFrame({'nominal':eur if c1=='G163' else 1/eur})
    daily['fx1_per_usd']=1/eur if c1=='G163' else 1.0
    daily['fx2_per_usd']=1.0 if c2=='USA' else 1/eur
    daily.index.name='date'
    return monthly,daily,['FRED / BLS CPIAUCNS + Eurostat EA19 HICP CP0000EZ19M086NEST · '+n1,'FRED / Federal Reserve DEXUSEU · '+n2]


def make_daily(monthly: pd.DataFrame,fx: pd.DataFrame) -> pd.DataFrame:
    result=fx.copy()
    # Explicit lagged proxy: use CPI from two calendar months before the FX date.
    # This avoids using the current month's future CPI, but is not a vintage-data backtest.
    cpi_month=(result.index.to_period('M')-2).to_timestamp()
    for col in ['cpi1','cpi2']:
        result[col]=monthly[col].reindex(cpi_month).to_numpy()
    result['cpi_month']=cpi_month.strftime('%Y-%m')
    result['rer']=result.nominal*result.cpi1/result.cpi2
    result['valid_rer']=result.rer.notna()
    return result


def normalize(frame: pd.DataFrame) -> tuple[pd.DataFrame,float]:
    result=frame.copy()
    average=float(result.rer.mean())
    result['sample_rer_mean']=average
    result['real_index']=100*result.rer/average if average>0 else np.nan
    return result,average


def annual_view(frame: pd.DataFrame) -> pd.DataFrame:
    # Average monthly RERs; do not multiply annual-average FX by annual-average CPIs.
    cols=[c for c in frame.select_dtypes(include=[np.number]).columns if c!='valid_rer']
    out=frame[cols].resample('YS').mean()
    out['valid_months']=frame.rer.resample('YS').count()
    out['fx_months']=frame.nominal.resample('YS').count()
    out.index.name='date'
    return out


@st.cache_data(ttl=86400,show_spinner=False)
def growth_data(live=False) -> tuple[dict,str]:
    path=DATA/'growth.json'
    if not live and path.exists():
        return json.loads(path.read_text())['values']['NGDP_RPCH'],snapshot_note('growth.json')
    try:
        raw=http_get('https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH','application/json').json()
        return raw['values']['NGDP_RPCH'],'Live IMF WEO real GDP growth'
    except (requests.RequestException,KeyError,ValueError):
        if path.exists(): return json.loads(path.read_text())['values']['NGDP_RPCH'],snapshot_note('growth.json')
        return {},'IMF GDP growth is unavailable'


def screen_window(frame: pd.DataFrame,c1: str,c2: str,growth: dict,start: str,end: str) -> dict:
    window=frame.loc[start:end]
    reasons=[]
    expected=len(pd.date_range(start,end,freq='MS'))
    if expected<60 or len(window)!=expected or window.rer.notna().sum()!=expected:
        reasons.append('Fewer than five complete, consecutive years of joint CPI and exchange-rate observations.')
    inflation=frame[['cpi1','cpi2']].pct_change(12,fill_method=None).mul(100).loc[start:end]
    if inflation.isna().any().any() or len(inflation)!=expected:
        reasons.append('Insufficient prior-year CPI to assess inflation throughout the reference period.')
    elif ((inflation.mean()<0)|(inflation.mean()>6)).any() or (((inflation>=-1)&(inflation<=8)).mean()<.9).any() or (inflation.abs().max()>12).any() or (inflation.std()>3).any():
        reasons.append('Inflation fails the controlled-inflation screen: mean 0–6%, at least 90% of months within −1–8%, no absolute rate above 12%, and volatility ≤3 percentage points.')
    ys=range(pd.Timestamp(start).year,pd.Timestamp(end).year+1)
    # IMF SDMX uses G163; DataMapper's Euro area aggregate is EURO (EUQ is Europe).
    g1=[growth.get('EURO' if c1=='G163' else c1,{}).get(str(y)) for y in ys]
    g2=[growth.get('EURO' if c2=='G163' else c2,{}).get(str(y)) for y in ys]
    if any(x is None for x in g1+g2):
        reasons.append('Comparable historical GDP-growth observations are missing.')
        growth_gap=None
    else:
        growth_gap=float(np.mean(np.abs(np.array(g1)-np.array(g2))))
        if growth_gap>3 or np.max(np.abs(np.array(g1)-np.array(g2)))>6 or min(g1+g2)<-3 or max(g1+g2)>10 or min(np.mean(g1),np.mean(g2))<0:
            reasons.append('Growth paths fail comparability: mean absolute gap >3 pp, a yearly gap >6 pp, negative average growth, or a growth observation outside −3–10%.')
    valid=window.rer.dropna()
    dispersion=float(valid.std()/valid.mean()) if len(valid)>1 else math.inf
    trend=float(np.polyfit(np.arange(len(valid))/12,np.log(valid),1)[0]) if len(valid)>1 else math.inf
    if dispersion>0.15 or abs(trend)>0.03:
        reasons.append('RER dispersion exceeds 15% or its fitted trend exceeds 3% per year.')
    return dict(passed=not reasons,reasons=reasons,start=start,end=end,growth_gap=growth_gap,
                dispersion=dispersion,trend=trend,months=len(valid),inflation=inflation,window=window)


def benchmark(frame: pd.DataFrame,c1: str,c2: str,growth: dict,method='Median',course=False,as_of=None) -> dict:
    if course:
        when=pd.Timestamp('1999-01-01')
        if {c1,c2}!={'USA','G163'}:
            return dict(available=False,reason='The January 1999 launch reference applies only to EUR/USD.')
        if when not in frame.index or pd.isna(frame.loc[when,'rer']):
            return dict(available=False,reason='The January 1999 joint CPI/FX observation is unavailable.')
        value=float(frame.loc[when,'rer'])
        return dict(available=True,value=value,start='1999-01-01',end='1999-01-01',
                    label='January 1999 · launch reference',method='Single month',low=value,high=value,
                    reason='Assumes the euro was fairly priced at launch. This is a user-imposed reference, not an independently estimated equilibrium.',course=True)
    earliest=max(pd.Timestamp(REGIME_START.get(c1,'1976-01')),pd.Timestamp(REGIME_START.get(c2,'1976-01')),frame.index.min())
    # Extend beyond the paper's estimation sample. Use only complete years at least
    # two years behind the retrieval year, and never a period after the displayed endpoint.
    # The start filter does not affect candidate selection; no current gap enters ranking.
    last_year=min(datetime.now(timezone.utc).year-2,frame.index.max().year-1)
    if as_of is not None:
        endpoint=pd.Timestamp(as_of)
        last_year=min(last_year,endpoint.year if endpoint.month==12 else endpoint.year-1)
    attempts=[]
    qualified=[]
    for year in range(last_year,earliest.year+3,-1):
        start=f'{year-4}-01-01'; end=f'{year}-12-01'
        if pd.Timestamp(start)<earliest: continue
        result=screen_window(frame,c1,c2,growth,start,end)
        attempts.append(result)
        if result['passed']:
            window=result['window'].rer.dropna()
            value=float(window.median() if method=='Median' else window.mean())
            qualified.append(dict(value=value,start=start,end=end,low=float(window.quantile(.25)),high=float(window.quantile(.75)),
                diagnostics={k:v for k,v in result.items() if k not in ['window','inflation']}))
    if qualified:
        chosen=qualified[0]
        return dict(available=True,**chosen,label=f"{chosen['start'][:4]}–{chosen['end'][:4]} · screened historical reference",method=method,course=False,
            reason='Latest five-year period passing the disclosed inflation, growth and RER stability checks. Historical reference only: these app-designed screens do not establish equilibrium or verify the current monetary regime.',
            alternatives=[dict(start=q['start'],end=q['end'],value=q['value']) for q in qualified],
            regime_documented=c1 in REGIME_START and c2 in REGIME_START)
    if attempts:
        reason='No eligible five-year window passed all stability screens. Latest candidate: '+' '.join(attempts[0]['reasons'])
    else:
        reason='Insufficient five-year history before this endpoint and after any documented regime start.'
    return dict(available=False,reason=reason)


def add_benchmark(frame: pd.DataFrame,b: dict,mean: float) -> pd.DataFrame:
    result=frame.copy()
    if b.get('available'):
        result['benchmark_rer']=b['value']
        result['benchmark_real_index']=100*b['value']/mean
        result['implied_nominal']=b['value']*result.cpi2/result.cpi1
        result['gap_pct']=100*(result.rer/b['value']-1)
        result['return_to_reference_pct']=100*(b['value']/result.rer-1)
    return result


def display_decimals(values,kind='rer') -> int:
    """Choose display precision only; calculations and exports retain source precision."""
    a=np.asarray(values,dtype=float).ravel()
    a=a[np.isfinite(a)]
    nonzero=np.abs(a[a!=0])
    if not len(nonzero): return 1 if kind=='rer' else 2
    magnitude=float(np.median(nonzero))
    if kind=='nominal':
        # Large quotes (e.g. CLP per USD) need whole units; ordinary FX needs cents.
        return 0 if magnitude>=100 else max(2,1-math.floor(math.log10(magnitude)))
    decimals=1
    if len(a)>1 and 0<float(np.quantile(a,.9)-np.quantile(a,.1))<1:
        decimals=2
    # Small raw RER levels must not disappear into zero after rounding.
    return max(decimals,1-math.floor(math.log10(magnitude)))


def number(value,decimals=1,sign=False) -> str:
    if not np.isfinite(value): return 'Unavailable'
    return format(value,('+' if sign else '')+f',.{decimals}f')


def custom_benchmark(frame,start,end,method='Median',as_of=None,daily=False) -> dict:
    start=pd.Timestamp(start).to_period('M').to_timestamp()
    end=pd.Timestamp(end).to_period('M').to_timestamp()
    if start>end:
        return dict(available=False,reason='Reference start must be on or before reference end.')
    completed=end+pd.offsets.MonthEnd(0) if daily else end
    if as_of is not None and completed>pd.Timestamp(as_of):
        return dict(available=False,reason='The reference period ends after the displayed endpoint. Choose an earlier reference or extend the chart dates.')
    months=pd.date_range(start,end,freq='MS')
    values=frame.rer.reindex(months)
    missing=values.isna() | ~np.isfinite(values) | (values<=0)
    if missing.any():
        dates=', '.join(d.strftime('%b %Y') for d in months[missing][:3])
        return dict(available=False,reason=f'Joint CPI/FX data are missing for {int(missing.sum()):,} reference month(s): {dates}. Choose a complete period; missing months are not filled or silently omitted.')
    value=float(values.median() if method=='Median' else values.mean())
    label=start.strftime('%b %Y') if start==end else f'{start:%b %Y}–{end:%b %Y}'
    return dict(available=True,value=value,start=str(start.date()),end=str(end.date()),
                label=label+' · custom reference',method=method,low=float(values.quantile(.25)),
                high=float(values.quantile(.75)),custom=True,months=len(months),
                reason='Your selected period is a valuation assumption, not an automatically screened equilibrium estimate. Insights are conditional on this reference being economically appropriate.')


def compare_sources(imf,fred,start,end,method,as_of):
    """Hold month, quote, reference dates and statistic fixed across source bundles."""
    common=imf.dropna(subset=['rer']).index.intersection(fred.dropna(subset=['rer']).index)
    common=common[common<=pd.Timestamp(as_of)]
    if common.empty: return None,pd.DataFrame()
    when=common.max()
    rows=[]
    for name,frame in [('IMF',imf),('FRED',fred)]:
        anchor=custom_benchmark(frame,start,end,method,as_of=when)
        row=frame.loc[when]
        rows.append(dict(Source=name,Observed=row.nominal,
                         Implied=anchor['value']*row.cpi2/row.cpi1 if anchor['available'] else np.nan,
                         Gap=100*(row.rer/anchor['value']-1) if anchor['available'] else np.nan))
    return when,pd.DataFrame(rows)


def chart(frame: pd.DataFrame,mode: str,quote: str,mean: float,b=None,band=True,raw=False,frequency='Monthly') -> go.Figure:
    fig=make_subplots(specs=[[{'secondary_y':True}]])
    factor=1 if raw else 100/mean if mean>0 else np.nan
    real_name='Raw RER' if raw else 'Real index'
    real_values=frame.rer if raw else frame.real_index
    real_format=f',.{display_decimals(real_values)}f'
    nominal_format=f',.{display_decimals(frame.nominal,"nominal")}f'
    if mode in ['Real','Both']:
        fig.add_trace(go.Scatter(x=frame.index,y=real_values,name=real_name,mode='lines+markers' if real_values.count()<=2 else 'lines',
              line=dict(color=TEAL,width=2.6),connectgaps=False,hovertemplate='%{y:'+real_format+'}<extra>'+real_name+'</extra>'),secondary_y=False)
    if mode in ['Nominal','Both']:
        fig.add_trace(go.Scatter(x=frame.index,y=frame.nominal,name='Nominal rate',mode='lines+markers' if frame.nominal.count()<=2 else 'lines',
              line=dict(color=ORANGE,width=2.2),connectgaps=False,hovertemplate='%{y:'+nominal_format+'}<extra>'+quote+'</extra>'),secondary_y=mode=='Both')
    if b and b.get('available'):
        if mode in ['Real','Both']:
            fig.add_trace(go.Scatter(x=[frame.index[0],frame.index[-1]],y=[factor*b['value']]*2,
                  name='RER reference',line=dict(color=PURPLE,dash='dash'),hovertemplate='%{y:'+real_format+'}<extra>Fixed RER reference</extra>'),secondary_y=False)
            if band and b['low']<b['high']:
                fig.add_hrect(y0=factor*b['low'],y1=factor*b['high'],fillcolor=PURPLE,opacity=.09,line_width=0,secondary_y=False)
        if mode in ['Nominal','Both']:
            fig.add_trace(go.Scatter(x=frame.index,y=frame.implied_nominal,name='PPP-implied rate',
                  line=dict(color=PURPLE,dash='dot',width=2),hovertemplate='%{y:'+nominal_format+'}<extra>PPP-implied '+quote+'</extra>'),secondary_y=mode=='Both')
    fig.update_layout(height=320,margin=dict(l=10,r=10,t=12,b=5),paper_bgcolor='white',plot_bgcolor='white',
        font=dict(family='Arial, sans-serif',color='#526370',size=12),hovermode='x unified',
        legend=dict(orientation='h',y=-.18,x=0,font=dict(size=11)),
        hoverlabel=dict(bgcolor='white',font_size=12),dragmode='zoom',separators='.,')
    # Let Plotly recalculate tick spacing from the actual chart width and zoom.
    # Two-line labels keep months readable without crowding narrow screens.
    date_stops=[
        dict(dtickrange=[None,2419200000],value='%d %b\n%Y'),
        dict(dtickrange=[2419200000,'M11'],value='%b\n%Y'),
        dict(dtickrange=['M11',None],value='%Y'),
    ]
    hover_date='%d %B %Y' if frequency=='Daily' else '%Y (annual average)' if frequency=='Annual' else '%B %Y'
    fig.update_xaxes(showgrid=False,showline=True,linecolor='#e8edef',type='date',
        tickmode='auto',nticks=0,tickformatstops=date_stops,
        hoverformat=hover_date,tickangle=0,automargin=True)
    fig.update_yaxes(title_text=real_name if mode!='Nominal' else quote,gridcolor='#edf1f2',zeroline=False,tickformat=real_format if mode!='Nominal' else nominal_format,secondary_y=False)
    fig.update_yaxes(title_text=quote if mode=='Both' else None,showgrid=False,visible=mode=='Both',zeroline=False,tickformat=nominal_format,secondary_y=True)
    return fig


def valuation(frame: pd.DataFrame,b: dict,tolerance: float) -> dict:
    if not b.get('available'):
        return dict(status='No valuation call',detail=b['reason'],actionable=False)
    valid=frame.dropna(subset=['rer','nominal','cpi1','cpi2'])
    if valid.empty:
        return dict(status='No valuation call',detail='No matched CPI and nominal exchange-rate observation exists in this selection.',actionable=False)
    last=valid.iloc[-1]
    when=valid.index[-1]
    if when<pd.Timestamp(b['end']):
        return dict(status='No valuation call',detail='The selected endpoint predates completion of the reference period. Using this benchmark would introduce look-ahead.',actionable=False)
    gap=100*(last.rer/b['value']-1)
    low=min(b['low'],b['value']*(1-tolerance/100))
    high=max(b['high'],b['value']*(1+tolerance/100))
    status='Possibly undervalued' if last.rer<low else 'Possibly overvalued' if last.rer>high else 'Near reference range'
    return dict(status=status,detail=f'{gap:+,.1f}% versus the historical RER reference',actionable=True,
                gap=gap,implied=b['value']*last.cpi2/last.cpi1,adjustment=100*(b['value']/last.rer-1),
                when=when,spot=last.nominal,low=low,high=high)


def recommendation(v: dict,profile: str,cur1: str,cur2: str,horizon: str,same_currency=False) -> str:
    if same_currency:
        return 'Both countries use the same currency over this sample. There is no bilateral currency trade or FX hedge; RER changes reflect relative prices.'
    if not v['actionable']:
        if profile.startswith('Company'):
            return 'Hedge contractual cash-flow risk according to amounts and maturities. Without a defensible valuation reference, this app cannot suggest a valuation-based hedge tilt.'
        return 'No buy/sell conclusion is supported. Use the evolution chart to study relative prices and seek a documented reference period.'
    cheap=v.get('status')=='Possibly undervalued'; rich=v.get('status')=='Possibly overvalued'
    if profile=='Trader / investor':
        if horizon=='Under 2 years':
            return 'No short-horizon trade signal: this RER framework does not establish near-term entry or exit timing. Keep the valuation gap on a watchlist.'
        if cheap:
            return f'Conditional buy bias for {cur1} versus {cur2} over a multi-year horizon, if the historical anchor and monetary regime remain relevant. Check carry, transaction costs and structural changes before taking a position.'
        if rich:
            return f'Conditional sell or underweight bias for {cur1} versus {cur2} over a multi-year horizon, if the historical anchor and monetary regime remain relevant. Check carry, transaction costs and structural changes first.'
        return 'No directional valuation edge within the reference range. Avoid treating the sample-average line as a trading target.'
    if profile=='Company · payments / debt in C1':
        return (f'Your {cur2}-based company must buy {cur1} for payments or debt. '+
                (f'Potential {cur1} appreciation increases that liability: consider locking scheduled purchases with forwards or capped-cost options.' if cheap else
                 'Match hedges to contracted payment dates; possible depreciation is not a reason to leave committed liabilities unprotected.'))
    if profile=='Company · receipts / assets in C1':
        return (f'Your {cur2}-based company will receive or holds assets in {cur1}. '+
                (f'Potential {cur1} depreciation threatens translated value: consider selling expected receipts forward or buying downside protection.' if rich else
                 'Use a planned hedge ratio for budget certainty; retain any upside only within your risk policy.'))
    return 'Interpret the gap as a conditional medium- to long-run relative-price signal. Check sensitivity to the reference method, inflation, growth and monetary-policy regime.'


def date_controls(low,high,frequency,key,cur1,cur2,full=False):
    """English-only selectors avoid locale-dependent calendars and long month lists."""
    controls=st.container(key='date_controls').columns([2,2,2.2],gap='medium')
    initial=low if full else max(low,high-pd.DateOffset(years=1 if frequency=='Daily' else 10))
    def set_range(years=None):
        start=low if years is None else max(low,high-pd.DateOffset(years=years))
        for side,when in [('start',start),('end',high)]:
            for part,value in [('year',when.year),('month',when.month),('day',when.day)]:
                st.session_state[f'{side}_{part}_{key}']=value
    for side,when in [('start',initial),('end',high)]:
        for part,value in [('year',when.year),('month',when.month),('day',when.day)]:
            st.session_state.setdefault(f'{side}_{part}_{key}',value)
        year_key=f'{side}_year_{key}'
        st.session_state[year_key]=min(high.year,max(low.year,st.session_state[year_key]))
    values=[]
    for col,side,title in [(controls[0],'start','From'),(controls[1],'end','Through')]:
        parts=col.columns(3 if frequency=='Daily' else 2 if frequency=='Monthly' else 1,gap='small')
        year=parts[0].selectbox(f'{title} year',list(range(low.year,high.year+1)),key=f'{side}_year_{key}',help='Type a year to jump directly to it.')
        if frequency=='Annual':
            values.append(pd.Timestamp(year,1,1) if side=='start' else pd.Timestamp(year,12,31)); continue
        months=list(range(low.month if year==low.year else 1,(high.month if year==high.year else 12)+1))
        mk=f'{side}_month_{key}'
        st.session_state[mk]=max(min(st.session_state[mk],max(months)),min(months))
        month=parts[1].selectbox(f'{title} month',months,format_func=lambda m:MONTH_NAMES[m-1],key=mk)
        when=pd.Timestamp(year,month,1)
        if frequency=='Daily':
            days=list(range(low.day if (year,month)==(low.year,low.month) else 1,(high.day if (year,month)==(high.year,high.month) else when.days_in_month)+1))
            dk=f'{side}_day_{key}'
            st.session_state[dk]=max(min(st.session_state[dk],max(days)),min(days))
            day=parts[2].selectbox(f'{title} day',days,key=dk)
            when=pd.Timestamp(year,month,day)
        elif side=='end': when+=pd.offsets.MonthEnd(0)
        values.append(when)
    controls[2].caption('Quick range')
    for col,label,years in zip(controls[2].columns(4,gap='small'),['1Y','5Y','10Y','All'],[1,5,10,None]):
        col.button(label,key=f'range_{label}_{key}',on_click=set_range,args=(years,),width='stretch')
    controls[2].caption(f'1 {cur1} = S {cur2} · rising S means {cur1} appreciation')
    st.caption(f'Available FX history: {low:%d %b %Y} – {high:%d %b %Y}. Select All for the widest view; missing CPI remains blank.')
    return values


def main():
    st.set_page_config(page_title='Parity · Currency lens',page_icon='↔',layout='wide',initial_sidebar_state='collapsed')
    st.markdown('''<style>
    .block-container{padding-top:3.25rem;padding-bottom:2rem;max-width:1540px}
    h1{font-size:1.95rem!important;letter-spacing:-.065rem;padding-top:0!important}
    h2{font-size:1.25rem!important;letter-spacing:-.025rem}
    h3{font-size:1.05rem!important}
    [data-testid="stMetricValue"]{font-size:1.6rem}
    [data-testid="stMetricLabel"]{font-size:.8rem;color:#647683}
    [data-testid="stVerticalBlock"]{gap:.65rem}
    [data-testid="stBaseButton-segmented_controlActive"]{background:#e6f2ee}
    .eyebrow{color:#167565;letter-spacing:.14em;font-size:.7rem;font-weight:700;margin-bottom:.15rem}
    .subline{color:#6b7c86;font-size:.9rem;margin:0 0 .5rem}
    .rule{border-top:1px solid #e7edef;margin:1rem 0 .8rem}
    .insight{border-left:3px solid #167565;background:#f5f9f8;padding:.8rem 1rem;border-radius:0 8px 8px 0}
    @media(min-width:761px) and (max-width:1000px){
      .st-key-country_controls [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:1fr 1fr;gap:10px!important}
      .st-key-country_controls [data-testid="stColumn"]{width:100%!important;min-width:0!important}
    }
    @media(max-width:1000px){
      .st-key-benchmark_dates [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:1fr 1fr;gap:10px!important}
      .st-key-benchmark_dates [data-testid="stColumn"]{width:100%!important;min-width:0!important}
      .st-key-benchmark_panel [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:1fr;gap:10px!important}
      .st-key-benchmark_panel [data-testid="stColumn"]{width:100%!important;min-width:0!important}
      .st-key-date_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:1fr 1fr;gap:10px!important}
      .st-key-date_controls [data-testid="stColumn"]{width:100%!important;min-width:0!important}
      .st-key-date_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3){grid-column:1/-1}
    }
    @media(max-width:760px){
      .block-container{padding:3rem 1rem 1rem}.stPlotlyChart{min-width:0}h1{font-size:1.4rem!important}
      .subline,.st-key-summary_metrics{display:none}
      .st-key-reference_controls [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:1.3fr 1fr;gap:10px!important}
      .st-key-reference_controls [data-testid="stColumn"]{width:100%!important;min-width:0!important}
      .st-key-reference_controls [data-testid="stColumn"]:nth-child(3){grid-column:1/-1}
      .st-key-country_controls [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:1fr 1fr;gap:10px!important}
      .st-key-country_controls [data-testid="stColumn"]{width:100%!important;min-width:0!important}
      .st-key-date_controls [data-testid="stHorizontalBlock"]{display:flex!important;flex-direction:row!important;gap:6px!important}
      .st-key-date_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:1fr;gap:10px!important}
      .st-key-date_controls [data-testid="stHorizontalBlock"] [data-testid="stColumn"]{flex:1 1 0!important}
      .st-key-country_controls [data-testid="stWidgetLabel"] p{font-size:.75rem}
      .st-key-country_controls [data-testid="stButtonGroup"] button{padding:4px 7px;font-size:.75rem}
    }
    </style>''',unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">PARITY / CURRENCY LENS</div>',unsafe_allow_html=True)
    st.title('Exchange rates, in perspective.')
    st.markdown('<p class="subline">Explore exchange rates, separate prices from currency moves, and put valuation in context.</p>',unsafe_allow_html=True)
    try:
        names=countries()
    except Exception as exc:
        st.error(f'Country catalogue unavailable: {exc}'); return
    options=sorted(names,key=lambda x:names[x])
    c=st.container(key='country_controls').columns([2.2,2.2,1.5,1.8],gap='medium')
    c1=c[0].selectbox('Country 1 · currency to evaluate',options,index=options.index('GBR'),format_func=lambda x:names[x],key='c1')
    c2=c[1].selectbox('Country 2 · comparison / home',options,index=options.index('USA'),format_func=lambda x:names[x],key='c2')
    frequency=c[2].selectbox('Time frame',['Daily','Monthly','Annual'],index=1,key='frequency')
    mode=c[3].segmented_control('Show',['Real','Nominal','Both'],default='Both',selection_mode='single',key='mode') or 'Both'
    if c1==c2:
        st.info('Choose two different countries to compare their currencies.'); return
    cur1,cur2=currency(c1),currency(c2)
    quote=f'{cur2} per {cur1}'
    use_fred=False
    if {c1,c2}=={'G163','USA'}:
        basis=st.selectbox('EUR / USD data source',['IMF','FRED'],key='basis',
            help='IMF: monthly CPI/HICP and period-average exchange rates. FRED: BLS US CPI, Eurostat 19-country euro-area HICP and Federal Reserve daily FX averaged by month. Each option uses a consistent set of series.')
        use_fred=basis.startswith('FRED')
        st.caption('Source choice affects series coverage, release timing and measurement. Benchmark choice determines the valuation reference. See “How sources and benchmarks affect the result” below.')
    with st.sidebar:
        st.subheader('Data & methodology')
        live=st.toggle('Refresh from live sources',value=False,key='live',help='Off uses the dated, verified public-source snapshot bundled with this app. On attempts live downloads and falls back explicitly if they fail.')
        raw_scale=st.selectbox('RER scale',['Index · selected mean = 100','Raw RER · original CPI units'],index=0,key='rer_scale').startswith('Raw')
        st.markdown('**Monthly (default):** IMF CPI and period-average FX from 1976 where reported; optional FRED sources for EUR/USD. **Annual:** averages of monthly calculations. **Daily:** BIS reference FX with ECB fallback, and CPI lagged two months; optional FRED EUR/USD. Daily coverage varies and RER is a proxy.')
        st.latex(r'S_t=\frac{FX_{2/USD,t}}{FX_{1/USD,t}},\quad q_t=S_t\frac{CPI_{1,t}}{CPI_{2,t}},\quad I_t=100\frac{q_t}{\overline q_{selected}}')
        st.caption('Country 1 appreciation raises S. An index above 100 means above the selected-sample average, not necessarily overvalued. CPI index levels are not comparable absolute price levels; only consistently constructed relative changes and benchmark ratios are meaningful.')
        st.markdown(f'[IMF data API](https://data.imf.org/en/Resource-Pages/IMF-API) · [ECB reference rates](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html) · [Eichenbaum, Johannsen & Rebelo (2019)]({PAPER})')
    try:
        with st.spinner('Aligning CPI and exchange-rate observations…'):
            if use_fred: monthly,fred_fx,notes=fred_pair(c1,c2,live)
            else: monthly,notes=monthly_pair(c1,c2,live)
            base=monthly
            if use_fred and frequency!='Daily':
                # Monthly FRED coverage ends at the latest month with both CPI series.
                base=monthly.loc[:monthly.dropna(subset=['rer']).index.max()]
            if frequency=='Daily':
                if use_fred: fx,note=fred_fx,'Daily FRED DEXUSEU; monthly mode averages this series.'
                else: fx,note=daily_fx(cur1,cur2,live)
                if c1 in EURO_START: fx=fx.loc[f'{EURO_START[c1]}-01-01':]
                if c2 in EURO_START: fx=fx.loc[f'{EURO_START[c2]}-01-01':]
                base=make_daily(monthly,fx);notes.append(note)
    except (ValueError,requests.RequestException,KeyError) as exc:
        st.error(str(exc));st.info('Choose another pair or time frame, or try refreshing the live sources. Missing data are never filled with invented observations.');return
    usable=base.dropna(subset=['nominal'])
    if usable.empty:
        st.warning('No overlapping observations for this pair and chart mode. Try Nominal if CPI is missing, or choose another pair.');return
    low,high=usable.index.min(),usable.index.max()
    key=f'{c1}_{c2}_{frequency}'
    start,end=date_controls(low,high,frequency,key,cur1,cur2)
    if start>end:
        st.warning('The start date must be on or before the end date.');return
    sample,mean=normalize(base.loc[start:end])
    if sample.empty or sample.nominal.dropna().empty:
        st.info('No published observations within these dates. Choose a wider range.');return
    display=annual_view(sample) if frequency=='Annual' else sample
    real=sample.dropna(subset=['rer']); nominal=sample.nominal.dropna()
    latest_real=real.iloc[-1] if not real.empty else None
    nominal_decimals=display_decimals(sample.nominal,'nominal')
    real_decimals=display_decimals(sample.rer if raw_scale else sample.real_index)
    metrics=st.container(key='summary_metrics').columns(4)
    metrics[0].metric(f'Latest {quote}',number(nominal.iloc[-1],nominal_decimals))
    metrics[1].metric('Latest raw RER' if raw_scale else 'Latest real index',number(latest_real.rer if raw_scale else latest_real.real_index,real_decimals) if latest_real is not None else 'Unavailable')
    metrics[2].metric('Nominal change in selection',number(100*(nominal.iloc[-1]/nominal.iloc[0]-1),sign=True)+'%')
    metrics[3].metric('Matched observations',f'{len(real):,} / {len(sample):,}')
    datefmt='%d %b %Y' if frequency=='Daily' else '%b %Y'
    st.caption(f'Latest FX: {nominal.index[-1].strftime(datefmt)}' + (f' · Latest RER inputs: {real.index[-1].strftime(datefmt)}' if not real.empty else '') + ' · '+('Monthly observations' if frequency=='Annual' else frequency+' observations'))
    st.caption(('Live source refresh enabled' if live else snapshot_note('fred_cpi.csv.gz' if use_fred else 'cpi.csv.gz'))+' · '+('Raw RER in source CPI units' if raw_scale else 'RER index: selected-sample mean = 100')+' · Data and scale settings in the sidebar')
    if frequency=='Daily':
        st.caption('Daily RER proxy = published daily FX × CPI from two months earlier. Monthly combines FX and CPI for the same month; revised CPI and unverified release dates prevent a point-in-time backtest.')
    if cur1==cur2:
        st.info('Same currency, different economies: the nominal rate is 1. RER movements measure relative inflation, not an exchange-rate trading opportunity.')
    st.subheader('01  /  Exchange-rate evolution')
    st.plotly_chart(chart(display,mode,quote,mean,raw=raw_scale,frequency=frequency),width='stretch',config={'displaylogo':False,'scrollZoom':False},key='evolution')
    if latest_real is not None:
        change=100*(real.rer.iloc[-1]/real.rer.iloc[0]-1)
        relative=100*((real.cpi1.iloc[-1]/real.cpi1.iloc[0])/(real.cpi2.iloc[-1]/real.cpi2.iloc[0])-1)
        fxchange=100*(real.nominal.iloc[-1]/real.nominal.iloc[0]-1)
        scale_note=('Raw RER levels depend on the CPI bases; valuation requires a consistently constructed reference.' if raw_scale else f'The real index averages **100 across valid {"daily proxy observations" if frequency=="Daily" else "months"}** in this selection; that average is not a fair-value estimate.')
        st.markdown(f'- Across matched observations, **{cur1} {"appreciated" if fxchange>=0 else "depreciated"} {abs(fxchange):,.1f}% nominally**; its RER {"rose" if change>=0 else "fell"} **{abs(change):,.1f}%** ({"real appreciation" if change>=0 else "real depreciation"}).\n- Relative CPI in Country 1 {"rose" if relative>=0 else "fell"} **{abs(relative):,.1f}%** versus Country 2. {scale_note}')
    else:
        st.markdown(f'- {cur1} changed **{100*(nominal.iloc[-1]/nominal.iloc[0]-1):+,.1f}%** nominally over the selection.\n- CPI coverage is insufficient to calculate a real comparison for these dates.')
    missing=int(sample.rer.isna().sum())
    if missing: st.caption(f'{missing} observations lack joint CPI/FX inputs. Missing RERs remain blank and are excluded from the mean; nominal observations are retained when available.')
    if frequency=='Annual':
        partial=display.loc[display.valid_months<12]
        if not partial.empty: st.caption('Incomplete annual RER averages: '+', '.join(f'{i.year} ({int(r.valid_months)}/12 months)' for i,r in partial.iterrows())+'. Each point averages available monthly RERs; the underlying monthly index still averages 100.')
    st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
    st.subheader('02  /  A reference for valuation')
    settings=st.container(key='reference_controls').columns([2.1,1,2.4],gap='medium')
    methods=['Automatic · screened history','Custom period']
    reference=settings[0].selectbox('Reference approach',methods,index=0,key='reference_v3_'+c1+c2)
    method=settings[1].selectbox('Reference statistic',['Median','Mean'],key='statistic')
    profile=settings[2].selectbox('Your perspective',['General analysis','Trader / investor','Company · payments / debt in C1','Company · receipts / assets in C1'],key='profile')
    horizon='2+ years'
    if profile=='Trader / investor': horizon=st.segmented_control('Decision horizon',['Under 2 years','2+ years'],default='2+ years',key='horizon') or '2+ years'
    growth,gnote=growth_data(live)
    b=benchmark(monthly,c1,c2,growth,method,as_of=sample.index.max())
    if reference=='Custom period':
        history=monthly.dropna(subset=['nominal'])
        bounds=(history.index.min(),history.index.max())
        initial_start=pd.Timestamp(b['start']) if b.get('available') else bounds[0]
        initial_end=pd.Timestamp(b['end']) if b.get('available') else initial_start
        ref_key=f'{c1}_{c2}'
        def set_reference_month(when):
            for side in ['start','end']:
                st.session_state[f'ref_{side}_year_{ref_key}']=when.year
                st.session_state[f'ref_{side}_month_{ref_key}']=when.month
        for side,when in [('start',initial_start),('end',initial_end)]:
            st.session_state.setdefault(f'ref_{side}_year_{ref_key}',when.year)
            st.session_state.setdefault(f'ref_{side}_month_{ref_key}',when.month)
        fields=st.container(key='benchmark_dates').columns(4,gap='small')
        endpoints=[]
        for i,side,title in [(0,'start','Reference start'),(2,'end','Reference end')]:
            year=fields[i].selectbox(title+' year',list(range(bounds[0].year,bounds[1].year+1)),key=f'ref_{side}_year_{ref_key}',help='Type a year to jump directly to it.')
            month=fields[i+1].selectbox(title+' month',list(range(1,13)),format_func=lambda m:MONTH_NAMES[m-1],key=f'ref_{side}_month_{ref_key}')
            endpoints.append(pd.Timestamp(year,month,1))
        if {c1,c2}=={'G163','USA'}:
            st.button('Use January 1999',on_click=set_reference_month,args=(pd.Timestamp('1999-01-01'),),key='jan1999_'+ref_key)
        st.caption('Both endpoints are included. Choose the same month twice for a single-month reference. Custom periods are your assumption of fair value; they do not have to pass the automatic stability screen.')
        b=custom_benchmark(monthly,*endpoints,method=method,as_of=sample.index.max(),daily=frequency=='Daily')
        if {c1,c2}=={'G163','USA'} and not b.get('available') and not use_fred and endpoints[0]<pd.Timestamp('1999-12-01'):
            b['reason']+=' For January 1999, select FRED above; the bundled IMF euro HICP starts in December 1999.'
    with st.expander('Why this benchmark? / Sensitivity',expanded=False):
        st.write(b.get('reason',''))
        tolerance=st.slider('Neutral range around reference (%)',min_value=2,max_value=15,value=5,step=1,help='An explicit judgment threshold, not a statistical confidence interval. Classification also incorporates the reference period’s interquartile range.')
        show_band=st.checkbox('Shade reference-period interquartile range',value=True)
        st.caption('Automatic rule: latest qualifying five-year calendar window, ending at least two calendar years before today and no later than the displayed endpoint. Every month requires joint data and prior-year CPI. Inflation: mean 0–6%, at least 90% of months within −1–8%, absolute maximum 12%, standard deviation ≤3 pp. Growth: mean absolute bilateral gap ≤3 pp, maximum yearly gap ≤6 pp, each yearly growth −3–10% and nonnegative period averages. RER: coefficient of variation ≤15%, absolute log trend ≤3% per year. These are disclosed app-design heuristics, not estimates from the paper. Temporary inflation misses or mild recessions no longer automatically disqualify an entire period.')
        st.caption('“Two calendar years” means candidate end-year ≤ current year minus 2 (for example, 2024 in 2026). WEO past-year values may still be estimates and revisions; this rule excludes forecast years, not all estimation uncertainty.')
        if b.get('available'):
            st.write(f"**{b['label']}** · {b['method']} raw RER = {number(b['value'],display_decimals(monthly.rer))}")
            if b.get('diagnostics'):
                d=b['diagnostics']
                st.caption(f"{d['months']:,} complete months · mean growth gap {number(d['growth_gap'])} pp · RER dispersion {number(100*d['dispersion'])}% · annual RER log trend {number(100*d['trend'])}%")
            if b.get('alternatives'):
                alternatives=pd.DataFrame(b['alternatives']).rename(columns={'value':'Raw RER reference'})
                st.dataframe(alternatives.style.format({'Raw RER reference':lambda v:number(v,display_decimals(monthly.rer))}),hide_index=True,width='stretch')
                st.caption('All passing windows are shown for sensitivity; overlapping windows are not independent estimates. The latest passing window is chosen without considering today’s exchange rate.')
        st.caption(f'Growth: IMF WEO, with EURO mapped to the euro-area aggregate (not EUQ/Europe). {gnote}. Known regime starts restrict candidate history; other pairs rely on the empirical checks only. The chart start date does not select the benchmark; an earlier endpoint excludes future candidates. Revised data are not real-time backtests.')
        st.markdown(f'[Paper]({PAPER}) · [Chile regime](https://si2.bcentral.cl/public/publicaciones_digitales/annualreport2018/en/4-politica-monetaria/) · [Peru regime](https://www.bcrp.gob.pe/eng-docs/Publications/Annual-Reports/2002/Annual-Report-2002-01.pdf) · [IMF group definitions](https://www.imf.org/external/datamapper/api/v1/groups)')
    if b.get('available') and (sample.index.max()<pd.Timestamp(b['end'])):
        b=dict(available=False,reason='The selected endpoint predates completion of the fixed benchmark period. No benchmark is displayed to avoid look-ahead.')
    enriched=add_benchmark(sample,b,mean)
    display2=annual_view(enriched) if frequency=='Annual' else enriched
    left,right=st.container(key='benchmark_panel').columns([2.5,1],gap='large')
    with left:
        if b.get('available'): st.caption(b['label']+' · fixed RER anchor; nominal PPP path adjusts with relative CPI')
        else: st.caption('Evolution shown without a benchmark · evidence is insufficient')
        st.plotly_chart(chart(display2,mode,quote,mean,b,show_band,raw=raw_scale,frequency=frequency),width='stretch',config={'displaylogo':False,'scrollZoom':False},key='reference_chart')
    with right:
        v=valuation(enriched,b,tolerance)
        if cur1==cur2: v=dict(status='No bilateral FX signal',detail='Shared currency: relative prices explain RER movements.',actionable=False)
        st.markdown(f'### {cur1} · {v["status"]}')
        st.write(v['detail'])
        if b.get('alternatives') and len(b['alternatives'])>1 and not real.empty:
            gaps=[100*(real.rer.iloc[-1]/x['value']-1) for x in b['alternatives']]
            st.caption(f'Across {len(gaps)} qualifying windows, the current gap ranges from {min(gaps):+,.1f}% to {max(gaps):+,.1f}%. See sensitivity before interpreting fair value.')
        if v['actionable']:
            st.metric('Reference-implied nominal rate',f'{number(v["implied"],nominal_decimals)} {quote}')
            st.caption(f'As of {v["when"].strftime(datefmt)} · movement from observed rate to reference: {v["adjustment"]:+,.1f}%. This differs from the valuation gap because the denominator differs.')
        st.markdown('**For your perspective**')
        st.write(recommendation(v,profile,cur1,cur2,horizon,cur1==cur2))
        st.caption('Automated, rules-based interpretation. AI-assisted design; no live AI model, probability estimate or guaranteed return. Historical monetary regimes may have changed; no current-regime verification is implied.')
    if b.get('available'):
        st.caption('Purple shading is historical dispersion, not a confidence interval. Both charts use the same RER scale. The fixed benchmark need not equal 100 on the index scale. Under fixed exchange rates, adjustment can occur through inflation.')
    if {c1,c2}=={'G163','USA'}:
        with st.expander('How sources and benchmarks affect the result'):
            st.markdown('**Benchmark: what counts as fair value?** January 1999 assumes the euro was fairly priced at launch. Automatic uses the median or mean of a screened five-year period. These are different economic assumptions and can produce different valuation signals even with identical data. A historical reference is not proof of equilibrium.')
            st.markdown('**Source: what is measured?** IMF supplies monthly CPI/HICP and period-average FX. FRED distributes BLS US CPI, Eurostat HICP for a 19-country euro-area aggregate, and Federal Reserve USD/EUR rates observed at noon in New York, which this app averages by month. Differences in price-index definitions or geographic coverage, FX observation times, revisions and missing months can change the measured trend and valuation gap. Neither source is automatically the correct fair-value estimate.')
            st.markdown('**Scale: how is it displayed?** Raw RER levels depend on the CPI index bases. Rescaling a consistent series does not change its percentage gap from its benchmark: both are rescaled together. An index above 100 means above the selected-period average, not necessarily overvalued. Switching sources preserves your scale and date choices where coverage permits; compare the same month and reference period.')
            st.caption('Euro trading starts in January 1999. In the bundled snapshot, IMF euro HICP starts December 1999, while FRED supports a January 1999 reference. CPI is monthly: Daily is a lagged-CPI proxy and should not be compared directly with a same-month monthly calculation. October 2025 US CPI is missing in the current FRED series and is left blank.')
            if b.get('available'):
                im,im_notes=monthly_pair(c1,c2,live)
                fm,_,fm_notes=fred_pair(c1,c2,live)
                common_month,comparison=compare_sources(im,fm,b['start'],b['end'],method,sample.index.max())
                if common_month is not None:
                    st.markdown(f'**Same-month source comparison · {common_month:%B %Y}**')
                    st.caption(f"Both rows evaluate {cur1} against {cur2}, using monthly data, the same {b['start'][:7]} to {b['end'][:7]} reference period and {method.lower()}. No independent automatic re-selection is made for the other source.")
                    st.dataframe(comparison.rename(columns={'Observed':f'Observed · {quote}','Implied':f'Reference-implied · {quote}','Gap':'Gap (%)'}).style.format({f'Observed · {quote}':lambda v:number(v,nominal_decimals),f'Reference-implied · {quote}':lambda v:number(v,nominal_decimals),'Gap (%)':lambda v:number(v,sign=True)},na_rep='Unavailable'),hide_index=True,width='stretch')
                    if comparison.Gap.isna().any(): st.caption('Unavailable means that source cannot supply all the selected reference months. References are never copied between differently based CPI series.')
                    st.caption('Comparison source status: '+' | '.join(im_notes+fm_notes))
            st.markdown('To explore a launch-date scenario: select **FRED → Custom period → Use January 1999**. Use **All** to view the full available history.')
            st.markdown('[IMF data](https://data.imf.org/en/Resource-Pages/IMF-API) · [BLS US CPI via FRED](https://fred.stlouisfed.org/series/CPIAUCNS) · [Eurostat EA19 HICP via FRED](https://fred.stlouisfed.org/series/CP0000EZ19M086NEST) · [Federal Reserve USD/EUR via FRED](https://fred.stlouisfed.org/series/DEXUSEU)')
    st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
    export=enriched.reset_index()
    export.insert(1,'country1',c1);export.insert(2,'country2',c2)
    export['quote_direction']=quote
    export['frequency']='D_PROXY' if frequency=='Daily' else 'M'
    export['data_basis']='FRED: BLS CPI, Eurostat EA19 HICP, Federal Reserve FX' if use_fred else 'IMF CPI / IMF monthly FX or BIS/ECB daily FX'
    export['chart_rer_scale']='Raw RER' if raw_scale else 'Selected-sample index = 100'
    export['benchmark_start']=b.get('start','');export['benchmark_end']=b.get('end','')
    export['benchmark_method']=b.get('method','Unavailable')
    export['benchmark_approach']=reference
    export['source_notes']=' | '.join(notes)
    export['valid_rer']=export.rer.notna()
    downloads=st.columns([1.5,1.5,3])
    downloads[0].download_button('↓ Download observations CSV',export.to_csv(index=False).encode('utf-8'),file_name=f'parity_{c1}_{c2}_{frequency.lower()}.csv',mime='text/csv',width='stretch')
    if frequency=='Annual':
        downloads[1].download_button('↓ Annual chart CSV',display2.reset_index().to_csv(index=False).encode('utf-8'),file_name=f'parity_{c1}_{c2}_annual_chart.csv',mime='text/csv',width='stretch')
    downloads[2].caption('The observations CSV includes raw CPI and FX inputs, nominal S, RER, sample mean, real index, missing-data flags, benchmark calculations and source dates.')
    with st.expander('Inspect observations & source status'):
        formats={}
        for col in export.select_dtypes(include=[np.number]).columns:
            digits=display_decimals(export[col],'nominal' if col in ['nominal','implied_nominal','fx1_per_usd','fx2_per_usd'] else 'rer')
            if col.endswith('_pct'): digits=1
            formats[col]=lambda value,d=digits:number(value,d)
        st.dataframe(export.style.format(formats,na_rep='—'),width='stretch',hide_index=True)
        st.caption('Rounded for readability. CSV downloads retain full precision and ungrouped numeric values for analysis.')
        for note in notes: st.caption(note)
        st.caption('Data may be revised or delayed. Official rates can differ from executable market rates, particularly under controls or multiple-rate regimes. BIS may express historical redenominations in current currency units; those historical levels are not executable quotes. National euro-member comparisons begin at euro adoption. Daily source coverage is narrower than monthly IMF coverage.')


if __name__=='__main__':
    main()
