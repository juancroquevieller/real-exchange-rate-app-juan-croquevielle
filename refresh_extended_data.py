"""Refresh BIS daily history and the FRED series used by the first homework."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import io, json
import pandas as pd
import requests

DATA=Path(__file__).parent/'data'
BIS_AREAS='GB+CL+PE+XM+AU+CA+CH+CN+CO+BR+MX+NZ+JP+SE+NO+DK+IN+ID+KR+ZA+HK+SG+TH+MY+PH+IL+TR+CZ+HU+PL+RO+IS+AR'
BIS_URL=f'https://stats.bis.org/api/v2/data/dataflow/BIS/WS_XRU/1.0/D.{BIS_AREAS}..?format=csv&startPeriod=1976-01-01'
FRED_CPI_URL='https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS,CP0000EZ19M086NEST'
FRED_FX_URL='https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU'
SOURCES={'bis_daily.csv.gz':BIS_URL,'fred_cpi.csv.gz':FRED_CPI_URL,'fred_eurusd.csv.gz':FRED_FX_URL}

def parse_bis(raw):
    frame=raw.loc[(raw.FREQ=='D') & (raw.COLLECTION=='A')].copy()
    frame['value']=pd.to_numeric(frame.OBS_VALUE,errors='coerce')*10.0**pd.to_numeric(frame.UNIT_MULT,errors='coerce').fillna(0)
    frame['date']=pd.to_datetime(frame.TIME_PERIOD)
    frame=frame.rename(columns={'CURRENCY':'currency','REF_AREA':'area'})
    # National euro predecessors are not a tradable euro history before launch.
    frame=frame.loc[(frame.currency!='EUR') | ((frame.area=='XM') & (frame.date>='1999-01-01'))]
    if frame.duplicated(['date','currency']).any():
        raise ValueError('Ambiguous BIS daily observations; inspect source dimensions.')
    return frame[['date','currency','area','value','OBS_STATUS','OBS_PRE_BREAK']].sort_values(['currency','date'])

def download(item):
    name,url=item
    response=requests.get(url,timeout=(15,180)); response.raise_for_status()
    frame=pd.read_csv(io.StringIO(response.text),low_memory=False)
    if name.startswith('bis'): frame=parse_bis(frame)
    else:
        expected={'observation_date','DEXUSEU'} if 'eurusd' in name else {'observation_date','CPIAUCNS','CP0000EZ19M086NEST'}
        if not expected.issubset(frame.columns): raise ValueError('Unexpected FRED schema')
    frame.to_csv(DATA/name,index=False,compression={'method':'gzip','mtime':0})
    print(name,len(frame),'observations saved',flush=True)
    return name,dict(url=url,retrieved_at=datetime.now(timezone.utc).isoformat(),bytes=(DATA/name).stat().st_size)

if __name__=='__main__':
    manifest_path=DATA/'sources.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name,note in pool.map(download,SOURCES.items()): manifest[name]=note
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
