"""Refresh attributable public-source snapshots. Run manually; never invent fallback data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import gzip, json, requests

ROOT = Path(__file__).parent / 'data'
ROOT.mkdir(exist_ok=True)
SOURCES = {
 'euro_cpi.csv.gz': ('https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/CPI/~/G163.HICP._T.IX.M','text/csv'),
 'cpi.csv.gz': ('https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/CPI/~/*.CPI._T.IX.M','text/csv'),
 'fx.csv.gz': ('https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/ER/~/*.XDC_USD.PA_RT.M','text/csv'),
 'countries.json': ('https://www.imf.org/external/datamapper/api/v1/countries','application/json'),
 'growth.json': ('https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH','application/json'),
 'ecb.csv.gz': ('https://data-api.ecb.europa.eu/service/data/EXR/D..EUR.SP00.A?startPeriod=1999-01-01&format=csvdata','text/csv'),
}

def download(item):
 name, (url, accept) = item
 try:
  r=requests.get(url,headers={'Accept':accept},timeout=(15,120))
  r.raise_for_status()
  if name.endswith('.gz'):
   if 'OBS_VALUE' not in r.text[:5000]: raise ValueError('Not observation CSV')
   raw=gzip.compress(r.content,mtime=0)
  else:
   r.json()
   raw=r.content
  (ROOT/name).write_bytes(raw)
  print(name, len(raw), 'bytes saved',flush=True)
  return name, {'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),'bytes':len(raw)}
 except Exception as e:
  print(name, 'FAILED',str(e),flush=True)
  return name, None

if __name__=='__main__':
 manifest_path=ROOT/'sources.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
 with ThreadPoolExecutor(max_workers=5) as pool:
  for name, result in pool.map(download,SOURCES.items()):
   if result: manifest[name]=result
 manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
