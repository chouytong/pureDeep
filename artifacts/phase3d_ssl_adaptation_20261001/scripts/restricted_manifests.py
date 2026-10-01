"""Create fold-local manifest views for subset-only development, without source edits."""
from pathlib import Path
import pandas as pd

def restrict(config,allowed,destination):
 allowed=set(map(str,allowed));destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 assert allowed
 paths={}
 for activity,source in config['data']['activity_manifests'].items():
  source=Path(source);f=pd.read_csv(source,dtype={'subject_id':str})
  out=f[f.subject_id.isin(allowed)].copy()
  assert set(out.subject_id)==allowed and out.subject_id.is_unique
  path=destination/f'{activity}.csv';out.to_csv(path,index=False);paths[activity]=str(path)
 config['data']['activity_manifests']=paths
 return paths
