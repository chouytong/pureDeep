"""Authorized code/report publication milestones; no experiment controller."""
import argparse, subprocess, importlib.util, json
from pathlib import Path
from common import HERE, ROOT, require, sha
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--message',required=True);args=p.parse_args()
script=ROOT/'tools/git_publish/publish_to_github.py'
result=subprocess.run(['python3',str(script),'--apply','--message',args.message]);require(result.returncode==0,'Git publisher failed')
spec=importlib.util.spec_from_file_location('pub',script);pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
head=pub.git('rev-parse','HEAD').stdout.strip();tag=f'patch-str-{args.stage}-20261008'
for command in [('tag','-a',tag,'-m',args.message),('push','origin',tag)]:
 r=pub.git(*command);require(r.returncode==0,r.stderr)
r=pub.git('ls-remote','origin','refs/heads/main','refs/tags/'+tag+'^{}');require(r.returncode==0 and len(r.stdout.splitlines())==2 and all(x.split()[0]==head for x in r.stdout.splitlines()),'Remote/tag mismatch')
require(not pub.git('status','--porcelain').stdout.strip(),'Publishing mirror not clean')
receipt=dict(status='PASS',commit=head,tag=tag,stage=args.stage,main_tag_verified=True)
Path(f'/home/zyt/pureDeep-patch-str-{args.stage}-receipt-20261008.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
