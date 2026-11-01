#!/usr/bin/env python3
"""Fetch exact upstream commits and apply reviewed patches. No hardware."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def git(path,*args):
 return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def main():
 argparse.ArgumentParser(description=__doc__).parse_args()
 for item in json.loads((ROOT/'sources.lock.json').read_text())['repositories']:
  dest=ROOT/item['path'];patch=ROOT/item['patch']
  if not (dest/'.git').exists():
   if dest.exists() and any(dest.iterdir()):raise RuntimeError(f'Nonempty untracked source directory: {dest}')
   dest.mkdir(parents=True,exist_ok=True)
   git(dest,'init','-q');git(dest,'remote','add','origin',item['url'])
  head=subprocess.run(['git','-C',str(dest),'rev-parse','--verify','HEAD'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  if head.returncode:
   git(dest,'fetch','--depth','1','origin',item['commit']);git(dest,'checkout','--detach','FETCH_HEAD')
  if git(dest,'rev-parse','HEAD')!=item['commit']:raise RuntimeError(f'Wrong checkout: {dest}; preserve edits before replacing it')
  actual=git(dest,'diff','--binary','--ignore-submodules=all','HEAD');expected=patch.read_text().strip()
  if actual==expected:pass
  elif not actual and expected:
   subprocess.run(['git','-C',str(dest),'apply','--check',str(patch)],check=True)
   subprocess.run(['git','-C',str(dest),'apply',str(patch)],check=True)
  else:raise RuntimeError(f'Unrecorded source edits: {dest}')
  if git(dest,'diff','--binary','--ignore-submodules=all','HEAD')!=expected:raise RuntimeError('Patch verification failed')
  print('PINNED',item['path'],item['commit'],flush=True)
if __name__=='__main__':main()
