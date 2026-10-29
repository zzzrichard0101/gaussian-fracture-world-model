"""Prepare existing PUBLIC report metadata on a fresh checkout, without overwrite.

task_aware_pilot hashes these two reports to check they stay unchanged; it does
not consume their private runs or numerical labels for control or data creation.
"""
from pathlib import Path
import shutil


def main():
    project=Path(__file__).resolve().parents[1];copied=[]
    for stage in ('observation-budget','guided-quadrature'):
        target=project/'artifacts'/stage/'report.json'
        if target.exists():continue
        source=project/'docs/results'/(stage+'.json')
        if not source.is_file():raise FileNotFoundError(f'Public reference missing: {source}')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);copied.append(stage)
    print(f'Public report metadata prepared: {copied}; existing report files retained.')


if __name__=='__main__':main()
