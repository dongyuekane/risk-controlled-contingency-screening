"""Validate final misses independently and summarize final frozen evidence."""
from pathlib import Path
import json,hashlib,platform,importlib.metadata
import analyze_frozen_results as base
import numpy as np,pandas as pd
from study2 import coupled30,CONT
from validate30 import monolithic30

HERE=Path(__file__).resolve().parent;OUT=HERE/'fresh_release'
def main():
    pred=pd.read_csv(OUT/'confirmation_predictions.csv');miss=pred[(pred.unsafe==1)&(pred.bypass==1)]
    raws={r['pair']:r for r in json.loads((OUT/'confirmation/confirmation_inputs.json').read_text())}
    rows=[];detail=[]
    for pair in sorted(miss.pair.unique()):
        r=raws[int(pair)];a=coupled30(r,CONT[r['event']]);b,ok=monolithic30(r,CONT[r['event']])
        assert ok and a['status']=='converged'
        cs=base.extract_constraints(r,b);cs.insert(0,'pair',int(pair));detail.append(cs)
        m=cs.loc[cs.normalized_violation.idxmax()];err=abs(m.normalized_violation-a['violation']);assert err<1e-6
        rows.append(dict(pair=int(pair),event=r['event'],outages_one_based='/'.join(str(x+1) for x in CONT[r['event']]),reference_margin=a['violation'],monolithic_margin=m.normalized_violation,margin_error=err,kind=m.kind,component=m.component,actual=m.actual,limit=m.limit,unit=m.unit,excess=m.excess))
    s=pd.DataFrame(rows);s.to_csv(OUT/'independent_false_safe_checks.csv',index=False)
    if detail:pd.concat(detail,ignore_index=True).to_csv(OUT/'independent_constraint_details.csv',index=False)
    miss.merge(s,on='pair').to_csv(OUT/'false_safe_by_method.csv',index=False)
    pv=pred.pivot(index='pair',columns='method',values='bypass');rng=np.random.default_rng(2026125707);bs=[]
    for name in ['One_pass','Event_HGB','Logistic','Physics_HGB','Feedback_NP']:
        diff=(pv.Selective_feedback-pv[name]).to_numpy();samples=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(5000)])
        bs.append(dict(method='Selective_feedback',comparator=name,bypass_delta=float(diff.mean()),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975))))
    pd.DataFrame(bs).to_csv(OUT/'paired_bypass_differences.csv',index=False)
    summary=json.loads((OUT/'summary.json').read_text());summary['independent_false_safe_checks']=len(rows);summary['maximum_monolithic_margin_error']=float(s.margin_error.max()) if rows else 0.
    if (OUT/'final_timing_summary.csv').exists():
        t=pd.read_csv(OUT/'final_timing_summary.csv').set_index('method');sel=float(t.loc['Selective_feedback','median_seconds']);summary['time_saving_vs_full']=float(1-sel/t.loc['Full_AC','median_seconds']);summary['time_saving_vs_electrical_hgb']=float(1-sel/t.loc['Physics_HGB','median_seconds']);summary['time_saving_vs_feedback']=float(1-sel/t.loc['Feedback_NP','median_seconds'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    env={'python':platform.python_version()}
    for pkg in ['numpy','pandas','scipy','scikit-learn','pypower','threadpoolctl','python-docx']:
        try:env[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:pass
    (OUT/'environment.json').write_text(json.dumps(env,indent=2))
    manifest=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name!='file_manifest.json' and '__pycache__' not in p.parts:
            manifest.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (HERE/'file_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary)

if __name__=='__main__':main()
