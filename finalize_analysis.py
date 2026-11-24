"""Additional summaries and independent AC checks for the frozen confirmation."""
from pathlib import Path
import json
import analyze_frozen_results as base
import numpy as np
import pandas as pd
from study2 import coupled30,CONT
from validate30 import monolithic30

HERE=Path(__file__).resolve().parent
cases=pd.read_csv(HERE/'confirmation_cases.csv')
pred=pd.read_csv(HERE/'confirmation_predictions.csv')
raw={r['pair']:r for r in json.loads((HERE/'confirmation_inputs.json').read_text())}
miss=pred[(pred.bypass==1)&(pred.unsafe==1)]
ids=sorted(miss.pair.unique());rows=[];details=[]
for pair in ids:
    r=raw[int(pair)];out=CONT[r['event']];a=coupled30(r,out);b,ok=monolithic30(r,out)
    assert ok and a['status']=='converged'
    cs=base.extract_constraints(r,b);cs.insert(0,'pair',int(pair));details.append(cs)
    worst=cs.loc[cs.normalized_violation.idxmax()];error=abs(worst.normalized_violation-a['violation'])
    assert error<1e-6,(pair,error)
    rows.append(dict(pair=int(pair),event=r['event'],outages_zero_based=';'.join(map(str,out)),order=len(out),reference_margin=a['violation'],monolithic_margin=worst.normalized_violation,margin_error=error,kind=worst.kind,component=worst.component,actual=worst.actual,limit=worst.limit,unit=worst.unit,excess=worst.excess))
severity=pd.DataFrame(rows);severity.to_csv(HERE/'confirmation_false_safe_severity.csv',index=False)
pd.concat(details,ignore_index=True).to_csv(HERE/'confirmation_false_safe_constraints.csv',index=False)
miss.merge(severity,on='pair').to_csv(HERE/'confirmation_false_safe_by_method.csv',index=False)
timing=pd.read_csv(HERE/'feedback_timing.csv');tr=timing.groupby('method').agg(median_seconds=('seconds','median'),min_seconds=('seconds','min'),max_seconds=('seconds','max'),n=('n','first'),bypassed=('bypassed','first'),guard_calls=('guard_calls','first')).reset_index()
reference=float(tr.loc[tr.method=='Full_AC','median_seconds'].iloc[0]);tr['relative_to_full']=tr.median_seconds/reference;tr['saving_percent']=100*(1-tr.relative_to_full);tr.to_csv(HERE/'timing_summary.csv',index=False)
pivot=pred.pivot(index='pair',columns='method',values='bypass');rng=np.random.default_rng(2026110721);comparisons=[]
for name in ['One_pass','Physics_HGB','Logistic','Event_HGB']:
    d=(pivot.Feedback_NP-pivot[name]).to_numpy();bs=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(5000)])
    comparisons.append(dict(method='Feedback_NP',comparator=name,bypass_delta=float(d.mean()),ci_low=float(np.quantile(bs,.025)),ci_high=float(np.quantile(bs,.975))))
pd.DataFrame(comparisons).to_csv(HERE/'feedback_paired_differences.csv',index=False)
summary=dict(fresh_confirmation_n=len(cases),unsafe=int(cases.unsafe.sum()),support=int(cases.supported.sum()),status=cases.status.value_counts().to_dict(),independent_rechecks=len(ids),maximum_monolithic_margin_error=float(severity.margin_error.max()),timing_scope='First 400 new independent secure-base samples, three repetitions, randomized method order, single thread, initialized cached matrices; model fit and initial cache construction excluded',risk_scope='Single uniformly sampled connected event per independently sampled secure-base state; unsafe-conditioned population risk, not each outage order or whole scan')
(HERE/'revision_summary.json').write_text(json.dumps(summary,indent=2))
print(tr.to_string(index=False));print(severity.to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))
