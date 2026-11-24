"""Read frozen scores; independently resolve actual AC constraint severity.

No threshold is fitted to the fresh test labels.  Original files are read only.
Run with the project Python environment (numpy/pandas/scipy/pypower installed).
"""
from pathlib import Path
import sys, json, hashlib, platform, time, os

SOURCE = Path(os.environ.get('POWER_STUDIES_WORK',str(Path(__file__).resolve().parent/'original_source')))
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE/'package/SCI_Three_Studies/code'))
sys.path.insert(0, str(SOURCE/'vendor'))
import numpy as np
import pandas as pd
from scipy.stats import beta
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
from study2 import coupled30, CONT
from grid import case, feeder
from validate30 import monolithic30


def extract_constraints(raw, pf):
    """Physical units plus the original label-defining normalized violations."""
    mainbus=case('case30')['bus']; gen=case('case30')['gen']
    fc,normal,_=feeder(raw['name']); n=len(fc['bus'])
    values=[]
    def add(kind, component, actual, limit, direction, unit, normalization):
        excess=(actual-limit) if direction=='upper' else (limit-actual)
        values.append(dict(kind=kind,component=component,actual=float(actual),limit=float(limit),direction=direction,unit=unit,excess=float(excess),normalized_violation=float(excess/normalization)))
    for i,v in enumerate(pf['bus'][:30,7]):
        add('transmission_voltage_low',f'bus_{i+1}',v,mainbus[i,12],'lower','pu',.1)
        add('transmission_voltage_high',f'bus_{i+1}',v,mainbus[i,11],'upper','pu',.1)
    for i,v in enumerate(pf['bus'][30:,7]):
        add('distribution_voltage_low',f'bus_{i+31}',v,.9,'lower','pu',.1)
        add('distribution_voltage_high',f'bus_{i+31}',v,1.1,'upper','pu',.1)
    for i,br in enumerate(pf['branch'][:41]):
        apparent=max(np.hypot(br[13],br[14]),np.hypot(br[15],br[16]))
        limit=case('case30')['branch'][i,5]
        add('transmission_thermal',f'branch_{i+1}',apparent,limit,'upper','MVA',limit)
    for i,g in enumerate(gen):
        p,q=pf['gen'][i,[1,2]]
        add('generator_P_high',f'generator_{i+1}',p,g[8],'upper','MW',g[8])
        add('generator_P_low',f'generator_{i+1}',p,g[9],'lower','MW',g[8])
        add('generator_Q_high',f'generator_{i+1}',q,g[3],'upper','MVAr',max(1,abs(g[3])))
        add('generator_Q_low',f'generator_{i+1}',q,g[4],'lower','MVAr',max(1,abs(g[4])))
    for group,offset in enumerate([41,41+n]):
        tb=pf['branch'][offset]
        add('transformer_thermal',f'group_{group+1}',np.hypot(tb[13],tb[14]),20.,'upper','MVA',20.)
        for i,br in enumerate(pf['branch'][offset+1:offset+n]):
            v=pf['bus'][int(br[0])-1,7]*fc['bus'][0,9]
            amps=np.hypot(br[13],br[14])*1000/(np.sqrt(3)*v*4)
            limit=normal[i,4]*raw['amp']
            add('distribution_current',f'group_{group+1}_branch_{i+1}',amps,limit,'upper','A',limit)
    return pd.DataFrame(values)


def main():
    audit=SOURCE/'study2_audit'
    cases=pd.read_csv(audit/'new_cases.csv')
    predictions=pd.read_csv(audit/'matched_predictions.csv')
    predictions=predictions[predictions.split.str.startswith('new_')].merge(cases[['pair','order','status','network']],on='pair',validate='many_to_one')
    rows=[]
    for (split,method,order),g in predictions.groupby(['split','method','order']):
        nu=int(g.unsafe.sum());miss=int((g.unsafe*g.bypass).sum())
        rows.append(dict(split=split,method=method,order=int(order),n=len(g),unsafe=nu,bypassed=int(g.bypass.sum()),false_safe=miss,false_safe_rate=miss/nu if nu else np.nan,upper95=float(beta.ppf(.95,miss+1,nu-miss)) if nu>miss else 1.))
    grouped=pd.DataFrame(rows);grouped.to_csv(HERE/'order_stratified_results.csv',index=False)
    raws={r['pair']:r for r in json.loads((audit/'new_inputs.json').read_text())}
    # Resolve the union of false-safe nominal samples from every matched baseline.
    # This permits severity comparisons without counting approximate PF quantities.
    misspred=predictions[(predictions.split=='new_nominal')&(predictions.unsafe==1)&(predictions.bypass==1)]
    ids=sorted(misspred.pair.unique()); allc=[]; summaries=[]
    for pair in ids:
        raw=raws[int(pair)];out=CONT[raw['event']]
        t=time.perf_counter();a=coupled30(raw,out);elapsed=time.perf_counter()-t
        pf,ok=monolithic30(raw,out)
        if not ok or a['status']!='converged':
            summaries.append(dict(pair=int(pair),status=a['status'],reference_margin=a['violation'],monolithic_converged=bool(ok),seconds=elapsed));continue
        cs=extract_constraints(raw,pf);cs.insert(0,'pair',int(pair));allc.append(cs)
        strongest=cs.loc[cs.normalized_violation.idxmax()]
        error=abs(float(strongest.normalized_violation)-a['violation'])
        assert error<1e-6,(pair,error)
        summaries.append(dict(pair=int(pair),event=int(raw['event']),order=len(out),outages_zero_based=';'.join(map(str,out)),status=a['status'],reference_margin=a['violation'],monolithic_margin=float(strongest.normalized_violation),margin_error=error,monolithic_converged=True,violated_constraints=int((cs.normalized_violation>1e-7).sum()),kind=strongest.kind,component=strongest.component,actual=strongest.actual,limit=strongest.limit,unit=strongest.unit,excess=strongest.excess,seconds=elapsed))
    summary=pd.DataFrame(summaries);summary.to_csv(HERE/'false_safe_ac_severity.csv',index=False)
    pd.concat(allc,ignore_index=True).to_csv(HERE/'false_safe_constraint_details.csv',index=False)
    misspred.merge(summary,on='pair',suffixes=('','_audit')).to_csv(HERE/'false_safe_by_method.csv',index=False)
    counts=[]
    for method,g in misspred.groupby('method'):
        sg=g.merge(summary,on='pair')
        counts.append(dict(method=method,false_safe=len(sg),maximum_normalized_violation=float(sg.monolithic_margin.max()),median_normalized_violation=float(sg.monolithic_margin.median()),unresolved=int((~sg.monolithic_converged).sum())))
    pd.DataFrame(counts).to_csv(HERE/'severity_by_method.csv',index=False)
    paths=[audit/'new_cases.csv',audit/'new_inputs.json',audit/'matched_predictions.csv',audit/'matched_thresholds.csv',SOURCE/'package/SCI_Three_Studies/code/study2.py',SOURCE/'package/SCI_Three_Studies/code/grid.py',SOURCE/'package/SCI_Three_Studies/code/validate30.py']
    manifest=dict(python=platform.python_version(),source=str(SOURCE),fresh_nominal=2000,new_ac_recomputations=len(ids),threshold_scope='Frozen alpha=0.01 predictions; no recalibration or model refitting',sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (HERE/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(grouped.query('split=="new_nominal" and method=="Physics_HGB"').to_string(index=False))
    print(summary[summary.pair.isin([710890,711066,711531,711865])].to_string(index=False))
    print('Resolved union of matched-method false-safe cases:',len(ids))


if __name__=='__main__': main()
