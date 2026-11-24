"""Run ONLY in a reserved serial benchmark window; no concurrent simulations."""
import json,time,pickle,hashlib
import cascade_experiment as ce
import feedback_guard_experiment as fb
import numpy as np,pandas as pd
from study2 import coupled30,features,CONT

OUT=ce.HERE/'fresh_release'
def main():
    protocol=json.loads((OUT/'protocol.json').read_text());blob=(OUT/'frozen_models.pkl').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==protocol['models_sha256']
    models=pickle.loads(blob);support=json.loads((OUT/'frozen_support.json').read_text());lo=np.array(support['lo']);hi=np.array(support['hi'])
    pars=json.loads((OUT/'frozen_router.json').read_text())
    ts=pd.read_csv(OUT/'thresholds.csv').set_index('method').threshold.to_dict();threshold=ts['Selective_feedback']
    raws=json.loads((OUT/'confirmation/confirmation_inputs.json').read_text())[:protocol['timing_n']]
    expected=pd.read_csv(OUT/'cascade_predictions.csv').set_index('pair').bypass
    methods=['Full_AC','One_pass','Feedback_NP','Physics_HGB','Logistic','Event_HGB','Selective_feedback']
    def pipeline(name):
        t0=time.perf_counter();keep=np.ones(len(raws),bool);ids=[];fs=[];feedback_calls=0;direct_bypass=0
        if name!='Full_AC':
            for i,r in enumerate(raws):
                if not fb.supported(r,lo,hi):continue
                ids.append(i)
                if name in ['Event_HGB','Selective_feedback']:
                    fs.append(dict(event=r['event'],P=sum(r['p']),Q=sum(r['q']),scale=r['scale'],tap=r['tap'],amp=r['amp'],n_out=len(CONT[r['event']])));continue
                lin=coupled30(r,CONT[r['event']],True)
                if name in ['One_pass','Feedback_NP']:
                    score=lin['violation']
                    if name=='Feedback_NP':score=max(score,fb.gate(r,lin));feedback_calls+=1
                    keep[i]=not(score<ts[name])
                else:fs.append(fb.feature_record(r,CONT[r['event']],lin))
            if name not in ['One_pass','Feedback_NP']:
                mn='Event_HGB' if name=='Selective_feedback' else name;cols,m=models[mn];d=pd.DataFrame(fs)
                if mn=='Event_HGB':d=fb.events(d)
                ss=m.predict_proba(d[cols])[:,1]
                if name!='Selective_feedback':
                    for i,s in zip(ids,ss):keep[i]=not(s<ts[name])
                else:
                    ze=ce.transform(ss,pars['event_center'],pars['event_scale'],True)
                    for i,p,z in zip(ids,ss,ze):
                        if z<threshold:keep[i]=False;direct_bypass+=1
                        elif p<=pars['router']:
                            r=raws[i];lin=coupled30(r,CONT[r['event']],True);f=max(lin['violation'],fb.gate(r,lin));feedback_calls+=1
                            keep[i]=not(ce.transform(f,pars['physics_center'],pars['physics_scale'])<threshold)
        for i,r in enumerate(raws):
            if keep[i]:coupled30(r,CONT[r['event']])
        elapsed=time.perf_counter()-t0
        if name=='Selective_feedback':assert np.array_equal(~keep,np.array([expected.loc[r['pair']] for r in raws],bool))
        return dict(method=name,seconds=elapsed,n=len(raws),bypassed=int(sum(~keep)),retained=int(sum(keep)),feedback_calls=feedback_calls,direct_bypass=direct_bypass)
    for r in raws:features(r,CONT[r['event']])
    rng=np.random.default_rng(ce.CONFIG['timing_seed']);rows=[]
    for rep in range(protocol['timing_repeats']):
        for name in rng.permutation(methods):
            row=pipeline(name);row['repeat']=rep;rows.append(row);print(row,flush=True)
    r=pd.DataFrame(rows);r.to_csv(OUT/'final_serial_timing.csv',index=False)
    s=r.groupby('method').agg(median_seconds=('seconds','median'),minimum_seconds=('seconds','min'),maximum_seconds=('seconds','max'),n=('n','first'),bypassed=('bypassed','first'),feedback_calls=('feedback_calls','first'),direct_bypass=('direct_bypass','first')).reset_index()
    base=float(s.loc[s.method=='Full_AC','median_seconds'].iloc[0]);s['saving_percent']=100*(1-s.median_seconds/base);s.to_csv(OUT/'final_timing_summary.csv',index=False)
    print(s.to_string(index=False),flush=True)

if __name__=='__main__':main()
