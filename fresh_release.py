"""Final independent calibration and confirmation after the score is frozen.

All score/routing parameters are copied byte-for-byte from frozen_router.json.
No search or refitting of routing uses these labels.  Every comparator receives
the same new calibration population.  Previous cohorts are development only.
"""
from pathlib import Path
import json,hashlib,shutil,pickle
import cascade_experiment as ce
import feedback_guard_experiment as fb
import numpy as np,pandas as pd
from scipy.stats import beta,binom

ROOT=ce.HERE
OUT=ROOT/'fresh_release';OUT.mkdir(exist_ok=True)
PROTOCOL=dict(calibration_seed=2026120700,n_calibration=3000,
              confirmation_seed=2026125700,n_confirmation=1000,
              alpha=.01,delta=.05,timing_n=400,timing_repeats=3,
              parameter_source='cascade/frozen_router.json',
              rule='All scales, model fits, router r=0.5 and composite score fixed before these cohorts. Only NP order-statistic thresholds use the fresh calibration labels. No choice uses final confirmation outcomes.')

def generate(part,seed,n,lo,hi):
    destination=OUT/part;destination.mkdir(exist_ok=True)
    oldhere=fb.HERE;oldconfig=fb.CONFIG.copy();fb.HERE=destination
    fb.CONFIG.update(confirmation_seed=seed,n_confirmation=n)
    data,raws=fb.generate(lo,hi)
    fb.HERE=oldhere;fb.CONFIG=oldconfig
    return data,raws

def all_scores(d,models,pars):
    e=fb.events(d);scores={name:m.predict_proba(e[cols])[:,1] for name,(cols,m) in models.items()}
    scores['One_pass']=d.lin_violation.to_numpy();scores['Feedback_NP']=d.feedback_score.to_numpy()
    ze=ce.transform(scores['Event_HGB'],pars['event_center'],pars['event_scale'],True)
    zf=ce.transform(scores['Feedback_NP'],pars['physics_center'],pars['physics_scale'])
    scores['Selective_feedback']=np.where(scores['Event_HGB']<=pars['router'],np.minimum(ze,zf),ze)
    return scores,ze,zf

def main():
    frozen=ce.OUT/'frozen_router.json';pars=json.loads(frozen.read_text())
    shutil.copy2(frozen,OUT/'frozen_router.json')
    PROTOCOL['parameter_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest()
    PROTOCOL['score_code_sha256']=hashlib.sha256(Path(ce.__file__).read_bytes()).hexdigest()
    models,lo,hi=fb.fit()
    model_bytes=pickle.dumps(models,protocol=5)
    (OUT/'frozen_models.pkl').write_bytes(model_bytes)
    PROTOCOL['models_sha256']=hashlib.sha256(model_bytes).hexdigest()
    (OUT/'frozen_support.json').write_text(json.dumps(dict(lo=lo.tolist(),hi=hi.tolist()),indent=2))
    (OUT/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    ca,_=generate('calibration',PROTOCOL['calibration_seed'],PROTOCOL['n_calibration'],lo,hi)
    scores,_,_=all_scores(ca,models,pars);thresholds={};calrows=[];n=int(ca.unsafe.sum())
    for name,s in scores.items():
        t,k=fb.threshold(s,ca.unsafe);thresholds[name]=t
        calrows.append(dict(method=name,threshold=t,order=k,n_unsafe=n,alpha=.01,delta=.05,tail_bound=float(binom.cdf(k-1,n,.01))))
    pd.DataFrame(calrows).to_csv(OUT/'thresholds.csv',index=False)
    # Threshold file is complete before final confirmation generation starts.
    d,raws=generate('confirmation',PROTOCOL['confirmation_seed'],PROTOCOL['n_confirmation'],lo,hi)
    scores,ze,zf=all_scores(d,models,pars);rows=[];pred=[]
    for name,s in scores.items():
        by=(s<thresholds[name])&d.supported.to_numpy()
        for order in [0,1,2]:
            ix=np.ones(len(d),bool) if order==0 else d.order.to_numpy()==order;y=d.unsafe.to_numpy()[ix];bb=by[ix];nu=int(sum(y));fn=int(sum(y*bb));m=d.ac_margin.to_numpy()[ix][(y*bb).astype(bool)]
            rows.append(dict(method=name,order=order,n=int(sum(ix)),unsafe=nu,bypassed=int(sum(bb)),false_safe=fn,false_safe_rate=fn/nu if nu else np.nan,upper95=float(beta.ppf(.95,fn+1,nu-fn)) if nu>fn else 1.,maximum_missed_margin=max(m,default=0)))
        pred.extend(dict(pair=int(r.pair),method=name,score=float(s[i]),bypass=int(by[i]),unsafe=int(r.unsafe),ac_margin=float(r.ac_margin)) for i,r in enumerate(d.itertuples()))
    t=thresholds['Selective_feedback'];direct=(ze<t)&d.supported.to_numpy();call=(~direct)&(scores['Event_HGB']<=pars['router'])&d.supported.to_numpy();by=direct|(call&(zf<t))
    assert np.array_equal(by,(scores['Selective_feedback']<t)&d.supported.to_numpy())
    pd.DataFrame(rows).to_csv(OUT/'confirmation_results.csv',index=False);pd.DataFrame(pred).to_csv(OUT/'confirmation_predictions.csv',index=False)
    pd.DataFrame(dict(pair=d.pair,event_probability=scores['Event_HGB'],event_scaled=ze,feedback_scaled=zf,composite_score=scores['Selective_feedback'],direct_bypass=direct,feedback_called=call,bypass=by,unsafe=d.unsafe,ac_margin=d.ac_margin)).to_csv(OUT/'cascade_predictions.csv',index=False)
    summary=dict(calibration_n=len(ca),calibration_unsafe=n,confirmation_n=len(d),confirmation_unsafe=int(d.unsafe.sum()),supported=int(d.supported.sum()),direct_bypass=int(sum(direct)),feedback_calls=int(sum(call)),feedback_bypass=int(sum(call&by)),high_score_to_ac=int(sum(d.supported.to_numpy()&~direct&~call)),outside_support=int(sum(~d.supported.to_numpy())),status=d.status.value_counts().to_dict(),score_parameters_sha256=PROTOCOL['parameter_sha256'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(pd.DataFrame(rows).query('order==0').to_string(index=False),flush=True);print(summary,flush=True)

if __name__=='__main__':main()
