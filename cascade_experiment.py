"""Training-selected cost-aware cascade with ONE calibrated composite score.

The original frozen HGB is the cheap event-aware score (no AC features).
One feedback-corrected physics score is computed only for undecided cases.
Both scales and the high-score rejection cutoff use original training data.
Only the scalar composite score is calibrated on independent pairs.
"""
from pathlib import Path
import json,time,shutil
import feedback_guard_experiment as fb
import numpy as np,pandas as pd
from scipy.special import logit
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from study2 import coupled30,CONT
from grid import main_connected,sample

HERE=Path(__file__).resolve().parent
OUT=HERE/'cascade';OUT.mkdir(exist_ok=True)
CONFIG=dict(confirm_seed=2026112700,n_confirm=1000,folds=3,router_candidates=[.2,.5,.8,1.],
            scale_quantiles=[.01,.10],cost_training_cases=120,alpha=.01,delta=.05,
            timing_n=400,timing_repeats=5,timing_seed=2026112711)

def transform(v,center,scale,is_probability=False):
    a=logit(np.clip(np.asarray(v),1e-12,1-1e-12)) if is_probability else np.asarray(v)
    return (a-center)/scale

def training():
    d=pd.read_csv(fb.SOURCE/'package/SCI_Three_Studies/outputs/study2_cases.csv').query('connected and split=="train"').reset_index(drop=True)
    raws={r['block']:r for r in json.loads((fb.SOURCE/'package/SCI_Three_Studies/data/study2_inputs.json').read_text())}
    if (OUT/'training_scores.csv').exists():return pd.read_csv(OUT/'training_scores.csv')
    ev=fb.events(d);cols=fb.SUPPORT+['n_out']+fb.EVENTCOLS;pred=np.empty(len(d))
    block_to_fold={b:i%CONFIG['folds'] for i,b in enumerate(sorted(d.block.unique()))};fold=d.block.map(block_to_fold).to_numpy()
    for k in range(CONFIG['folds']):
        m=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,l2_regularization=4,random_state=11)
        m.fit(ev.loc[fold!=k,cols],d.loc[fold!=k,'unsafe']);pred[fold==k]=m.predict_proba(ev.loc[fold==k,cols])[:,1]
    phys=[];times=[]
    selected=set(np.linspace(0,len(d)-1,CONFIG['cost_training_cases'],dtype=int))
    for i,r in enumerate(d.itertuples()):
        raw=raws[r.block];out=CONT[r.event]
        t=time.perf_counter();lin=coupled30(raw,out,True);gm=fb.gate(raw,lin);fp=time.perf_counter()-t
        phys.append(max(lin['violation'],gm))
        if i in selected:
            t=time.perf_counter();coupled30(raw,out);ac=time.perf_counter()-t
            times.append(dict(row=i,block=r.block,event=r.event,feedback_seconds=fp,ac_seconds=ac))
        if (i+1)%1000==0:print('training physics',i+1,flush=True)
    d['event_oof_score']=pred;d['feedback_score']=phys
    d.to_csv(OUT/'training_scores.csv',index=False);pd.DataFrame(times).to_csv(OUT/'training_costs.csv',index=False)
    return d

def select_router(d):
    y=d.unsafe.to_numpy();mask=y==1
    e=logit(np.clip(d.event_oof_score.to_numpy(),1e-12,1-1e-12));p=d.feedback_score.to_numpy()
    ec,eu=np.quantile(e[mask],CONFIG['scale_quantiles']);pc,pu=np.quantile(p[mask],CONFIG['scale_quantiles'])
    pars=dict(event_center=float(ec),event_scale=float(max(eu-ec,1e-12)),physics_center=float(pc),physics_scale=float(max(pu-pc,1e-12)))
    costs=pd.read_csv(OUT/'training_costs.csv');cp=float(costs.feedback_seconds.median());ca=float(costs.ac_seconds.median())
    ze=transform(d.event_oof_score,ec,pars['event_scale'],True);zp=transform(p,pc,pars['physics_scale']);rows=[]
    for r in CONFIG['router_candidates']:
        ss=np.where(d.event_oof_score<=r,np.minimum(ze,zp),ze);t,k=fb.threshold(ss,y)
        direct=ze<t;feedback=(~direct)&(d.event_oof_score.to_numpy()<=r);bypass=direct|(feedback&(zp<t))
        cost=float(sum(feedback)*cp+sum(~bypass)*ca)
        rows.append(dict(router=r,pseudo_threshold=t,pseudo_order=k,direct_bypass=int(sum(direct)),feedback_calls=int(sum(feedback)),bypassed=int(sum(bypass)),training_cost=cost,feedback_cost=cp,ac_cost=ca))
    dr=pd.DataFrame(rows);dr.to_csv(OUT/'router_selection.csv',index=False)
    pars['router']=float(dr.loc[dr.training_cost.idxmin(),'router']);pars['scope']='Training-only proxy cost selects routing. Actual calibration and held-out timing are separate.'
    (OUT/'frozen_router.json').write_text(json.dumps(pars,indent=2));return pars

def calibration(models,pars):
    ca=pd.read_csv(fb.SOURCE/'package/SCI_Three_Studies/outputs/revision2_pairs.csv').query('split=="np_calibration"').reset_index(drop=True)
    fca=pd.read_csv(HERE/'feedback_calibration.csv');assert list(fca.pair)==list(ca.pair)
    ev=fb.events(ca);cols,m=models['Event_HGB'];es=m.predict_proba(ev[cols])[:,1]
    ze=transform(es,pars['event_center'],pars['event_scale'],True);zp=transform(fca.feedback_score,pars['physics_center'],pars['physics_scale'])
    ss=np.where(es<=pars['router'],np.minimum(ze,zp),ze);t,k=fb.threshold(ss,ca.unsafe)
    result=dict(threshold=t,order=k,n_unsafe=int(ca.unsafe.sum()),alpha=.01,delta=.05,
                definition='s=min(z_event,z_feedback) when event_probability<=router; s=z_event otherwise. Exactly one NP order-statistic calibration is applied to s.')
    (OUT/'cascade_calibration.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(dict(pair=ca.pair,unsafe=ca.unsafe,event_score=es,event_scaled=ze,feedback_scaled=zp,composite_score=ss)).to_csv(OUT/'calibration_scores.csv',index=False)
    return t

def confirm(models,pars,t,lo,hi,base_thresholds):
    # Generate only after training/router and independent calibration are frozen.
    if (OUT/'confirmation_cases.csv').exists():d=pd.read_csv(OUT/'confirmation_cases.csv');raws=json.loads((OUT/'confirmation_inputs.json').read_text())
    else:
        old_here=fb.HERE;old_conf=fb.CONFIG.copy();fb.HERE=OUT
        fb.CONFIG.update(confirmation_seed=CONFIG['confirm_seed'],n_confirmation=CONFIG['n_confirm'])
        d,raws=fb.generate(lo,hi);fb.HERE=old_here;fb.CONFIG=old_conf
    ev=fb.events(d);cols,m=models['Event_HGB'];es=m.predict_proba(ev[cols])[:,1]
    ze=transform(es,pars['event_center'],pars['event_scale'],True);zp=transform(d.feedback_score,pars['physics_center'],pars['physics_scale'])
    direct=(ze<t)&d.supported.to_numpy();feedback=(~direct)&(es<=pars['router'])&d.supported.to_numpy();bypass=direct|(feedback&(zp<t))
    ss=np.where(es<=pars['router'],np.minimum(ze,zp),ze);assert np.array_equal(bypass,(ss<t)&d.supported.to_numpy())
    oldhere=fb.HERE;fb.HERE=OUT;fb.classify(d,models,base_thresholds);fb.HERE=oldhere
    rows=[]
    for order in [0,1,2]:
        ix=np.ones(len(d),bool) if order==0 else d.order.to_numpy()==order;y=d.unsafe.to_numpy()[ix];bb=bypass[ix];n=int(sum(y));f=int(sum(y*bb));margin=d.ac_margin.to_numpy()[ix][(y*bb).astype(bool)]
        rows.append(dict(method='Selective_feedback',order=order,n=int(sum(ix)),unsafe=n,bypassed=int(sum(bb)),false_safe=f,false_safe_rate=f/n if n else np.nan,upper95=float(beta.ppf(.95,f+1,n-f)) if n>f else 1.,maximum_missed_margin=max(margin,default=0),direct_bypass=int(sum(direct[ix])),feedback_calls=int(sum(feedback[ix]))))
    pd.concat([pd.read_csv(OUT/'confirmation_results.csv'),pd.DataFrame(rows)],ignore_index=True).to_csv(OUT/'confirmation_results.csv',index=False)
    pd.DataFrame(dict(pair=d.pair,event_probability=es,event_scaled=ze,feedback_scaled=zp,score=ss,direct_bypass=direct,feedback_called=feedback,bypass=bypass,unsafe=d.unsafe,ac_margin=d.ac_margin)).to_csv(OUT/'cascade_predictions.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

def main():
    (OUT/'protocol.json').write_text(json.dumps(CONFIG,indent=2));train=training();pars=select_router(train);models,lo,hi=fb.fit();t=calibration(models,pars)
    ts=pd.read_csv(HERE/'feedback_thresholds.csv').set_index('method').threshold.to_dict()
    confirm(models,pars,t,lo,hi,ts)
    print('No headline timing run here. Reserve a serial timing window.',flush=True)

if __name__=='__main__':main()
