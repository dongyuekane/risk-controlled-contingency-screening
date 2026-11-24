"""Frozen physical feedback gate; fresh confirmation and matched timing.

The development observation is the four already-public nominal false-safe cases.
The added gate uses only fixed engineering limits, without a fitted gate threshold.
Confirmation seeds and all method settings are declared here before generation.
The direct corrected-physics NP baseline tests whether the ML score is necessary.
"""
from pathlib import Path
import sys,json,time,hashlib,platform,os
sys.dont_write_bytecode=True
SOURCE=Path(os.environ.get('POWER_STUDIES_WORK',str(Path(__file__).resolve().parent/'original_source')))
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE/'package/SCI_Three_Studies/code'))
sys.path.insert(0,str(SOURCE/'vendor'))
import numpy as np,pandas as pd,sklearn,scipy
from scipy.stats import binom,beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
from study2 import coupled30,features,CONT,FEATURES2,dc_matrix
from grid import feeder,radial,case,sample,main_connected

CONFIG=dict(alpha=.01,delta=.05,confirmation_seed=2026110700,n_confirmation=1000,
            gate_tolerance=1e-7,timing_n=400,timing_repeats=3,timing_seed=20261107,
            hypothesis='Reuse the first-pass main-grid boundary voltages to reject candidate bypasses whose nonlinear radial feeder sweep violates any distribution constraint; preserve the original NP threshold.')
SUPPORT=['P','Q','scale','tap','amp']
EVENTCOLS=[f'outage_{i}' for i in range(41)]

def threshold(scores,y):
    s=np.sort(np.asarray(scores)[np.asarray(y)==1]);n=len(s);ks=np.arange(1,n+1)
    valid=ks[binom.cdf(ks-1,n,CONFIG['alpha'])<=CONFIG['delta']]
    if not len(valid):return -np.inf,0
    k=int(valid[-1]);return float(s[k-1]),k

def gate(raw,lin):
    if lin['status']!='converged':return 3.
    fc,edges,_=feeder(raw['name']);p=np.asarray(raw['p']);q=np.asarray(raw['q'])
    fa=radial(fc,edges,p,q,lin['boundary_a']*raw['tap'],False,raw['amp'])
    fb=radial(fc,edges,.95*p,.95*q,lin['boundary_b']*raw['tap'],False,raw['amp'])
    if not fa['ok'] or not fb['ok']:return 3.
    return float(max((.9-min(fa['vmin'],fb['vmin']))/.1,
                     (max(fa['vmax'],fb['vmax'])-1.1)/.1,
                     fa['thermal']-1,fb['thermal']-1,
                     fa['tr_ratio']-1,fb['tr_ratio']-1))

def feature_record(raw,out,lin):
    p=case('case30')['bus'][:,2].copy()*raw['scale'];fc,_,_=feeder(raw['name']);nom=fc['bus'][:,2].sum()
    p[6]=max(0,p[6]/raw['scale']-4*nom)*raw['scale']+4*sum(raw['p'])
    p[20]=max(0,p[20]/raw['scale']-3.8*nom)*raw['scale']+3.8*sum(raw['p'])
    inj=-p.copy()
    for g in case('case30')['gen']:inj[int(g[0])-1]+=g[1]*raw['scale']
    flow=dc_matrix(tuple(out))@inj[1:];bflow=dc_matrix(())@inj[1:];rating=case('case30')['branch'][:,5]
    result=dict(dc_ratio=float(max(abs(flow)/rating)),dc_change=float(max(abs(flow-bflow)/rating)),lin_violation=lin['violation'],lin_vmin=lin.get('vmin',.5),lin_tr=lin.get('tr',3.),lin_main=lin.get('main',3.),P=sum(raw['p']),Q=sum(raw['q']),scale=raw['scale'],tap=raw['tap'],amp=raw['amp'],n_out=len(out))
    return result

def events(d):
    d=d.copy();a=np.zeros((len(d),41))
    for i,e in enumerate(d.event.astype(int)):a[i,list(CONT[e])]=1
    return pd.concat([d.reset_index(drop=True),pd.DataFrame(a,columns=EVENTCOLS)],axis=1)

def fit():
    d=pd.read_csv(SOURCE/'package/SCI_Three_Studies/outputs/study2_cases.csv').query('connected and split=="train"')
    lo=d[SUPPORT].min().to_numpy();hi=d[SUPPORT].max().to_numpy();d=events(d)
    models={}
    for name,cols,m in [
        ('Physics_HGB',FEATURES2,HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,l2_regularization=4,random_state=11)),
        ('Logistic',FEATURES2,make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=1000))),
        ('Event_HGB',SUPPORT+['n_out']+EVENTCOLS,HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,l2_regularization=4,random_state=11))]:
        models[name]=(cols,m.fit(d[cols],d.unsafe))
    return models,lo,hi

def supported(raw,lo,hi):
    z=np.array([sum(raw['p']),sum(raw['q']),raw['scale'],raw['tap'],raw['amp']])
    return raw['name']=='case33bw' and bool(np.all(z>=lo)&np.all(z<=hi))

def calibrate(models):
    ca=pd.read_csv(SOURCE/'package/SCI_Three_Studies/outputs/revision2_pairs.csv').query('split=="np_calibration"').copy()
    cr={r['pair']:r for r in json.loads((SOURCE/'package/SCI_Three_Studies/data/revision2_inputs.json').read_text())}
    if (HERE/'feedback_calibration.csv').exists():
        ds=pd.read_csv(HERE/'feedback_calibration.csv');assert list(ds.pair)==list(ca.pair)
    else:
        rows=[]
        for j,r in enumerate(ca.itertuples()):
            raw=cr[r.pair];lin=coupled30(raw,CONT[r.event],True);gm=gate(raw,lin)
            rows.append(dict(pair=r.pair,unsafe=r.unsafe,gate_margin=gm,feedback_score=max(lin['violation'],gm)))
            if (j+1)%500==0:print('calibration feedback',j+1,flush=True)
        ds=pd.DataFrame(rows);ds.to_csv(HERE/'feedback_calibration.csv',index=False)
    ca=events(ca);scores={'One_pass':ca.lin_violation.to_numpy(),'Feedback_NP':ds.feedback_score.to_numpy()}
    for name,(cols,m) in models.items():scores[name]=m.predict_proba(ca[cols])[:,1]
    ts={};rows=[]
    for name,s in scores.items():
        t,k=threshold(s,ca.unsafe);ts[name]=t;rows.append(dict(method=name,threshold=t,order=k,n_unsafe=int(ca.unsafe.sum()),alpha=CONFIG['alpha'],delta=CONFIG['delta']))
    frozen=pd.read_csv(SOURCE/'study2_audit/matched_thresholds.csv').query('alpha==.01').set_index('method').threshold
    for name in ['One_pass','Physics_HGB','Logistic','Event_HGB']:assert abs(ts[name]-frozen[name])<1e-12,(name,ts[name],frozen[name])
    pd.DataFrame(rows).to_csv(HERE/'feedback_thresholds.csv',index=False)
    return ts

def generate(lo,hi):
    if (HERE/'confirmation_cases.csv').exists():return pd.read_csv(HERE/'confirmation_cases.csv'),json.loads((HERE/'confirmation_inputs.json').read_text())
    connected=[i for i,c in enumerate(CONT) if main_connected(c,'case30')];rows=[];raws=[]
    for j in range(CONFIG['n_confirmation']):
        seed=CONFIG['confirmation_seed']+j;attempt=0;raw=sample('case33bw',seed)
        while not coupled30(raw)['safe']:
            attempt+=1;assert attempt<500;raw=sample('case33bw',seed+1000000*attempt)
        event=int(np.random.default_rng(seed+800000).choice(connected));out=CONT[event]
        raw.update(pair=seed,event=event,attempt=attempt)
        lin=coupled30(raw,out,True);f=feature_record(raw,out,lin);gm=gate(raw,lin);a=coupled30(raw,out)
        rows.append(dict(pair=seed,event=event,order=len(out),supported=supported(raw,lo,hi),unsafe=int(not a['safe']),status=a['status'],ac_margin=a['violation'],gate_margin=gm,feedback_score=max(lin['violation'],gm),**f))
        raws.append(raw)
        if (j+1)%100==0:print('confirmation',j+1,flush=True)
    d=pd.DataFrame(rows);d.to_csv(HERE/'confirmation_cases.csv',index=False);(HERE/'confirmation_inputs.json').write_text(json.dumps(raws))
    return d,raws

def classify(d,models,ts):
    d=events(d);scores={name:m.predict_proba(d[cols])[:,1] for name,(cols,m) in models.items()}
    scores['One_pass']=d.lin_violation.to_numpy();scores['Feedback_NP']=d.feedback_score.to_numpy()
    by={name:(s<ts[name])&d.supported.to_numpy() for name,s in scores.items()}
    by['Physics_HGB_guard']=by['Physics_HGB']&(d.gate_margin.to_numpy()<=CONFIG['gate_tolerance'])
    rows=[];pred=[]
    for name,b in by.items():
        for order in [0,1,2]:
            ix=np.ones(len(d),bool) if order==0 else d.order.to_numpy()==order;y=d.unsafe.to_numpy()[ix];bb=b[ix];f=int(sum(y*bb));n=int(sum(y))
            margins=d.ac_margin.to_numpy()[ix][(y*bb).astype(bool)]
            rows.append(dict(method=name,order=order,n=int(sum(ix)),unsafe=n,bypassed=int(sum(bb)),false_safe=f,false_safe_rate=f/n if n else np.nan,upper95=beta.ppf(.95,f+1,n-f) if n>f else 1,maximum_missed_margin=max(margins,default=0)))
        for i,r in enumerate(d.itertuples()):pred.append(dict(pair=r.pair,method=name,bypass=int(b[i]),unsafe=r.unsafe,ac_margin=r.ac_margin))
    pd.DataFrame(rows).to_csv(HERE/'confirmation_results.csv',index=False);pd.DataFrame(pred).to_csv(HERE/'confirmation_predictions.csv',index=False)
    print(pd.DataFrame(rows).query('order==0').to_string(index=False),flush=True)

def time_methods(raws,models,ts,lo,hi):
    subset=raws[:CONFIG['timing_n']];methods=['Full_AC','One_pass','Feedback_NP','Physics_HGB','Physics_HGB_guard','Logistic','Event_HGB']
    def pipeline(name):
        start=time.perf_counter();keep=np.ones(len(subset),bool);lins={};fs=[];ids=[];gates=0
        if name!='Full_AC':
            for i,r in enumerate(subset):
                if not supported(r,lo,hi):continue
                ids.append(i)
                if name=='Event_HGB':
                    fs.append(dict(event=r['event'],P=sum(r['p']),Q=sum(r['q']),scale=r['scale'],tap=r['tap'],amp=r['amp'],n_out=len(CONT[r['event']])));continue
                lin=coupled30(r,CONT[r['event']],True);lins[i]=lin
                if name in ['One_pass','Feedback_NP']:
                    val=lin['violation']
                    if name=='Feedback_NP':val=max(val,gate(r,lin));gates+=1
                    keep[i]=not (val<ts[name])
                else:fs.append(feature_record(r,CONT[r['event']],lin))
            if name not in ['One_pass','Feedback_NP']:
                mn='Physics_HGB' if name=='Physics_HGB_guard' else name
                cols,m=models[mn];dd=pd.DataFrame(fs)
                if mn=='Event_HGB':dd=events(dd)
                ss=m.predict_proba(dd[cols])[:,1]
                for i,s in zip(ids,ss):
                    keep[i]=not (s<ts[mn])
                    if not keep[i] and name=='Physics_HGB_guard':
                        gates+=1;keep[i]=gate(subset[i],lins[i])>CONFIG['gate_tolerance']
        for i,r in enumerate(subset):
            if keep[i]:coupled30(r,CONT[r['event']])
        return dict(method=name,seconds=time.perf_counter()-start,n=len(subset),bypassed=int(sum(~keep)),retained=int(sum(keep)),guard_calls=gates)
    for r in subset:features(r,CONT[r['event']])
    rng=np.random.default_rng(CONFIG['timing_seed']);rows=[]
    for rep in range(CONFIG['timing_repeats']):
        for name in rng.permutation(methods):
            r=pipeline(name);r['repeat']=rep;rows.append(r);print('timing',r,flush=True)
    pd.DataFrame(rows).to_csv(HERE/'feedback_timing.csv',index=False)

def main():
    (HERE/'feedback_protocol.json').write_text(json.dumps(CONFIG,indent=2))
    models,lo,hi=fit();ts=calibrate(models);d,raws=generate(lo,hi);classify(d,models,ts);time_methods(raws,models,ts,lo,hi)
    (HERE/'feedback_environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,source=str(SOURCE)),indent=2))

if __name__=='__main__':main()
