"""Conditional operating-risk calibration, not field outage-probability estimation."""
import json,time,pickle,itertools
from functools import lru_cache
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,brier_score_loss
from scipy.special import logit
from grid import *

CONT=[(i,) for i in range(41)]+list(itertools.combinations(range(41),2))

@lru_cache(None)
def dc_matrix(outages):
    mc=case('case30');br=mc['branch'];A=np.zeros((41,30));sus=np.zeros(41)
    for i,e in enumerate(br):
        if i in outages:continue
        a,b=e[:2].astype(int)-1;A[i,a]=1;A[i,b]=-1;sus[i]=1/e[3]
    B=A.T@(sus[:,None]*A)
    try:H=sus[:,None]*A[:,1:]@np.linalg.inv(B[1:,1:])
    except np.linalg.LinAlgError:return None
    return H

def coupled30(raw,outages=(),linear=False):
    if not main_connected(outages,'case30'):return dict(safe=False,violation=3.,status='separation')
    fc,n,_=feeder(raw['name']);p=np.array(raw['p']);q=np.array(raw['q']);scale=raw['scale'];nom=fc['bus'][:,2:4].sum(axis=0);v=np.ones(2);ok=False
    for k in range(1 if linear else 35):
        fa=radial(fc,n,p,q,v[0]*raw['tap'],linear,raw['amp']);fb=radial(fc,n,p*.95,q*.95,v[1]*raw['tap'],linear,raw['amp'])
        if not fa['ok'] or not fb['ok']:return dict(safe=False,violation=3.,status='distribution_nonconvergence')
        pd=case('case30')['bus'][:,2].copy()*scale;qd=case('case30')['bus'][:,3].copy()*scale
        for idx,mult,fac,f in [(6,4,1.,fa),(20,4,.95,fb)]:
            pd[idx]=max(0,case('case30')['bus'][idx,2]-mult*fac*nom[0])*scale+mult*f['source'].real
            qd[idx]=max(0,case('case30')['bus'][idx,3]-mult*fac*nom[1])*scale+mult*f['source'].imag
        a=main_ac(pd,qd,scale,outages,net='case30')
        if not a['ok']:return dict(safe=False,violation=3.,status=a['reason'])
        nv=abs(a['v'][[6,20]]);err=max(abs(v-nv));v=nv
        if linear or err<1e-8:ok=True;break
    b=case('case30')['bus'];tv=max(np.max((b[:,12]-abs(a['v']))/.1),np.max((abs(a['v'])-b[:,11])/.1),a['ratio']-1,a['genviol'])
    dv=max((.9-min(fa['vmin'],fb['vmin']))/.1,(max(fa['vmax'],fb['vmax'])-1.1)/.1,fa['thermal']-1,fb['thermal']-1,fa['tr_ratio']-1,fb['tr_ratio']-1)
    viol=max(tv,dv) if ok else 3.
    return dict(safe=bool(ok and viol<=1e-7),violation=float(viol),status='converged' if ok else 'coupling_nonconvergence',vmin=min(fa['vmin'],fb['vmin']),vmain=float(min(abs(a['v']))),dist=max(fa['thermal'],fb['thermal']),tr=max(fa['tr_ratio'],fb['tr_ratio']),main=a['ratio'],genviol=a['genviol'],boundary_a=float(v[0]),boundary_b=float(v[1]),residual=max(a['residual'],fa['residual'],fb['residual']),iterations=k+1)

def features(raw,outages):
    tdc=time.perf_counter()
    p=case('case30')['bus'][:,2].copy()*raw['scale'];fc,_,_=feeder(raw['name']);nom=fc['bus'][:,2].sum()
    p[6]=max(0,p[6]/raw['scale']-4*nom)*raw['scale']+4*sum(raw['p'])
    p[20]=max(0,p[20]/raw['scale']-3.8*nom)*raw['scale']+3.8*sum(raw['p'])
    inj=-p.copy()
    for g in case('case30')['gen']:inj[int(g[0])-1]+=g[1]*raw['scale']
    H=dc_matrix(tuple(outages));flow=H@inj[1:];bflow=dc_matrix(())@inj[1:];rating=case('case30')['branch'][:,5]
    dcsec=time.perf_counter()-tdc;lin=coupled30(raw,outages,True)
    return dict(dc_sec=dcsec,dc_ratio=float(max(abs(flow)/rating)),dc_change=float(max(abs(flow-bflow)/rating)),lin_violation=lin['violation'],lin_vmin=lin.get('vmin',.5),lin_tr=lin.get('tr',3.),lin_main=lin.get('main',3.),P=sum(raw['p']),Q=sum(raw['q']),scale=raw['scale'],tap=raw['tap'],amp=raw['amp'],n_out=len(outages))

FEATURES2=['dc_ratio','dc_change','lin_violation','lin_vmin','lin_tr','lin_main','P','Q','scale','tap','amp','n_out']

def generate():
    rows=[];raws=[];t0=time.perf_counter()
    # Every test/stress/external operating block enumerates all 41 singles and 820 pairs.
    # Training and calibration sample 140 event identities per operating block.
    specs=[('train',60,200000,'case33bw'),('calibration',25,210000,'case33bw'),('test',12,220000,'case33bw'),('external69',10,230000,'case69'),('stress',10,240000,'case33bw')]
    for split,count,start,name in specs:
        for j in range(count):
            attempt=0;raw=sample(name,start+j,split=='stress');base=coupled30(raw)
            while split in ['train','calibration','test'] and not base['safe']:
                attempt+=1;raw=sample(name,start+j+1000000*attempt);base=coupled30(raw)
                if attempt>500:raise RuntimeError('Unable to sample a secure starting point')
            raw['split']=split;raw['block']=start+j;raw['sampling_attempt']=attempt;raws.append(raw)
            rng=np.random.default_rng(start+j+999)
            ids=sorted(rng.choice(len(CONT),140,replace=False)) if split in ['train','calibration'] else range(len(CONT))
            for cid in ids:
                out=CONT[cid];t=time.perf_counter();connected=main_connected(out,'case30');topsec=time.perf_counter()-t
                # Entire double-event identities with index modulo 7 == 0 are held out.
                unseen=len(out)==2 and cid%7==0
                if split in ['train','calibration'] and unseen:continue
                if connected:
                    t=time.perf_counter();f=features(raw,out);pre=time.perf_counter()-t
                    t=time.perf_counter();a=coupled30(raw,out);acsec=time.perf_counter()-t
                else:f={k:np.nan for k in FEATURES2};a=dict(safe=False,violation=3.,status='separation');pre=0.;acsec=0.
                rows.append(dict(block=start+j,split=split,network=name,event=cid,order=len(out),outages=';'.join(map(str,out)),unseen_event=unseen,connected=connected,base_safe=base['safe'],unsafe=int(not a['safe']),violation=a['violation'],status=a['status'],pre_sec=pre,ac_sec=acsec,topology_sec=topsec,residual=a.get('residual',np.nan),**f))
            if j%5==0:print('Study2',split,j,'rows',len(rows),'seconds',round(time.perf_counter()-t0,1),flush=True)
    (ROOT/'data/study2_inputs.json').write_text(json.dumps(raws));df=pd.DataFrame(rows);df.to_csv(ROOT/'outputs/study2_cases.csv',index=False);return df

def learn(df):
    d=df[df.connected].copy();X=d[FEATURES2];y=d.unsafe.to_numpy();tr=d.split=='train';ca=d.split=='calibration';metrics=[];pred=[];training=[]
    for seed in [11,23,37]:
        t=time.perf_counter();m=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,l2_regularization=4,random_state=seed).fit(X[tr],y[tr])
        lr=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=1000)).fit(X[tr],y[tr])
        cols=['P','Q','scale','tap','amp','n_out'];rawm=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,l2_regularization=4,random_state=seed).fit(d.loc[tr,cols],y[tr]);fit=time.perf_counter()-t
        t=time.perf_counter();p=np.clip(m.predict_proba(X)[:,1],1e-5,1-1e-5);inf=(time.perf_counter()-t)/len(d)
        cap=ca & (d.block%2==0);cat=ca & (d.block%2==1)
        cal=LogisticRegression(C=1e3).fit(logit(p[cap]).reshape(-1,1),y[cap])
        t=time.perf_counter();pc=cal.predict_proba(logit(p).reshape(-1,1))[:,1];calinf=(time.perf_counter()-t)/len(d)
        t=time.perf_counter();pp=lr.predict_proba(X)[:,1];lrinf=(time.perf_counter()-t)/len(d)
        t=time.perf_counter();pr=rawm.predict_proba(d[cols])[:,1];rawinf=(time.perf_counter()-t)/len(d)
        thresholds=np.r_[0.,np.unique(pc[cat])];valid=[x for x in thresholds if sum((pc[cat]<x)&(y[cat]==1))/max(1,sum(y[cat]))<=.01];theta=float(max(valid))
        t=time.perf_counter();supportcols=['P','Q','scale','tap','amp'];lo=d.loc[tr,supportcols].min().to_numpy();hi=d.loc[tr,supportcols].max().to_numpy();ood=np.any((d[supportcols].to_numpy()<lo)|(d[supportcols].to_numpy()>hi),axis=1)|(d.network.to_numpy()!='case33bw')
        guardinf=(time.perf_counter()-t)/len(d)
        infer={'DC':0.,'Logistic':lrinf,'Raw_ML':rawinf,'Physics_ML':inf,'Calibrated':inf+calinf,'Guarded':inf+calinf+guardinf}
        training.append(dict(seed=seed,fit_seconds=fit,threshold=theta,inference_sec=inf,calibration_inference_sec=calinf,guard_sec=guardinf))
        (ROOT/'outputs'/f'study2_model_{seed}.pkl').write_bytes(pickle.dumps(dict(model=m,calibrator=cal,threshold=theta,features=FEATURES2,lo=lo,hi=hi)))
        for split in ['test','external69','stress']:
            mask=d.split==split
            subsets=[('all',mask),('N1',mask&(d.order==1)),('N2',mask&(d.order==2)),('unseen_N2',mask&d.unseen_event)]
            for sub,z in subsets:
                if not z.any():continue
                yy=y[z]
                methods={'DC':(np.clip((d.dc_ratio.to_numpy()-.6)/.8,0,1),d.dc_ratio.to_numpy()<.9,False),'Logistic':(pp,pp<.5,True),'Raw_ML':(pr,pr<.5,False),'Physics_ML':(p,p<.5,True),'Calibrated':(pc,pc<theta,True),'Guarded':(pc,(pc<theta)&~ood,True)}
                for name,(probs,bypass,usespre) in methods.items():
                    skipped=bypass[z];fn=skipped&(yy==1);pre=float(d.loc[z,'pre_sec'].sum()) if usespre else 0.
                    # DC feature construction is timed separately in analysis microbenchmark;
                    # pre_sec contains a conservative upper bound for DC at this stage.
                    if name=='DC':pre=float(d.loc[z,'dc_sec'].sum())
                    sec=pre+float(d.loc[z & ~bypass,'ac_sec'].sum())+int(z.sum())*infer[name]
                    metrics.append(dict(seed=seed,split=split,subset=sub,model=name,n=int(z.sum()),unsafe=int(yy.sum()),bypassed=int(skipped.sum()),false_safe=int(fn.sum()),false_safe_rate=float(fn.sum()/max(1,yy.sum())),unsafe_among_bypassed=float(fn.sum()/max(1,skipped.sum())),brier=brier_score_loss(yy,probs[z]),auroc=roc_auc_score(yy,probs[z]) if len(set(yy))==2 else np.nan,total_sec=sec,exhaustive_sec=float(d.loc[z,'ac_sec'].sum()),pre_sec=pre,theta=theta))
            for pos,i in enumerate(d.index):
                if d.loc[i,'split']!=split:continue
                pred.append(dict(index=i,block=d.loc[i,'block'],event=d.loc[i,'event'],seed=seed,split=split,unsafe=y[pos],p_raw=p[pos],p_cal=pc[pos],guard=bool(ood[pos]),bypass=bool(pc[pos]<theta and not ood[pos])))
    for name,data in [('metrics',metrics),('predictions',pred),('training',training)]:pd.DataFrame(data).to_csv(ROOT/'outputs'/f'study2_{name}.csv',index=False)

def boundary():
    rows=[];t0=time.perf_counter();events=[(),(0,),(0,9)]
    for ev in events:
        assert main_connected(ev,'case30')
        for multiplier in np.arange(.6,1.301,.05):
            for j in range(120):
                rng=np.random.default_rng(280000+j);raw=sample('case33bw',280000+j);raw['scale']=.85
                fc,_,_=feeder('case33bw');noise=np.clip(np.exp(.12*rng.normal(size=len(fc['bus']))),.65,1.4)
                raw['p']=(fc['bus'][:,2]*multiplier*noise).tolist();raw['q']=(fc['bus'][:,3]*multiplier*noise).tolist();raw['amp']=1.;raw['tap']=1.045
                a=coupled30(raw,ev);rows.append(dict(event='base' if not ev else '-'.join(map(str,ev)),multiplier=round(float(multiplier),2),draw=j,secure=int(a['safe']),violation=a['violation'],status=a['status']))
    pd.DataFrame(rows).to_csv(ROOT/'outputs/study2_boundary.csv',index=False);print('BOUNDARY',len(rows),time.perf_counter()-t0,flush=True)

if __name__=='__main__':
    t=time.perf_counter();d=generate();learn(d);boundary();(ROOT/'outputs/study2_run.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t,rows=len(d))));print('Study2 DONE',flush=True)
