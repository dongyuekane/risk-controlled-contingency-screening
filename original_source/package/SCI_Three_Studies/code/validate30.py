"""Independent full-network Newton crosschecks for Study 2."""
import json
import numpy as np,pandas as pd
from pypower.api import runpf,ppoption
from grid import *
from study2 import coupled30

def monolithic30(raw,outages):
    fc,n,_=feeder(raw['name']);mc=case('case30');bus=mc['bus'].copy();gen=mc['gen'].copy();br=mc['branch'].copy();scale=raw['scale']
    bus[:,2:4]*=scale;gen[:,1]*=scale
    for k in outages:br[k,10]=0
    nom=fc['bus'][:,2:4].sum(axis=0);p=np.array(raw['p']);q=np.array(raw['q'])
    for idx,mult,factor in [(6,4,1.),(20,4,.95)]:
        bus[idx,2:4]=np.maximum(mc['bus'][idx,2:4]-mult*factor*nom,0)*scale
        start=len(bus);fb=fc['bus'].copy();fb[:,0]=np.arange(start+1,start+len(fb)+1);fb[:,1]=1;fb[:,2]=p*factor*mult;fb[:,3]=q*factor*mult;bus=np.vstack([bus,fb])
        eb=np.zeros((len(n),13))
        for k,e in enumerate(n):eb[k]=[start+e[0]+1,start+e[1]+1,e[2]*100/fc['baseMVA']/mult,e[3]*100/fc['baseMVA']/mult,0,1e6,1e6,1e6,0,0,1,-360,360]
        tb=np.array([[idx+1,start+1,.006*100/(5*mult),.04*100/(5*mult),0,1e6,1e6,1e6,1/raw['tap'],0,1,-360,360]])
        br=np.vstack([br,tb,eb])
    return runpf(dict(version='2',baseMVA=100.,bus=bus,gen=gen,branch=br),ppoption(VERBOSE=0,OUT_ALL=0,PF_TOL=1e-10))

def run():
    rows=[]
    for name in ['case33bw','case69']:
        for j in range(20):
            r=sample(name,88000+j);out=[(),(0,),(0,9),(2,20)][j%4]
            a=coupled30(r,out);b,ok=monolithic30(r,out)
            if a['status']!='converged' or not ok:raise AssertionError((name,j,a,ok))
            err=max(abs(a['boundary_a']-b['bus'][6,7]),abs(a['boundary_b']-b['bus'][20,7]),abs(a['vmin']-min(b['bus'][30:,7])))
            assert err<3e-7,(name,j,err)
            rows.append(dict(network=name,seed=r['seed'],outages=str(out),voltage_error=err,residual=a['residual'],passed=True))
    pd.DataFrame(rows).to_csv(ROOT/'outputs/independent_validation30.csv',index=False);print('30-bus full-network checks',len(rows),'PASS; max',max(x['voltage_error'] for x in rows))
if __name__=='__main__':run()
