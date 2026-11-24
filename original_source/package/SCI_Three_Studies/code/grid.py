"""Balanced constant-PQ T-D benchmark. All augmentation assumptions are explicit.
Primary solver: 9-bus Newton/root + radial backward/forward sweeps; validation:
independent PYPOWER monolithic Newton solver in validate.py.
No relay dynamics, optimal power flow, Q-limit conversion or intentional islands.
"""
import re
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.optimize import root
ROOT=Path(__file__).resolve().parents[1]

@lru_cache(None)
def case(name):
    if name=='case30':
        from pypower.case30 import case30

        source = case30()
        return {key: np.asarray(source[key]).copy() for key in ('bus', 'branch', 'gen', 'baseMVA')}
    text=(ROOT/'data/public'/f'{name}.m').read_text(); out={}
    for field in ['bus','branch','gen']:
        body=re.search(r'mpc\.'+field+r'\s*=\s*\[(.*?)\];',text,re.S).group(1)
        out[field]=np.array([[float(x) for x in l.split('%')[0].strip().rstrip(';').split()] for l in body.splitlines() if l.split('%')[0].strip()])
    out['baseMVA']=float(re.search(r'mpc.baseMVA\s*=\s*([\d.]+)',text).group(1))
    if name!='case9':
        out['bus'][:,2:4]/=1000
        out['branch'][:,2:4]/=out['bus'][0,9]**2/out['baseMVA']
    return out

def feeder(name):
    c=case(name); br=c['branch']; normal=[];ties=[]
    for k,e in enumerate(br):
        a=[e[0]-1,e[1]-1,e[2],e[3],400. if k<5 else 300.]
        (normal if e[10] else ties).append(a)
    if not ties:
        zb=c['bus'][0,9]**2/c['baseMVA']
        ties=[[10,42,.8/zb,.8/zb,240.],[14,45,.8/zb,.8/zb,240.],[26,64,.8/zb,.8/zb,240.],[49,58,.8/zb,.8/zb,240.],[19,68,.8/zb,.8/zb,240.]]
    for e in ties:e[4]=240.
    return c,np.array(normal),np.array(ties)

def tree(edges,n):
    adj=[[] for _ in range(n)]
    for k,e in enumerate(edges):
        a,b=e[:2].astype(int);adj[a].append((b,k));adj[b].append((a,k))
    par=np.full(n,-2,int);pe=np.zeros(n,int);par[0]=-1;order=[0]
    for a in order:
        for b,k in adj[a]:
            if par[b]==-2:par[b]=a;pe[b]=k;order.append(b)
    return np.array(order),par,pe

def topology_ok(edges,n):
    return len(edges)==n-1 and len(tree(edges,n)[0])==n

def reachable(edges,n):
    return tree(edges,n)[0]

def radial(fc,edges,p,q,vsource=1.,linear=False,amp=1.,tr_mva=5.,zmult=1.,allow_islands=False):
    n=len(p);base=fc['baseMVA'];kv=fc['bus'][0,9]
    order,par,pe=tree(edges,n)
    if len(order)!=n or len(edges)!=n-1:
        if not allow_islands:return {'ok':False,'reason':'topology'}
        active=np.zeros(n,bool);active[order]=True;p=np.asarray(p)*active;q=np.asarray(q)*active
        internal=sum(active[int(e[0])] and active[int(e[1])] for e in edges)
        if internal!=len(order)-1:return {'ok':False,'reason':'energized_cycle'}
    z=(edges[:,2]+1j*edges[:,3])*zmult;zt=(.006+1j*.04)*base/tr_mva
    s=(p+1j*q)/base;v=np.full(n,complex(vsource));ok=False
    for it in range(1 if linear else 100):
        if np.min(abs(v[order]))<.25 or not np.all(np.isfinite(v)):break
        ib=np.conj(s/v)
        for w in order[:0:-1]:ib[par[w]]+=ib[w]
        vn=np.full(n,complex(vsource));vn[0]=vsource-zt*ib[0]
        for w in order[1:]:vn[w]=vn[par[w]]-z[pe[w]]*ib[w]
        if max(abs(v-vn))<1e-10:ok=True;v=vn;break
        v=vn
    if linear:ok=True
    # Re-evaluate currents at the returned voltages for consistent residuals.
    ib=np.conj(s/v)
    for w in order[:0:-1]:ib[par[w]]+=ib[w]
    amps=np.array([abs(ib[w])*base*1000/(np.sqrt(3)*kv) for w in order[1:]])
    limits=np.array([edges[pe[w],4]*amp for w in order[1:]])
    source=vsource*np.conj(ib[0])*base
    loss=(sum(z[pe[w]].real*abs(ib[w])**2 for w in order[1:])+zt.real*abs(ib[0])**2)*base
    return dict(ok=bool(ok and min(abs(v[order]))>.5),vmin=float(min(abs(v[order]))),vmax=float(max(abs(v[order]))),thermal=float(max(amps/limits,default=0.)),tr_ratio=float(abs(source)/tr_mva),source=source,loss=float(loss),residual=float(abs(source.real-p.sum()-loss)),v=v,served_mw=float(sum(p)))

def main_case(outages=(),upgrade=False,net='case9'):
    main=case(net); br=main['branch'].copy()
    if isinstance(upgrade,dict):
        br[[2,4],2:4]*=upgrade['t_z'];br[[2,4],5:8]*=upgrade['t_rating']
    elif upgrade:
        br[[2,4],2:4]*=.7;br[[2,4],5:8]*=1.5
    for k in outages:br[k,10]=0
    return br

def main_connected(outages=(),net='case9'):
    br=main_case(outages,net=net); ed=br[br[:,10]>0,:2]-1;n=len(case(net)['bus'])
    return len(reachable(ed,n))==n

def main_ac(pd,qd,scale=1.,outages=(),upgrade=False,net='case9'):
    b=case(net)['bus']; gen=case(net)['gen'];br=main_case(outages,upgrade,net)
    if not main_connected(outages,net):return {'ok':False,'reason':'separation'}
    n=len(b);Y=np.diag((b[:,4]+1j*b[:,5])/100.)
    for e in br:
        if e[10]==0:continue
        u,w=e[:2].astype(int)-1;y=1/complex(e[2],e[3]);sh=1j*e[4]/2;tap=e[8] or 1
        Y[u,u]+=(y+sh)/tap**2;Y[w,w]+=y+sh;Y[u,w]-=y/tap;Y[w,u]-=y/tap
    pq=np.where(b[:,1]==1)[0];ns=np.where(b[:,1]!=3)[0];vm=b[:,7].copy();pg=np.zeros(n)
    for g in gen:pg[int(g[0])-1]=g[1]*scale;vm[int(g[0])-1]=g[5]
    P=(pg-pd)/100.;Q=-qd/100.
    def unpack(x):
        va=np.zeros(n);va[ns]=x[:len(ns)];vv=vm.copy();vv[pq]=x[len(ns):];return vv*np.exp(1j*va)
    def fun(x):
        v=unpack(x);s=v*np.conj(Y@v);return np.r_[s.real[ns]-P[ns],s.imag[pq]-Q[pq]]
    rr=root(fun,np.r_[np.zeros(len(ns)),np.ones(len(pq))],tol=1e-9)
    v=unpack(rr.x);res=max(abs(fun(rr.x)))
    if res>1e-7 or min(abs(v))<.5 or max(abs(v))>1.5:return {'ok':False,'reason':'nonconvergence','residual':float(res)}
    ratios=[];flows=[]
    for e in br:
        if e[10]==0:ratios.append(0.);flows.append(0j);continue
        a,w=e[:2].astype(int)-1;y=1/complex(e[2],e[3]);sh=1j*e[4]/2
        sa=v[a]*np.conj((y+sh)*v[a]-y*v[w])*100
        sb=v[w]*np.conj((y+sh)*v[w]-y*v[a])*100
        ratios.append(max(abs(sa),abs(sb))/e[5]);flows.append(sa)
    inj=v*np.conj(Y@v)*100;gval=inj+pd+1j*qd;gviol=[]
    for g in gen:
        s=gval[int(g[0])-1]
        gviol += [(s.real-g[8])/g[8],(g[9]-s.real)/g[8],(s.imag-g[3])/max(1,abs(g[3])),(g[4]-s.imag)/max(1,abs(g[4]))]
    return dict(ok=True,v=v,ratio=float(max(ratios)),line_ratios=np.array(ratios),flows=np.array(flows),genviol=float(max(gviol)),residual=float(res),loss=float(inj.real.sum()),generation=gval)

def coupled(name,p,q,scale=1.,outages=(),edges=None,fraction=1.,amp=1.,tap=1.04,linear=False,mode='coupled',plan=0,global_frac=1.,zmult=1.,partial=False,feeder_frac=1.,dispatch_ratio=1.):
    fc,normal,ties=feeder(name);candidate=normal.copy() if edges is None else edges.copy()
    # Plan bits: upstream corridor bundle, all feeder transformers, target feeder
    # reconductoring, tie-switch availability (handled by the candidate generator).
    if isinstance(plan,dict):
        tr_mva=plan['tr_mva'];candidate[:,2:4]*=plan['f_z'];candidate[:,4]*=plan['f_rating']
        transmission_upgrade=plan
    else:
        tr_mva=7.5 if plan&2 else 5.
        if plan&4:candidate[:,2:4]*=.65;candidate[:,4]*=1.5
        transmission_upgrade=bool(plan&1)
    p=np.asarray(p)*global_frac*feeder_frac;q=np.asarray(q)*global_frac*feeder_frac;scale*=global_frac
    if not main_connected(outages):return dict(safe=False,violation=3.,status='separation',iterations=0)
    va=vb=1.; ok=False
    for k in range(1 if linear or mode!='coupled' else 35):
        fa=radial(fc,candidate,p*fraction,q*fraction,va*tap,linear,amp,tr_mva,zmult,partial)
        ah=radial(fc,normal,p,q,va*tap,linear,amp,tr_mva,zmult)
        bh=radial(fc,normal,p*.95,q*.95,vb*tap,linear,amp,tr_mva,zmult)
        if not all(f['ok'] for f in [fa,ah,bh]):return dict(safe=False,violation=3.,status='distribution_nonconvergence',iterations=k+1)
        pd=case('case9')['bus'][:,2].copy()*scale;qd=case('case9')['bus'][:,3].copy()*scale
        nominal=fc['bus'][:,2:4].sum(axis=0);sa=17*ah['source']+fa['source'];sb=20*bh['source']
        pd[4]=max(0,90-18*nominal[0])*scale+sa.real;qd[4]=max(0,30-18*nominal[1])*scale+sa.imag
        pd[6]=max(0,100-19*nominal[0])*scale+sb.real;qd[6]=max(0,35-19*nominal[1])*scale+sb.imag
        main=main_ac(pd,qd,scale*dispatch_ratio,outages,transmission_upgrade)
        if not main['ok']:return dict(safe=False,violation=3.,status=main['reason'],iterations=k+1)
        na,nb=abs(main['v'][[4,6]]);err=max(abs(na-va),abs(nb-vb));va,vb=na,nb
        if linear or mode!='coupled' or err<1e-8:ok=True;break
    fs=[fa,ah,bh];vmin=min(f['vmin'] for f in fs);vmax=max(f['vmax'] for f in fs)
    dist=max(f['thermal'] for f in fs);tr=max(f['tr_ratio'] for f in fs)
    mvmin=float(min(abs(main['v'])));mvmax=float(max(abs(main['v'])))
    dv=max((.9-vmin)/.1,(vmax-1.1)/.1,dist-1,tr-1)
    tv=max((.9-mvmin)/.1,(mvmax-1.1)/.1,main['ratio']-1,main['genviol'])
    violation=dv if mode=='dist_only' else tv if mode=='trans_only' else max(dv,tv)
    if not ok:violation=3.
    return dict(safe=bool(ok and violation<=1e-7),violation=float(violation),status='converged' if ok else 'coupling_nonconvergence',iterations=k+1,vmin=vmin,vmax=vmax,dist_thermal=dist,main_thermal=main['ratio'],tr_ratio=tr,main_vmin=mvmin,boundary_a=float(va),boundary_b=float(vb),genviol=main['genviol'],residual=max(main['residual'],*[f['residual'] for f in fs]),loss_mw=fa['loss']+17*ah['loss']+20*bh['loss']+main['loss'],target_loss=fa['loss'],boundary_mismatch=err)

def sample(name,seed,shift=False):
    rng=np.random.default_rng(seed);fc,_,_=feeder(name);n=len(fc['bus'])
    scale=rng.uniform(.72,1.10 if not shift else 1.22)
    ld=rng.uniform(.55,1.15 if not shift else 1.35)
    common=rng.normal(); local=np.clip(np.exp(.13*(.65*common+np.sqrt(1-.65**2)*rng.normal(size=n))-.5*.13**2),.65,1.45)
    gross=fc['bus'][:,2]*ld*local
    pv=np.zeros(n);pv[7::4]=gross[7::4]*rng.uniform(0,.6 if not shift else .9)
    p=gross-pv;q=fc['bus'][:,3]*ld*local*rng.uniform(.9,1.1)
    return dict(name=name,p=p.tolist(),q=q.tolist(),scale=scale,amp=rng.uniform(.9,1.1),tap=rng.uniform(1.025,1.05),seed=seed)

def evaluate(raw,**kwargs):
    return coupled(raw['name'],raw['p'],raw['q'],raw['scale'],amp=raw['amp'],tap=raw['tap'],**kwargs)

def candidates(name,fault):
    fc,n,t=feeder(name)
    if fault<0:return [(-1,n)]
    return [(i,np.vstack([np.delete(n,fault,axis=0),tie])) for i,tie in enumerate(t) if topology_ok(np.vstack([np.delete(n,fault,axis=0),tie]),len(fc['bus']))]

def physics_features(raw,edges,outages=(),fraction=1.):
    lin=evaluate(raw,edges=edges,outages=outages,fraction=fraction,linear=True)
    fc,_,_=feeder(raw['name']);zb=fc['bus'][0,9]**2/fc['baseMVA']
    return dict(lin_violation=lin['violation'],lin_vmin=lin.get('vmin',.5),lin_dist=lin.get('dist_thermal',3.),lin_main=lin.get('main_thermal',3.),lin_tr=lin.get('tr_ratio',3.),boundary=lin.get('boundary_a',.5),P=sum(raw['p']),Q=sum(raw['q']),scale=raw['scale'],amp=raw['amp'],tap=raw['tap'],fraction=fraction,Rsum=float(sum(edges[:,2])*zb),Xsum=float(sum(edges[:,3])*zb),n_out=len(outages))

FEATURES=['lin_violation','lin_vmin','lin_dist','lin_main','lin_tr','boundary','P','Q','scale','amp','tap','fraction','Rsum','Xsum','n_out']
