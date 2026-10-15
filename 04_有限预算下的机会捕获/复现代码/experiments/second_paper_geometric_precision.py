from pathlib import Path
import ast,json,sys,itertools,time
root=Path(__file__).resolve().parents[1]

import numpy as np
from scipy.stats import chi2
src=ast.parse((root/'experiments/second_paper_bounded_revision.py').read_text(encoding='utf-8-sig'))
ns={'np':np}
for node in src.body:
 if isinstance(node,ast.FunctionDef):exec(compile(ast.Module(body=[node],type_ignores=[]),'<functions>', 'exec'),ns)
params,curve,peak,exact,old= [ns[k] for k in ['params','curve','peak','exact','refined_cap']]

def coefficients(q,x,left):
 q=np.atleast_1d(q)
 h=-np.expm1(q*np.log1p(-x));g=h/x;gp=(q*x*np.exp((q-1)*np.log1p(-x))-h)/x**2
 if left:
  gam=(x*gp-g+q)/x**2;beta=2*(g-q)/x-gp;alf=q
 else:
  gam=(1-g-gp*(1-x))/(1-x)**2;beta=gp-2*gam*x;alf=g-gp*x+gam*x*x
 return np.stack([alf+2*beta+3*gam,-beta-3*gam,gam],axis=1)

def spherical_cap(y,radius,c,Q,tau,grid_divisor=2):
 a=y[0];B=2*y[0]-y[1];D=3*y[0]-3*y[1]+y[2]
 if not(a>B>D>0 and a*D>B*B):return None
 rr,M,s,mg,MG,ell=params(y)
 qf,uf=peak(rr,M,s,c,Q);qg,ug=peak(0,MG,ell,c,Q,mg)
 star=qf if uf>=ug else qg
 left=(1<=star<=2) or star>=3
 b=coefficients(np.array([star]),s if left else ell,left)[0]
 u=b/np.linalg.norm(b) if np.linalg.norm(b)>1e-14 else np.array([1.,0,0])
 seed=np.eye(3)[int(np.argmin(abs(u)))];v=np.cross(u,seed);v/=np.linalg.norm(v)
 O=np.stack([u,v,np.cross(u,v)],axis=1)
 vertices=y+radius*(np.array(list(itertools.product([-1,1],repeat=3)))@O.T)
 for z in vertices:
  aa=z[0];bb=2*z[0]-z[1];dd=3*z[0]-3*z[1]+z[2]
  if not(aa>bb>dd>0 and aa*dd>bb*bb):return None
 lo=vertices.min(axis=0);hi=vertices.max(axis=0)
 al,ah=lo[0],hi[0];Bl=2*lo[0]-hi[1];Bh=2*hi[0]-lo[1];Dl=3*lo[0]-3*hi[1]+lo[2];Dh=3*hi[0]-3*lo[1]+hi[2]
 if min(al,Bl,Dl,al-Bh,Bl-Dh)<=0:return None
 sl=Dl/Bh;sh=Dh/Bl;el=(Bl-Dh)/(ah-Bl);eh=(Bh-Dl)/(al-Bh)
 if not 0<sl<=sh<1 or not 0<el<=eh<1:return None
 rh=ah-Bl*Bl/Dh;Mh=Bh**3/Dl**2;wh=(ah-Bl)/(1-eh);Gh=wh/el
 Lips=max(rh+Mh*(-np.log1p(-sh)),Gh*(-np.log1p(-eh)))+c
 cells=max(8,int(np.ceil(grid_divisor*Q*Lips/tau)));q=np.linspace(0,Q,cells+1)
 useF=((q>=1)&(q<=2))|(q>=3)
 b=np.where(useF[:,None],coefficients(q,s,True),coefficients(q,ell,False))
 upper=max(0,float(np.max(b@y+radius*np.linalg.norm(b,axis=1)-c*q)+Lips*Q/(2*cells)))
 low=[]
 for z in vertices:
  r0,M0,s0,mg0,MG0,e0=params(z)
  low.append(np.maximum(curve(q,r0,M0,s0),curve(q,0,MG0,e0,mg0))-c*q)
 lower=max(0,float(np.max(np.min(low,axis=0))))
 return lower,upper,cells

def run_pair(p,c,Q,tau,rep,seed):
 sigma=.05;alpha=.05;rng=np.random.default_rng(seed+rep);total=np.zeros(3);prev=0
 done={};good={'box':True,'sphere':True};cpu={'box':0.,'sphere':0.};target=exact(p,c,Q)['capture_upper']
 for j in range(1,27):
  N=32*2**(j-1);batch=N-prev;total+=batch*p+sigma*np.sqrt(batch)*rng.normal(size=3);prev=N;y=total/N
  e=sigma*np.sqrt(2*np.log(6*j*(j+1)/alpha)/N)
  rad=sigma*np.sqrt(chi2.isf(alpha/(j*(j+1)),3)/N)
  good['box'] &= bool(np.max(abs(y-p))<=e)
  good['sphere'] &= bool(np.linalg.norm(y-p)<=rad)
  for method in ['box','sphere']:
   if method in done:continue
   start=time.perf_counter();cap=old(y,e,c,Q,tau) if method=='box' else spherical_cap(y,rad,c,Q,tau)
   cpu[method]+=time.perf_counter()-start
   if cap is None:continue
   if good[method]:assert cap[0]-1e-9<=target<=cap[1]+1e-9,(method,p,Q,target,cap)
   if cap[1]-cap[0]<=tau:done[method]=dict(records=3*N,stage=j,width=cap[1]-cap[0],coverage_event=good[method],certificate_cpu_seconds=cpu[method],grid_cells=cap[2])
  if len(done)==2:break
 return dict(readouts=p.tolist(),c=c,Q=Q,tau=tau,rep=rep,target=target,methods=done,censored_methods=[x for x in ['box','sphere'] if x not in done])

if __name__=='__main__':
 p0=np.array([.4,.64,.816]);rows=[]
 for Q in [10.,20.]:
  for rep in range(40):rows.append(run_pair(p0,2/15,Q,.08,rep,2026101191))
 families=[('spread',np.array([.12,.32,.65]),np.array([1.,.6,.1])),
           ('concentrated',np.array([.18,.28,.38]),np.array([.6,.8,.6])),
           ('fast',np.array([.45,.65,.82]),np.array([.8,.5,.3]))]
 cases=[]
 for fi,(name,v,w) in enumerate(families):
  p=np.array([np.sum(w*(-np.expm1(k*np.log1p(-v)))) for k in [1,2,3]])
  for cost in [.6,1.4]:
   for Q in [10.,20.]:
    for tau in [.04,.08]:
     for rep in range(4):
      item=run_pair(p,cost*params(p)[0],Q,tau,rep,2026101200+fi*100)
      item['family']=name;item['cost_multiple']=cost;cases.append(item)
 groups=[]
 for Q in [10.,20.]:
  group=[x for x in rows if x['Q']==Q]
  groups.append(dict(Q=Q,**{m:dict(median_records=float(np.median([x['methods'][m]['records'] for x in group])),median_cpu_seconds=float(np.median([x['methods'][m]['certificate_cpu_seconds'] for x in group])),coverage_failures=sum(not x['methods'][m]['coverage_event'] for x in group)) for m in ['box','sphere']}))
 summaries=[]
 for name,_,_ in families:
  group=[x for x in cases if x['family']==name]
  ratios=[x['methods']['sphere']['records']/x['methods']['box']['records'] for x in group if len(x['methods'])==2]
  summaries.append(dict(family=name,pairs=len(group),censored=sum(bool(x['censored_methods']) for x in group),median_record_ratio_sphere_box=float(np.median(ratios)),best_ratio=min(ratios),worst_ratio=max(ratios)))
 out=dict(status='passed',baseline_groups=groups,other_family_comparisons=summaries,paired_paths=rows+cases,
   scope='Synthetic Gaussian cumulative logs; same path for two certificates, no empirical-market claim or optimality-ratio claim. Sphere and box use identical staged coverage allocation; data-selected rotation is protected by the full sphere.')
 (root/'experiments/second_paper_geometric_precision.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in out.items() if k!='paired_paths'}))
