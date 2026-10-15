from pathlib import Path
import sys,json,time
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'experiments'))
from second_paper_geometric_precision import np,chi2,spherical_cap,coefficients,params,exact

def run(p,Q,c,tau,rep):
 sigma=.05;alpha=.05;rng=np.random.default_rng(2026101311+rep)
 pilotN=2048;pilot=p+sigma/np.sqrt(pilotN)*rng.normal(size=3)
 a=pilot[0];B=2*pilot[0]-pilot[1];D=3*pilot[0]-3*pilot[1]+pilot[2]
 if a>B>D>0 and a*D>B*B:
  r,M,s,mg,MG,ell=params(pilot);q=np.linspace(0,Q,41);left=((q>=1)&(q<=2))|(q>=3)
  b=np.where(left[:,None],coefficients(q,s,True),coefficients(q,ell,False))
  K=float(np.max(np.linalg.norm(b,axis=1)))
  N0=max(32,int(np.ceil((2*sigma*K*np.sqrt(chi2.isf(alpha/2,3))/(.94*tau))**2)))
 else:N0=32
 total=np.zeros(3);prev=0;good=True;cpu=0.;target=exact(p,c,Q)['capture_upper']
 for j in range(1,20):
  N=N0*2**(j-1);batch=N-prev;total+=batch*p+sigma*np.sqrt(batch)*rng.normal(size=3);prev=N;y=total/N
  radius=sigma*np.sqrt(chi2.isf(alpha/(j*(j+1)),3)/N)
  good &= bool(np.linalg.norm(y-p)<=radius)
  start=time.perf_counter();cap=spherical_cap(y,radius,c,Q,tau,grid_divisor=32);cpu+=time.perf_counter()-start
  if cap is not None:
   if good:assert cap[0]-1e-9<=target<=cap[1]+1e-9
   if cap[1]-cap[0]<=tau:
    return dict(Q=Q,rep=rep,total_records=3*(pilotN+N),pilot_records=3*pilotN,validation_records=3*N,N0=N0,stages=j,width=cap[1]-cap[0],coverage_event=good,certificate_cpu_seconds=cpu)
 raise AssertionError('not stopped')
rows=[run(np.array([.4,.64,.816]),Q,2/15,.08,rep) for Q in [10.,20.] for rep in range(40)]
out=dict(status='passed',paths=rows,groups=[dict(Q=Q,median_total_records=float(np.median([x['total_records'] for x in rows if x['Q']==Q])),median_validation_stages=float(np.median([x['stages'] for x in rows if x['Q']==Q])),median_cpu_seconds=float(np.median([x['certificate_cpu_seconds'] for x in rows if x['Q']==Q])),coverage_failures=sum(not x['coverage_event'] for x in rows if x['Q']==Q)) for Q in [10.,20.]],
 scope='Pilot is charged in total records. Fresh validation is conditionally Gaussian given pilot; verification, not pilot prediction, determines stopping. Finer grid charged in CPU; no optimal-allocation claim.')
(root/'experiments/second_paper_pilot_sphere.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='paths'}))
