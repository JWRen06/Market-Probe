"""Synthetic common-deadline cash panel. Includes paid validation and funding stops."""
from pathlib import Path
import json,sys
import numpy as np
base=Path(__file__).parent
front=json.loads((base/'episode_budget_quote_frontier.json').read_text())['rows']
worlds={'flat':None,'central_window':None,'budget1':front[0],'budget1_3':front[-1]}
M=1011/1600
def success(name,d):
    if name=='flat':return np.full_like(d,.5)
    if name=='central_window':return ((d>=.25)&(d<.75)).astype(float)
    r=worlds[name];t=r['t'];w=r['floor'];b=r['budget']
    return w*(.2+2*t)+(b-1.2*w)*(d<t)+w*np.maximum(d-2*t,0)
def quote(kind,d):return np.full_like(d,.11) if kind=='flat' else (.11/.35)*(d<.35)
def sample(rule,u):
    if rule=='uniform':return u
    if rule=='triangular':return 1-np.sqrt(1-u)
    if rule=='immediate':return np.zeros_like(u)
    # Exact inverse CDF of unnormalised p: 1 then1.2-1.1t.
    y=u*M;ret=y.copy();mask=y>.35
    v=y[mask]-.35
    ret[mask]=.35+(.815-np.sqrt(.815**2-2*1.1*v))/1.1
    return ret
rows=[];seed=20261011;reps=256;N=512;research_fee=.001
for name in worlds:
 for kind in ['flat','early']:
  for kappa in [0,12]:
   v0=.39/(.7+.5*kappa);jmax=kappa+1.2
   grid=np.linspace(0,1,10001,endpoint=False)
   f=success(name,grid);q=quote(kind,grid)
   rates=(f-q)/(.2+grid+kappa*f)
   oracle=float(grid[np.argmax(rates)])
   for H in [200,2000]:
    for initial in [.35,500]:
     for method in ['immediate','uniform','triangular','structure_rule','paid_validation','full_information']:
      training_allowed=H>=N*jmax
      seed+=1;rng=np.random.default_rng(seed)
      cash=np.zeros(reps);time=np.zeros(reps);balance=np.full(reps,initial,dtype=float)
      stopped=np.zeros(reps,dtype=bool);done=np.zeros(reps,dtype=bool)
      records=np.zeros(reps,dtype=int);sums=np.zeros(reps)
      approved=np.zeros(reps,dtype=bool);classified=np.zeros(reps,dtype=bool)
      count=np.zeros(reps,dtype=int)
      while True:
       active=(~stopped)&(~done)&(time+jmax<=H)
       if not np.any(active):break
       ids=np.flatnonzero(active);u=rng.random(len(ids))
       if method=='full_information':d=np.full(len(ids),oracle)
       elif method=='paid_validation':
        train=(records[ids]<N)&training_allowed
        d=sample('structure',u)
        fallback=(~train)&(~approved[ids]);d[fallback]=u[fallback]
       else:d=sample({'structure_rule':'structure'}.get(method,method),u)
       fees=quote(kind,d)
       train=((records[ids]<N)&training_allowed) if method=='paid_validation' else np.zeros(len(ids),dtype=bool)
       expense=fees+research_fee*train
       funded=balance[ids]+1e-12>=expense
       stopped[ids[~funded]]=True
       ids=ids[funded];d=d[funded];fees=fees[funded];train=train[funded];expense=expense[funded]
       if not len(ids):continue
       z=(rng.random(len(ids))<success(name,d)).astype(float)
       j=.2+d+kappa*z;y=z-expense
       cash[ids]+=y;balance[ids]+=y;time[ids]+=j;count[ids]+=1
       assert np.min(balance[ids])>=-1e-9 and np.max(time[ids])<=H+1e-9
       if method=='paid_validation':
        ti=ids[train];records[ti]+=1
        sums[ti]+=(z-fees-v0*j)[train]
        test=ti[records[ti]==N]
        R=(1-kappa*v0)+v0+(.11 if kind=='flat' else .11/.35)
        approved[test]=sums[test]/N-R*np.sqrt(np.log(20)/(2*N))>0
        classified[test]=True
      rows.append(dict(world=name,quote=kind,kappa=kappa,deadline=H,initial_fee_balance=initial,method=method,
       mean_cash=float(np.mean(cash)),mean_cash_standard_error=float(np.std(cash,ddof=1)/np.sqrt(reps)),cash_quantiles_05_50_95=np.quantile(cash,[.05,.5,.95]).tolist(),
       mean_relative_to_continuous_uniform=float(np.mean(cash)-v0*H),
       positive_cash_fraction=float(np.mean(cash>0)),above_uniform_fraction=float(np.mean(cash>v0*H)),
       funding_stop_fraction=float(np.mean(stopped)),mean_completed_cycles=float(np.mean(count)),
       validation_completed_fraction=float(np.mean(classified)) if method=='paid_validation' else None,
       approval_fraction=float(np.mean(approved)) if method=='paid_validation' else None,
       refused_before_measurement_for_deadline=(not training_allowed) if method=='paid_validation' else None,
       structural_single_window_applicable=name!='budget1_3'))
out=dict(status='passed',replications_per_case=reps,cases=len(rows),rows=rows,
 scope='Synthetic independent legal opportunity draws; fixed common deadlines; fee balance excludes trading principal. Uniform benchmark has unlimited balance and is continuous rate v0H, not a matched small-balance strategy. Full information is a free ideal benchmark with numerical grid action. Paid validation uses512 actual completed candidate cycles at research fee.001; no statement of optimal learning or real-market performance.')
(base/'revision_unified_cash_panel.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(dict(status='passed',cases=len(rows),replications=reps,
 validation_summary=[r for r in rows if r['method']=='paid_validation' and r['initial_fee_balance']==500 and r['deadline']==2000 and r['world']=='budget1'])))
