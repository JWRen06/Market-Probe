from pathlib import Path
import ast,json,sys
root=Path(__file__).resolve().parents[1]

import numpy as np
src=ast.parse((root/'experiments/second_paper_bounded_revision.py').read_text(encoding='utf-8-sig'));ns={'np':np}
for node in src.body:
 if isinstance(node,ast.FunctionDef):exec(compile(ast.Module(body=[node],type_ignores=[]),'<functions>','exec'),ns)
p=np.array([.4,.64,.816]);sigma=.05;alpha=.05;tau=.08;c=2/15;rows=[];viol=0
for Q in [10.,20.,50.]:
 rng=np.random.default_rng(2026101142)
 target=ns['exact'](p,c,Q)['capture_lower']
 for rep in range(80):
  total=np.zeros(3);prev=0;good=True;stopped=False
  for stage in range(1,24):
   N=32*2**(stage-1);batch=N-prev;total+=batch*p+sigma*np.sqrt(batch)*rng.normal(size=3);prev=N;y=total/N
   e=sigma*np.sqrt(2*np.log(6*stage*(stage+1)/alpha)/N);good &= bool(np.max(abs(y-p))<=e)
   result=ns['deployment_cap'](y,e,c,Q,tau)
   if result is not None:
    low,high,q,R,cells=result
    if good:
     rr,M,s,mg,MG,ell=ns['params'](p)
     actual=0 if q==0 else min(ns['curve'](q,rr,M,s),ns['curve'](q,0,MG,ell,mg))-c*q
     viol+=not(low-1e-9<=target<=high+1e-9 and actual>=low-1e-9)
    if high-low<=tau:
     rows.append(dict(Q=Q,rep=rep,total_records=3*N,guarantee_low=low,guarantee_high=high,
         exact_robust_value=target,action=q,search_cap=R,grid_cells=cells,coverage_event=good))
     stopped=True;break
  assert stopped
assert viol==0
out=dict(status='passed',on_event_violations=viol,paths=rows,
         summary=[dict(Q=Q,median_total_records=float(np.median([x['total_records'] for x in rows if x['Q']==Q])),
              median_action=float(np.median([x['action'] for x in rows if x['Q']==Q]))) for Q in [10.,20.,50.]],
         target='best robust deployment guarantee and a certified action, not the actual-world optimal profit',
         scope='Synthetic Gaussian cumulative statistics, same seeds for budgets; no paid-information net-value claim')
(root/'experiments/second_paper_bounded_deployment.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='paths'}))
