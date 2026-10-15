from pathlib import Path
import sys,json,ast
root=Path(__file__).resolve().parents[1]

import numpy as np
from scipy.optimize import linprog

def rule(t):
 n=len(t)
 if n%2:
  d=(n-1)//2
  A=np.array([[t[i+j+1] for i in range(d)] for j in range(d)])
  inner=np.sort(np.polynomial.polynomial.polyroots(np.r_[np.linalg.solve(A,-t[d+1:2*d+1]),1.]).real)
  nodes=np.r_[0.,inner];H=np.array([[t[i+j] for j in range(d+1)] for i in range(d+1)])
 else:
  d=n//2;m=d-1
  if m:
   A=np.array([[t[i+j+1]-t[i+j+2] for i in range(m)] for j in range(m)])
   rhs=np.array([-(t[m+j+1]-t[m+j+2]) for j in range(m)])
   inner=np.sort(np.polynomial.polynomial.polyroots(np.r_[np.linalg.solve(A,rhs),1.]).real)
  else:inner=np.array([])
  nodes=np.r_[0.,inner,1.];H=np.array([[t[i+j]-t[i+j+1] for j in range(d)] for i in range(d)])
 weights=np.linalg.solve(np.vander(nodes,len(nodes),increasing=True).T,t[:len(nodes)])
 assert np.min(weights)>0
 return nodes,weights,1/np.linalg.inv(H)[0,0]

def fvalue(q,nodes,weights,a):
 ans=q*weights[0]
 for v,w in zip(nodes[1:],weights[1:]):
  ans+=w/v*(1. if v==1 else -np.expm1(q*np.log1p(-v)))
 return a*ans

rng=np.random.default_rng(2026101411);rows=[];largest_mix_error=0.;largest_moment_error=0.
for case in range(6):
 v=np.linspace(.05,.95,14);reward=rng.uniform(.3,1.,len(v));a=float(sum(reward*v));prob=reward*v/a
 for n in range(2,8):
  t=np.array([sum(prob*v**k) for k in range(n)]);nodes,weights,rho=rule(t)
  pn=np.array([sum(reward*(-np.expm1(k*np.log1p(-v)))) for k in range(1,n+1)])
  rn=a*rho;c=.6*rn;C=float(a*sum(weights[1:]/nodes[1:]))
  Q0=max(n+2.,2*pn[-1]/(rn-c))
  target_rows=[]
  for Q in [Q0,2*Q0,4*Q0]:
   F=fvalue(Q,nodes,weights,a);target=F-c*Q
   assert target>pn[-1] and rn>c
   D=a*Q-F;eps=1.1*.08/D
   assert 0<eps<1
   te=t*(1-eps);te[0]=1.;ne,we,re=rule(te)
   mixed=fvalue(Q,ne,we,a);expected=(1-eps)*F+eps*a*Q
   err=abs(mixed-expected)/max(1.,abs(expected));largest_mix_error=max(largest_mix_error,err);assert err<3e-6
   target_rows.append(dict(Q=Q,upper=target,D_Q=D,necessary_records=4*.05**2*.9**2*D*D/(.08**2*max(abs(a*np.arange(1,n+1)-pn))**2),alternative_upper=mixed-c*Q))
   assert mixed-c*Q>target+.08
  residual=max(abs(sum(weights*nodes**k)-t[k]) for k in range(n));largest_moment_error=max(largest_moment_error,residual)
  # A fixed finite reward witness limits the robust action at all deployment caps.
  witness_mass=float(sum(reward));witness_action_cutoff=witness_mass/c
  high_eta=.4*rn;high_cutoff=max(n,2*(C+1e-12)/high_eta)
  assert fvalue(high_cutoff,nodes,weights,a)-(rn+high_eta)*high_cutoff<1e-8
  Hcheck=np.array([[t[i+j] for j in range((n+1)//2)] for i in range((n+1)//2)]) if n%2 else np.array([[t[i+j]-t[i+j+1] for j in range(n//2)] for i in range(n//2)])
  rows.append(dict(case=case,n=n,matrix_condition=float(np.linalg.cond(Hcheck)),r_n=rn,c=c,finite_witness_mass=witness_mass,witness_action_cutoff=witness_action_cutoff,high_fee_tail_cutoff=high_cutoff,precision_cases=target_rows))

# An independently bounded reward LP cannot retain positive profits beyond M/c.
mass_rows=[];v=np.unique(np.r_[np.linspace(.0001,.9999,1000),.2,.6]);actual=np.zeros(len(v))
actual[np.argmin(abs(v-.2))]=1.;actual[np.argmin(abs(v-.6))]=.5
p=np.array([sum(actual*(-np.expm1(k*np.log1p(-v)))) for k in [1,2,3]])
eq=np.array([-np.expm1(k*np.log1p(-v)) for k in [1,2,3]])
M=2.;c=.1
for q in [M/c,2*M/c,5*M/c]:
 h=-np.expm1(q*np.log1p(-v));lp=linprog(-h,A_eq=eq,b_eq=p,A_ub=np.ones((1,len(v))),b_ub=[M],bounds=(0,None),method='highs')
 assert lp.success and -lp.fun-c*q<=1e-9
 mass_rows.append(dict(q=q,mass_bound=M,c=c,LP_upper_net=-lp.fun-c*q))

# Non-Gaussian, history-dependent noise with constant conditional means.
src=ast.parse((root/'experiments/second_paper_bounded_revision.py').read_text(encoding='utf-8-sig'));ns={'np':np}
for node in src.body:
 if isinstance(node,ast.FunctionDef):exec(compile(ast.Module(body=[node],type_ignores=[]),'<functions>','exec'),ns)
noise_rows=[];p=np.array([.4,.64,.816]);sigma=.05;alpha=.05;tau=.08;c=.6*ns['params'](p)[0];Q=10.
for rep in range(40):
 total=np.zeros(3);prev=0;previous_mean=p.copy();good=True
 for j in range(1,27):
  N=32*2**(j-1);batch=N-prev;scale=sigma*(.2+.8*(previous_mean>p))
  total+=batch*p+scale*(2*rng.binomial(batch,.5,size=3)-batch);prev=N;y=total/N;previous_mean=y
  e=sigma*np.sqrt(2*np.log(6*j*(j+1)/alpha)/N);good &= bool(max(abs(y-p))<=e)
  cap=ns['refined_cap'](y,e,c,Q,tau)
  if cap is None:continue
  target=ns['exact'](p,c,Q)['capture_upper']
  if good:assert cap[0]-1e-9<=target<=cap[1]+1e-9
  if cap[1]-cap[0]<=tau:
   noise_rows.append(dict(rep=rep,records=3*N,coverage_event=good,width=cap[1]-cap[0]));break
 else:raise AssertionError('bounded noise not stopped')
# Check the bounded Bernoulli experiment's KL inequality, separately from noise validation.
bernoulli_checks=[]
for n in range(2,8):
 # Fresh interior rewards are enough to check the exact scalar KL inequality.
 v=np.linspace(.05,.95,14);reward=np.linspace(.3,1.,14);a=float(sum(reward*v))
 p=np.array([sum(reward*(-np.expm1(k*np.log1p(-v)))) for k in range(1,n+1)])
 H=2*max(p);eps=.0001;pe=(1-eps)*p+eps*a*np.arange(1,n+1)
 x=p/H;y=pe/H
 assert np.all((y>0)&(y<1))
 kl=x*np.log(x/y)+(1-x)*np.log((1-x)/(1-y))
 bound=(p-pe)**2/(pe*(H-pe))
 assert np.all(kl<=bound+1e-14)
 bernoulli_checks.append(dict(n=n,max_KL=float(max(kl)),max_bound=float(max(bound))))
out=dict(status='passed',finite_scale_families=rows,matrix_rule_checks=len(rows),precision_comparisons=sum(len(x['precision_cases']) for x in rows),max_relative_mixture_rule_error=largest_mix_error,max_moment_residual=largest_moment_error,bounded_Bernoulli_KL_checks=bernoulli_checks,
 mass_bound_LP=mass_rows,non_Gaussian_paths=noise_rows,non_Gaussian_median_records=float(np.median([x['records'] for x in noise_rows])),non_Gaussian_coverage_failures=sum(not x['coverage_event'] for x in noise_rows),
 scope='General n and fee phase formula checks, independent bounded-mass LP, and synthetic history-dependent bounded-noise validation. High-order moment conditioning is reported; no market calibration.')
(root/'experiments/second_paper_general_task_separation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['finite_scale_families','non_Gaussian_paths']}))
