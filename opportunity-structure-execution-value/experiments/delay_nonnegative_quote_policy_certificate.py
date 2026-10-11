"""LP proposes a bounded rule; Fraction drawdown proves its continuous-world guarantee."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).parent/'_local_dependencies'))
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from fractions import Fraction as F
rows=[]
for n,k,C in [(80,0,F(8,100)),(80,0,F(11,100)),(160,0,F(118,1000)),(80,1,F(6,100)),(80,12,F(1,100))]:
 k=F(k);v=(F(1,2)-C)/(F(7,10)+k/2);beta=1-k*v
 pairs=[(i,j) for i in range(n) for j in range(i+1,n+1)];nv=len(pairs)
 rr=[];cc=[];vv=[]
 for col,(i,j) in enumerate(pairs):
  for q in range(i,j):rr.append(q);cc.append(col);vv.append(float(beta))
 for q in range(n):rr.append(q);cc.append(nv+q);vv.append(-1.)
 for col in range(nv):rr.append(n);cc.append(col);vv.append(1.)
 ub=coo_matrix((vv,(rr,cc)),shape=(n+1,nv+n)).tocsr()
 rhs=np.append(float(v)*(.2+(np.arange(n)+.5)/n),1.)
 eq=np.zeros((2,nv+n))
 for col,(i,j) in enumerate(pairs):eq[0,col]=(j-i)/n;eq[1,col]=(j*j-i*i)/(2*n*n)
 objective=np.append(np.zeros(nv),np.full(n,1/n))
 sol=linprog(objective,A_ub=ub,b_ub=rhs,A_eq=eq,b_eq=[.5,.25],bounds=(0,None),method='highs');assert sol.success
 # All rounded values below are exact rationals; validity does not rely on LP tolerances.
 p=[max(F(0),min(F(1),F(str(-n*z)))) for z in sol.ineqlin.marginals[:n]]
 eta_a,eta_m=map(lambda z:F(str(z)),sol.eqlin.marginals)
 K=F(0);maxK=F(0);drawdown=F(0);points=[]
 for i,pi in enumerate(p):
  lo,hi=F(i,n),F(i+1,n);A=beta*pi-eta_a
  locations=[hi]
  if eta_m and lo<A/eta_m<hi:locations.insert(0,A/eta_m)
  for t in locations:
   current=K+A*(t-lo)-eta_m*(t*t-lo*lo)/2
   drawdown=max(drawdown,maxK-current);maxK=max(maxK,current);points.append((t,current))
  K+=A*(hi-lo)-eta_m*(hi*hi-lo*lo)/2
 mass=sum(p,F(0))/n;assert mass>0
 duration_weight=sum((pi*(F(1,5)/n+F(2*i+1,2*n*n)) for i,pi in enumerate(p)),F(0))
 lower=eta_a/2+eta_m/4-drawdown-v*duration_weight
 gap=lower-C*max(p)
 assert gap>0
 gain=gap/(mass*(F(6,5)+k))
 gross_lower=(eta_a/2+eta_m/4-drawdown)/beta
 cash_lower=(gross_lower-C*max(p))/(duration_weight+k*gross_lower)
 assert cash_lower>v and gross_lower>C*max(p)
 # Certificate directly uses the worst interval drop, covering every A<=E continuously.
 rows.append(dict(cells=n,kappa=str(k),uniform_average_fee=str(C),baseline_cash=str(v),
   exact_surplus_margin=str(gap),cash_rate_gain_lower=str(gain),numerical_gain_lower=float(gain),
   cash_rate_lower=str(cash_lower),numerical_cash_rate_lower=float(cash_lower),
   waiting_density=[str(pi/mass) for pi in p],maximum_density=str(max(p)/mass),
   eta_total=str(eta_a),eta_time=str(eta_m),continuous_interval_penalty=str(drawdown),
   grid_objective=float(sol.fun),scope='Arbitrary nonnegative time quotes with integral C and all single-window worlds matching exact two moments. Density proposed by finite LP; Fraction antiderivative checks give continuum certificate. No statement of globally optimal value.'))
out=dict(status='passed',rows=rows,kind='continuous exact rational action certificates')
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(dict(status='passed',rows=[{key:r[key] for key in ['cells','kappa','uniform_average_fee','numerical_gain_lower','numerical_cash_rate_lower','maximum_density']} for r in rows])))
