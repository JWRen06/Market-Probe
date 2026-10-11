"""Independent cellwise BV formulation; no analytic extremizer used in the LP."""
from pathlib import Path
from fractions import Fraction as F
import sys,json
sys.path.insert(0,str(Path(__file__).parent/'_local_dependencies'))
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from episode_budget_quote_frontier import root

rows=[]
for b in [1.,1.1,1.2,1.3]:
    t,_,_=root(F(str(b)))
    exact=float((F(1,2)-F(str(b))*t)/(F(7,10)-F(6,5)*t+2*t*t))
    values=[]
    for n in [40,80,160,320,640]:
        # f[0:n], upward increments[0:n-1], and the response floor w.
        nv=2*n
        aub=lil_matrix((2*n,nv)); rhs=np.zeros(2*n)
        for i in range(n):
            aub[i,i]=-1; aub[i,-1]=.2+(i+1)/n
        for i in range(1,n):
            aub[n+i-1,i]=1; aub[n+i-1,i-1]=-1
            aub[n+i-1,n+i-1]=-1
        aub[-1,0]=1; aub[-1,n:2*n-1]=1;rhs[-1]=b
        eq=np.zeros((2,nv));eq[0,:n]=1/n
        eq[1,:n]=(np.arange(n)+.5)/(n*n)
        objective=np.zeros(nv);objective[-1]=-1
        sol=linprog(objective,A_ub=aub.tocsr(),b_ub=rhs,A_eq=eq,
                    b_eq=[.5,.25],bounds=[(0,1)]*n+[(0,None)]*n,method='highs')
        assert sol.success,sol.message
        f=sol.x[:n];w=sol.x[-1]
        budget=f[0]+np.maximum(np.diff(f),0).sum()
        error=max(float(np.max(aub@sol.x-rhs)),float(np.max(abs(eq@sol.x-[.5,.25]))))
        assert error<1e-7 and budget<=b+1e-7 and w<=exact+1e-7
        values.append(dict(cells=n,feasible_floor=float(w),continuous_gap=exact-w,
                           actual_BV_budget=float(budget),constraint_error=error))
    assert values[-1]['continuous_gap']<.002
    rows.append(dict(budget=b,analytic_floor=exact,grids=values))

# Certified critical structural budget at C=.11, kappa0: w=39/70.
target=F(39,70)
def equation(t):return 4*target*(F(13,30)-F(7,20)*t+t**3/3)-(1-t)
lo,hi=F(0),F(1,4)
for _ in range(180):
    mid=(lo+hi)/2
    if equation(mid)<0:lo=mid
    else:hi=mid
assert equation(lo)<=0<=equation(hi)
def budget(t):return (80*t**3-96*t*t+36*t+5)/(40*t**4-42*t*t+52*t)
# B(t) is decreasing on this interval; the proof gives dt/dB<0.
bl,bh=budget(hi),budget(lo)
assert 1<bl<bh<F(13,10) and bh-bl<F(1,10**50)
out=dict(status='passed',rows=rows,critical_budget=dict(lower=str(bl),upper=str(bh),
    midpoint=float((bl+bh)/2),width=str(bh-bl)),
    scope='Independent piecewise-constant primal feasibility; analytic continuum upper bound remains a separate proof. Expected interval budget, not reentry probability.')
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(dict(status='passed',finest=[r['grids'][-1] for r in rows],critical_budget=out['critical_budget']['midpoint'])))
