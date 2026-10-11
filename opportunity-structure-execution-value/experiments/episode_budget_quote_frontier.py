"""Exact window certificates plus independently evaluated budget frontier."""
from fractions import Fraction as F
from pathlib import Path
import json

def polynomial(b,t):
    return 40*b*t**4-80*t**3+(96-42*b)*t*t+(52*b-36)*t-5

def root(b):
    lo,hi=F(0),F(1,2)
    for _ in range(220):
        mid=(lo+hi)/2
        if polynomial(b,mid)<0:lo=mid
        else:hi=mid
    return (lo+hi)/2,lo,hi

rows=[]
checks=0
for b in (F(1),F(101,100),F(11,10),F(6,5),F(13,10)):
    t,lo,hi=root(b)
    assert polynomial(b,lo)<=0<=polynomial(b,hi)
    w=(F(1,2)-b*t)/(F(7,10)-F(6,5)*t+2*t*t)
    s=2*t;d=b-F(6,5)*w;p0=w*(F(1,5)+s)
    assert d>0 and p0>0
    early=p0+d;end=F(6,5)*w
    assert max(early,end)<1
    assert d+p0+w*(1-s)==b
    a=b*t+w*(F(7,10)-F(6,5)*t+2*t*t)
    m=b*t*t/2+w*(F(13,30)-F(3,5)*t*t+F(4,3)*t**3)
    assert a==F(1,2)
    assert abs(m-F(1,4))<F(1,10**60)
    lam=t*t/2
    def primitive(x):
        y=min(x,s)
        return y*y/2-t*y
    for i in range(31):
        A=F(i,30)
        for j in range(i,31):
            E=F(j,30)
            terminal=(E==1 and A<1)
            residual=primitive(E)-primitive(A)-lam*terminal+lam
            assert residual>=0
            checks+=1
    K=F(13,30)-F(7,10)*t+F(3,5)*t*t-F(2,3)*t**3
    upper=(F(1,4)-t/2+lam*b)/K
    assert abs(upper-w)<F(1,10**60)
    # Exact layer accounting: constant, early-only, late-only and reentry paths.
    both=max(F(0),min(early,end)-p0)
    only_early=max(F(0),early-end)
    only_late=max(F(0),end-early)
    total=p0+2*both+only_early+only_late
    assert total==b
    critical=[float((F(1,2)-F(7,10)*w)/(1+k*w)) for k in (0,1,12)]
    rows.append(dict(budget=float(b),t=float(t),floor=float(w),
                     maximum_success_probability=float(max(early,end)),
                     reentry_probability_in_layer_realization=float(both),
                     critical_cost_kappa0_1_12=critical))

# Same binwise zeroth/first moments, unbounded necessary interval budget:
# f=.5+.25*cos(2*pi*n*K*t) has zero cosine integrals and first moments per bin.
hidden_oscillation=[dict(bins=8,cycles_per_bin=n,a=.5,m=.25,
                         minimum_interval_budget=.75+.5*n*8)
                    for n in (1,10,100)]
result=dict(status='passed',exact_window_checks=checks,rows=rows,
            root_bracket_width=str(F(1,2**221)),
            moment_residual_upper='1e-60',hidden_oscillation=hidden_oscillation,
            scope='Budget1..1.3; finite-episode legal realizations. Analytic proof covers continuum; grid only implementation. No universal market claim.')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
