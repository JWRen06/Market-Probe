"""Exact moments and common-deadline expected-cash sufficient plans."""
from fractions import Fraction as F
from pathlib import Path
from math import ceil,log,sqrt
import json
eta=F(1,5);eps=F(1,20)
N=ceil(288*log(20)/float(eta**2))
parts=[(F(0),F(1,9),F(2)),(F(1,9),F(8,9),F(1)),(F(8,9),F(1),F(0))]
ed=sum(r*(b*b-a*a)/2 for a,b,r in parts)
ed2=sum(r*(b**3-a**3)/3 for a,b,r in parts)
assert ed==F(65,162)
def Hprimitive(t,side):return t*t/2-t/2-side*t/3
rh=F(0)
for a,b,r in parts:
    for l,u,s in [(a,min(b,F(1,2)),-1),(max(a,F(1,2)),b,1)]:
        if u>l:rh+=r*(Hprimitive(u,s)-Hprimitive(l,s))
assert rh==F(-2,81)
rows=[]
for k in [1,2,4,8,16,32,64,128,256,512,1024,2048]:
    v=F(3,10)/(F(7,10)+F(k,2));alpha=1-k*v;g=v/81
    jb=F(1,5)+ed+k*(F(1,2)+eta*rh)
    sa=g;sb=g+alpha*eta*rh
    assert sb<0 and sa>0
    jmax=F(k)+F(6,5)
    for c in [F(0),F(1,100)]:
        bound=jmax*(N+1+(c*N+v*jmax)/((1-eps)*g))
        H=ceil(bound)+1
        cash=(1-eps)*g*(F(H)/jmax-N-1)-c*N-v*jmax
        assert cash>0
        variance=alpha*alpha/4+v*v*(ed2-ed*ed)
        n75=ceil(.6744897501960817**2*float(variance/g**2))
        rows.append(dict(kappa=k,research_fee=float(c),records=N,
             expected_training_time=float(N*(F(7,10)+F(k,2))),
             deadline=H,deadline_over_kappa_squared=H/k**2,
             lower_expected_relative_cash=float(cash),A_rate_gain=float(g/(F(1,5)+ed+F(k,2))),
             B_menu_rate_loss=float(-sb/jb),per_cycle_variance=float(variance),
             normal_approx_75pct_cycles=n75,
             warning='Normal cycle count is asymptotic illustration, not a finite probability certificate.'))
out=dict(status='passed',eta=float(eta),records=N,rule_mean_wait=str(ed),rule_h_integral=str(rh),
    rows=rows,scope='Fixed A/B success curves; kappa-dependent adjustment price. Two-rule menu; completed common-deadline expected guarantee, not adaptive high-probability necessity.')
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(dict(status='passed',records=N,last_rows=rows[-2:])))
