from fractions import Fraction as F
from pathlib import Path
import json

C=F(2,25);tau=F(1,5);v0=F(3,5)
lower=(F(1,2)-2*C)/(tau+F(1,3))
assert lower==F(51,80)>v0
def primitive_f(t):
    return F(2,5)*t+F(3,10)*t*t-F(2,5)*max(t-F(1,2),F(0))
def primitive_tf(t):
    return t*t/5+t**3/5-F(1,5)*max(t*t-F(1,4),F(0))
assert primitive_f(1)==F(1,2) and primitive_tf(1)==F(1,4)
checks=0
for kap in map(F,[0,1,12,100]):
    threshold=F(5)/(52+30*kap)
    for fraction in map(F,[F(1,2),F(9,10),F(1),F(11,10)]):
        cost=threshold*fraction
        tri=(F(1,2)-2*cost)/(tau+F(1,3)+kap/2)
        uniform=(F(1,2)-cost)/(tau+F(1,2)+kap/2)
        assert (tri>uniform)==(cost<threshold)
        if cost==threshold:assert tri==uniform
        checks+=1
# Two-bin densities with arbitrary early mass; the common return world
# bounds all densities by its residual range, not just these examples.
for lam in map(F,[F(2,5),F(1),F(100)]):
    for i in range(101):
        early=F(i,100)
        residual=F(2,5)*(early-F(1,2))
        tv=abs(early-F(1,2))
        assert residual-lam*tv<=0
        checks+=1
out=dict(status='passed',exact_checks=checks,
    uniform_cash=str(v0),all_nonnegative_quote_triangle_lower=str(lower),
    direct_response_uniform_gate=str(F(2,5)),
    financial_world='probability 2/5 window [0,1/2); probability 3/5 uniform maturity and expiry 1',
    scope='Fixed mean nonnegative timing fees versus density response fee; strong permitted-response distinction, not a market empirical claim')
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(out)
