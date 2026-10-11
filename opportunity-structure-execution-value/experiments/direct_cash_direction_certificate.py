"""Exact integral identities and Hoeffding certificate/power budgets."""
from fractions import Fraction as F
from math import log,ceil,sqrt
from pathlib import Path
import json
W=F(83651,240000);alpha=F(6,5)-W/F(7,10);beta=F(11,10)
s=alpha/beta;C=F(11,100)
threshold=F(1,55)+C*(1-W/F(7,10))
mean=alpha/2-beta/4;d=mean-threshold
assert d==F(161907,61600000)
pos=alpha*alpha/(2*beta);neg=(beta-alpha)**2/(2*beta)
u=alpha/2;v=(beta-alpha)/2
assert pos==u*s and neg==v*(1-s) and u+v==beta/2
assert (s*s+(1-s)*(s+2))/3==(2-s)/3
rows=[]
for name,R in [('uniform',float(beta)),('cash_direction',float(beta/2)),('joint_K_max2',float(beta/2+F(2,55)))]:
    precision_N=ceil(R*R*log(20)/(2*float(d)**2))+1
    power_N=ceil(2*R*R*log(20)/float(d)**2)+1
    assert R*sqrt(log(20)/(2*precision_N))<float(d)
    assert 2*R*sqrt(log(20)/(2*power_N))<float(d)
    rows.append(dict(method=name,range=R,precision_only_records=precision_N,
                     records_for_validity_and_95pct_positive_certificate=power_N))
# Continuous densities: integrate exact monomial success curves on each side.
checks=0
for j in range(8):
    # f=t^j in [0,1]; rho*X equals g*f on both sides pointwise.
    true=alpha/F(j+1)-beta/F(j+2)
    plus=(2/alpha)*(alpha*s**(j+1)/F(j+1)-beta*s**(j+2)/F(j+2))
    minus=(2/(beta-alpha))*(beta*(1-s**(j+2))/F(j+2)-alpha*(1-s**(j+1))/F(j+1))
    assert u*plus-v*minus==true
    checks+=1
out=dict(status='passed',cash_margin=str(d),mean_probe_wait=str((2-s)/3),
         exact_unbiased_integral_checks=checks,rows=rows,
         fixed_rule_budget_limit=float(55*(mean-C*(1-W/F(7,10)))),
         scope='Confidence and certificate power for the fixed candidate with trusted B; no claim of profitable data collection or minimal actual variance. Joint K requires additional complete opportunity observation.')
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
