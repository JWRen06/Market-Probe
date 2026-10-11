"""Exact reward/time accounting and price certificate for the explicit purchase task."""
from fractions import Fraction as F
from math import log,ceil,exp
from pathlib import Path
import json
pi=F(1,10);eps=F(1,200);eta=F(1,5);v=F(1,4)
tau=F(1,5);jmax=F(11,5);lam=7*v/9
gamma=v/81;delta=(2*eta*(1-v)-v)/81
N=ceil(288*log(1/float(eps))/float(eta**2));assert N==38148
muA=tau+F(65,162)+F(1,2);muB=muA-2*eta/81
assert gamma==F(1,324) and delta==F(1,1620)
# Independent analytic integration of h, h^2, and r_e*h over rational pieces.
def h0(t):return t-F(1,6)
def h1(t):return t-F(5,6)
def integrate_linear(lo,hi,offset):return (hi*hi-lo*lo)/2-offset*(hi-lo)
def integrate_square(lo,hi,offset):return (hi**3-lo**3)/3-offset*(hi*hi-lo*lo)+offset**2*(hi-lo)
assert integrate_linear(F(0),F(1,2),F(1,6))+integrate_linear(F(1,2),F(1),F(5,6))==0
hh=integrate_square(F(0),F(1,2),F(1,6))+integrate_square(F(1,2),F(1),F(5,6))
assert hh==F(1,36)
rh=2*integrate_linear(F(0),F(1,9),F(1,6))+integrate_linear(F(1,9),F(1,2),F(1,6))+integrate_linear(F(1,2),F(8,9),F(5,6))
assert rh==-F(2,81)
meanA=F(1,2)-tau-lam/9-v*muA
meanB=F(1,2)+eta*rh-tau-lam/9-v*muB
assert meanA==gamma and meanB==-delta
A=pi*(1-eps)*gamma/jmax-(1-pi)*eps*delta/tau
D=pi*(1-eps)*gamma*(N+1)+v*jmax
assert A>0
H=1000000;c=F(1,1000)
lower=A*H-D-c*N;cap=(A*H-D)/N
assert lower>0 and cap>c and H>=(N+1)*jmax
no_info_slope=pi*gamma/muA-(1-pi)*delta/muB
no_info_upper=no_info_slope*H+(1-pi)*delta*jmax/muB
assert no_info_slope<0 and no_info_upper<-v*jmax
rows=[]
for horizon in [100000,200000,400000,1000000]:
 rows.append(dict(deadline=horizon,training_guaranteed=horizon>=(N+1)*jmax,
  acquisition_lower=float(A*horizon-D-c*N),sufficient_per_record_price=float((A*horizon-D)/N)))
result=dict(status='passed',records=N,classification_error_upper=exp(-N*float(eta**2)/288),
 prior_A=float(pi),net_purchase_lower=float(lower),per_record_price_cap=float(cap),
 required_deadline=ceil(float((D+c*N)/A)),no_information_static_candidate_upper=float(no_info_upper),
 coefficients_exact={'A':str(A),'D':str(D),'gamma':str(gamma),'delta':str(delta)},rows=rows,
 scope='Prior expected net cash vs actual funded uniform baseline; explicit two-action paid-record task, no tail top-up, all cycles settled, not optimal among free adaptive learning protocols')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
