from pathlib import Path
import sys,json,math
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/_local_dependencies'))
import numpy as np
rng=np.random.default_rng(202610117)
N=4000;H=7000;ell=.02;delta=.05;Jmax=1.2
v0=F(4,7);vr=F(8,9);J0=F(7,10);Jr=F(9,20)
mu=F(2,5)-v0*Jr
L=2*(1+.2+float(v0)*(Jmax-.2))
e=L*math.sqrt(math.log(1/delta)/(2*N))
assert float(mu)>=2*e
lower=(1-delta)*float(mu)*(H/Jmax-N-1)-N*ell-float(v0)*Jmax
upper=float(vr)*Jmax-N*(float((vr-v0)*J0)+ell)
# Independent renewal simulations use actual completed durations and cash.
# A start is allowed only if the remaining horizon is at least Jmax.
def deployment(remaining,early):
 max_steps=int(remaining/.2)+1
 D=rng.random(max_steps)*(.5 if early else 1)
 Z=rng.binomial(1,.5,max_steps)
 J=.2+D
 starts=np.r_[0,np.cumsum(J[:-1])]
 k=int(np.count_nonzero(starts<=remaining-Jmax))
 assert k==0 or J[:k].sum()<=remaining+1e-8
 return float((Z[:k]-.1).sum())
rows=[]
for path in range(256):
 D=rng.random(N);Z=rng.binomial(1,.5,N);J=.2+D
 X=(2*(D<.5))*(Z-.1-float(v0)*J)
 approved=bool(X.mean()>e)
 train=float((Z-.1).sum()-N*ell)
 paid=train+deployment(H-float(J.sum()),approved)
 uniform=deployment(H,False)
 immediate=deployment(H,True)
 rows.append((paid-uniform,paid-immediate,approved))
a=np.array(rows,dtype=float)
mean=a[:,:2].mean(axis=0);se=a[:,:2].std(axis=0,ddof=1)/math.sqrt(len(a))
assert mean[0]+5*se[0]>=lower
assert mean[1]-5*se[1]<=upper
result={'seed':202610117,'model':'same nonnegative neutral quote model; flat f=.5, c=.1; early candidate r=2 on first half',
 'N':N,'H':H,'record_fee':ell,'delta':delta,'true_cycle_excess':float(mu),'confidence_radius':e,
 'ex_ante_margin_condition':float(mu)>=2*e,'lower_vs_uniform':lower,'upper_vs_immediate_candidate':upper,
 'independent_paths':len(a),'mean_vs_uniform':float(mean[0]),'se_vs_uniform':float(se[0]),
 'mean_vs_immediate_candidate':float(mean[1]),'se_vs_immediate_candidate':float(se[1]),
 'approval_fraction':float(a[:,2].mean()),'passes':True,
 'scope':'synthetic renewal validation; no real RFQ calibration and no claim of optimal information acquisition'}
(ROOT/'experiments/first_paper_third_review_bridge.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result))
