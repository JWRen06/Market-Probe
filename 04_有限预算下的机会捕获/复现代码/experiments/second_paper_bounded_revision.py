from pathlib import Path
import sys,json,math
root=Path(__file__).resolve().parents[1]

import numpy as np
from scipy.optimize import linprog
p=np.array([.4,.64,.816]);a=p[0];B=2*p[0]-p[1];D=3*p[0]-3*p[1]+p[2]
def params(p):
 a=p[0];B=2*p[0]-p[1];D=3*p[0]-3*p[1]+p[2]
 r=a-B*B/D;s=D/B;Mf=B**3/D**2
 ell=(B-D)/(a-B);wg=(a-B)/(1-ell);Mg=wg/ell;mg=a-wg
 return r,Mf,s,mg,Mg,ell
def curve(q,r,M,z,intercept=0):return intercept+r*q+M*(-np.expm1(q*np.log1p(-z)))
def peak(r,M,z,c,Q,intercept=0):
 zz=-np.log1p(-z);dc=c-r
 if dc<=0:q=Q
 elif M*zz<=dc:q=0.
 else:q=min(Q,max(0.,np.log(M*zz/dc)/zz))
 return q,float(curve(q,r,M,z,intercept)-c*q)
def exact(p,c,Q):
 r,M,s,mg,Mg,ell=params(p)
 qf,uf=peak(r,M,s,c,Q);qg,ug=peak(0,Mg,ell,c,Q,mg)
 cand=np.unique(np.clip([0,1,2,3,Q,qf,qg],0,Q))
 low=np.array([0 if q==0 else min(curve(q,r,M,s),curve(q,0,Mg,ell,mg))-c*q for q in cand]);i=int(np.argmax(low))
 d2=p[1]-p[0];d3=p[2]-p[1];k12=(p[1]-2*d3)/(p[0]-d3);k23=2+d3/d2
 knots=[(0,0),(0,2*p[0]-p[1]),(1,p[0]),(2,p[1]),(3,p[2]),(k12,p[0]*k12),(k23,p[2]),(Q,p[2]+(Q-3)*d3)]
 shapeU=max(y-c*q for q,y in knots);shapeV=max(0,max(p-np.arange(1,4)*c));shapeq=float(np.argmax(np.r_[0,p-np.arange(1,4)*c]))
 return dict(c=c,Q=Q,capture_upper=max(0,uf,ug),capture_lower=float(low[i]),capture_action=float(cand[i]),concave_upper=float(shapeU),concave_lower=float(shapeV),concave_action=shapeq,upper_reduction=float(shapeU-max(0,uf,ug)),guarantee_improvement=float(low[i]-shapeV))
rows=[exact(p,c,Q) for c in [.04,.1,2/15,.15,.2] for Q in [3,4,6,10,20]]
assert all(x['upper_reduction']>-1e-10 and x['guarantee_improvement']>-1e-10 for x in rows)
# Independent discrete moment LP at bounded scales, not only formula comparisons.
t=B/a;u=D/a;s=u/t;ell=(t-u)/(1-t)
v=np.unique(np.r_[np.linspace(1e-6,1-1e-6,3001),s,ell]);eq=np.array([np.ones_like(v),v,v*v]);mom=np.array([1,t,u]);lpchecks=0;err=0.
for q in [1.3,2.4,3.2,4.,6.,10.,20.]:
 g=-np.expm1(q*np.log1p(-v))/v
 pp=params(p);fv=curve(q,*pp[:3]);gv=curve(q,0,pp[4],pp[5],pp[3])
 for sign,target in [(1,min(fv,gv)),(-1,-max(fv,gv))]:
  z=linprog(sign*a*g,A_eq=eq,b_eq=mom,bounds=(0,None),method='highs');assert z.success
  gap=abs(z.fun-target);err=max(err,gap);assert gap<4e-5;lpchecks+=1
# Rigorous interval arithmetic for unknown moment directions.
def interval_cap(y,e,c,Q):
 lo=y-e;hi=y+e;al,ah=lo[0],hi[0]
 Bl=2*lo[0]-hi[1];Bh=2*hi[0]-lo[1];Dl=3*lo[0]-3*hi[1]+lo[2];Dh=3*hi[0]-3*lo[1]+hi[2]
 if min(al,Bl,Dl,al-Bh,Bl-Dh)<=0:return None
 sl=Dl/Bh;sh=Dh/Bl;el=(Bl-Dh)/(ah-Bl);eh=(Bh-Dl)/(al-Bh)
 if not 0<sl<=sh<1 or not 0<el<=eh<1:return None
 rl=max(0,al-Bh*Bh/Dl);rh=ah-Bl*Bl/Dh
 Ml=Bl**3/Dh**2;Mh=Bh**3/Dl**2
 wl=(al-Bh)/(1-el);wh=(ah-Bl)/(1-eh)
 ml=max(0,al-wh);mh=ah-wl;Gl=wl/eh;Gh=wh/el
 upper=max(0,peak(rh,Mh,sh,c,Q)[1],peak(0,Gh,eh,c,Q,mh)[1])
 lower=max(0,peak(rl,Ml,sl,c,Q)[1],peak(0,Gl,el,c,Q,ml)[1])
 return lower,upper

def refined_cap(y,e,c,Q,tau):
 import itertools
 lo=y-e;hi=y+e;vertices=np.array(list(itertools.product(*zip(lo,hi))))
 for z in vertices:
  aa=z[0];bb=2*z[0]-z[1];dd=3*z[0]-3*z[1]+z[2]
  if not (aa>bb>dd>0 and aa*dd>bb*bb):return None
 # Interval slope bound for the two sharp upper curves; valid for every true point in the box.
 al,ah=lo[0],hi[0];Bl=2*lo[0]-hi[1];Bh=2*hi[0]-lo[1];Dl=3*lo[0]-3*hi[1]+lo[2];Dh=3*hi[0]-3*lo[1]+hi[2]
 if min(al,Bl,Dl,al-Bh,Bl-Dh)<=0:return None
 sl=Dl/Bh;sh=Dh/Bl;el=(Bl-Dh)/(ah-Bl);eh=(Bh-Dl)/(al-Bh)
 if not 0<sl<=sh<1 or not 0<el<=eh<1:return None
 rh=ah-Bl*Bl/Dh;Mh=Bh**3/Dl**2;wh=(ah-Bl)/(1-eh);Gh=wh/el
 Lips=max(rh+Mh*(-np.log1p(-sh)),Gh*(-np.log1p(-eh)))+c
 cells=max(8,int(np.ceil(2*Q*Lips/tau)));q=np.linspace(0,Q,cells+1)
 rr,MF,ss,mg,MG,ell=params(y)
 useF=((q>=1)&(q<=2))|(q>=3)
 def coeff(x,left):
  H=-np.expm1(q*np.log1p(-x));gp=(q*x*np.exp((q-1)*np.log1p(-x))-H)/(x*x);gv=H/x
  if left:
   gam=(x*gp-gv+q)/(x*x);beta=2*(gv-q)/x-gp;alf=q
  else:
   gam=(1-gv-gp*(1-x))/(1-x)**2;beta=gp-2*gam*x;alf=gv-gp*x+gam*x*x
  return np.stack([alf+2*beta+3*gam,-beta-3*gam,gam],axis=1)
 bf=coeff(ss,True);bg=coeff(ell,False);coef=np.where(useF[:,None],bf,bg)
 upper=float(np.max(coef@y+e*np.abs(coef).sum(axis=1)-c*q)+Lips*Q/(2*cells))
 low=[]
 for z in vertices:
  r0,M0,s0,mg0,MG0,e0=params(z)
  f=curve(q,r0,M0,s0);g=curve(q,0,MG0,e0,mg0)
  low.append(np.maximum(f,g)-c*q)
 lower=max(0,float(np.max(np.min(low,axis=0))))
 return lower,upper,cells
def deployment_cap(y,e,c,Q,tau):
 import itertools
 lo=y-e;hi=y+e;vertices=np.array(list(itertools.product(*zip(lo,hi))))
 for z in vertices:
  aa=z[0];bb=2*z[0]-z[1];dd=3*z[0]-3*z[1]+z[2]
  if not (aa>bb>dd>0 and aa*dd>bb*bb):return None
 al,ah=lo[0],hi[0];Bl=2*lo[0]-hi[1];Bh=2*hi[0]-lo[1];Dl=3*lo[0]-3*hi[1]+lo[2];Dh=3*hi[0]-3*lo[1]+hi[2]
 if min(al,Bl,Dl,al-Bh,Bl-Dh)<=0:return None
 sl=Dl/Bh;sh=Dh/Bl;el=(Bl-Dh)/(ah-Bl);eh=(Bh-Dl)/(al-Bh)
 if not 0<sl<=sh<1 or not 0<el<=eh<1:return None
 rh=ah-Bl*Bl/Dh;Mh=Bh**3/Dl**2;wh=(ah-Bl)/(1-eh);wl=(al-Bh)/(1-el);Gh=wh/el
 T=ah-wl+Gh;R=min(Q,T/c) if c>0 else Q
 Lips=max(rh+Mh*(-np.log1p(-sh)),Gh*(-np.log1p(-eh)))+c
 cells=max(8,int(np.ceil(2*R*Lips/tau)));q=np.linspace(0,R,cells+1)
 rr,MF,ss,mg,MG,ell=params(y)
 useF=~(((q>=1)&(q<=2))|(q>=3))
 def coeff(x,left):
  H=-np.expm1(q*np.log1p(-x));gp=(q*x*np.exp((q-1)*np.log1p(-x))-H)/(x*x);gv=H/x
  if left:
   gam=(x*gp-gv+q)/(x*x);beta=2*(gv-q)/x-gp;alf=q
  else:
   gam=(1-gv-gp*(1-x))/(1-x)**2;beta=gp-2*gam*x;alf=gv-gp*x+gam*x*x
  return np.stack([alf+2*beta+3*gam,-beta-3*gam,gam],axis=1)
 bf=coeff(ss,True);bg=coeff(ell,False);coef=np.where(useF[:,None],bf,bg)
 coef[0]=0 # Actual non-execution, not a fast opportunity right limit.
 cash=coef@y-e*np.abs(coef).sum(axis=1)-c*q;i=int(np.argmax(cash));lower=max(0,float(cash[i]));action=float(q[i])
 high=[]
 for z in vertices:
  r0,M0,s0,mg0,MG0,e0=params(z)
  f=curve(q,r0,M0,s0);g=curve(q,0,MG0,e0,mg0)
  high.append(np.minimum(f,g)-c*q)
 upper=max(0,float(np.max(high))+Lips*R/(2*cells))
 return lower,upper,action,R,cells

rng=np.random.default_rng(2026101112);sigma=.05;alpha=.05;tau=.08;seq=[];viol=0;eventfail=0
for Q in [4.,10.,20.]:
 target=exact(p,2/15,Q)['capture_upper']
 for rep in range(80):
  total=np.zeros(3);prev=0;allgood=True;stop=None
  for stage in range(1,24):
   N=32*2**(stage-1);batch=N-prev;total+=batch*p+sigma*np.sqrt(batch)*rng.normal(size=3);prev=N;y=total/N
   e=sigma*np.sqrt(2*np.log(6*stage*(stage+1)/alpha)/N)
   allgood &= bool(np.max(abs(y-p))<=e)
   cap=refined_cap(y,e,2/15,Q,tau)
   if cap is not None:
    if allgood and not cap[0]-1e-9<=target<=cap[1]+1e-9:viol+=1
    if cap[1]-cap[0]<=tau:
     stop=dict(Q=Q,rep=rep,records_per_arm=N,total_records=3*N,lower=cap[0],upper=cap[1],target=target,grid_cells=cap[2],coverage_event=allgood);break
  assert stop is not None
  seq.append(stop);eventfail+=not allgood
assert viol==0
# Finite-Q lower bound: testable peak condition and D_Q coefficients.
rr,MM,ss,mg,GM,el=params(p);lower_bounds=[]
for Q in [4.,10.,20.,50.]:
 c=rr;z=exact(p,c,Q);tail=curve(Q,rr,MM,ss)-c*Q
 assert abs(tail-z['capture_upper'])<1e-10
 DQ=a*Q-curve(Q,rr,MM,ss);C0=max(abs(a*np.arange(1,4)-p));tau0=.05
 Nbound=4*sigma*sigma*(1-.05-.05)**2*DQ**2/(tau0*tau0*C0*C0)
 eps=1.1*tau0/DQ;new=(1-eps)*p+eps*a*np.arange(1,4)
 assert exact(new,c,Q)['capture_upper']>z['capture_upper']+tau0
 lower_bounds.append(dict(Q=Q,D_Q=DQ,epsilon=eps,necessary_total_record_budget=Nbound))
out=dict(status='passed',readouts=p.tolist(),bounded_rows=rows,independent_LP_checks=lpchecks,max_LP_error=err,unknown_direction_sequential=seq,coverage_event_failures=eventfail,on_event_interval_violations=viol,finite_cap_lower_bound=lower_bounds,scope='exact bounded-Q formulas and Gaussian interval-based unknown-direction procedure; synthetic, not optimal allocation constants or actual market data')
(root/'experiments/second_paper_bounded_revision.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=out['status'],LP_checks=lpchecks,max_LP_error=err,on_event_violations=viol,example=exact(p,.1,10),sequential_records=[dict(Q=Q,median=float(np.median([x['total_records'] for x in seq if x['Q']==Q]))) for Q in [4.,10.,20.]])))

