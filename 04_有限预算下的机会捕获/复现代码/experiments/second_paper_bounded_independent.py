from pathlib import Path
import sys, json, ast, itertools
root=Path(__file__).resolve().parents[1]

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import brentq
src=ast.parse((root/'experiments/second_paper_bounded_revision.py').read_text(encoding='utf-8-sig'))
ns={'np':np}
for node in src.body:
    if isinstance(node,ast.FunctionDef):exec(compile(ast.Module(body=[node],type_ignores=[]),'<functions>','exec'),ns)
params,curve,exact,refined=[ns[n] for n in ['params','curve','exact','refined_cap']]
p=np.array([.4,.64,.816]);rng=np.random.default_rng(2026101131)
checks=[];maxerr=0
def concave_lp(pp,q,Q):
    xs=np.unique([0.,1.,2.,3.,q,Q]);n=len(xs)
    eq=[];rhs=[]
    for x,y in zip([0,1,2,3],[0,*pp]):
        v=np.zeros(n);v[np.where(xs==x)[0][0]]=1;eq.append(v);rhs.append(y)
    ub=[]
    for j in range(n-1):
        v=np.zeros(n);v[j]=1;v[j+1]=-1;ub.append(v)
    for j in range(n-2):
        h0=xs[j+1]-xs[j];h1=xs[j+2]-xs[j+1]
        v=np.zeros(n);v[j]=1/h0;v[j+1]=-1/h0-1/h1;v[j+2]=1/h1;ub.append(v)
    obj=np.zeros(n);obj[np.where(xs==q)[0][0]]=-1
    z=linprog(obj,A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=np.zeros(len(ub)),bounds=(0,None),method='highs')
    assert z.success,z.message
    return -z.fun
def concave_upper(pp,q):
    d2=pp[1]-pp[0];d3=pp[2]-pp[1]
    if q<1:return pp[0]-d2*(1-q)
    if q<=2:return min(pp[0]*q,pp[1]-d3*(2-q))
    if q<=3:return min(pp[1]+d2*(q-2),pp[2])
    return pp[2]+d3*(q-3)
worlds=[p]
for _ in range(20):
    v=rng.uniform(.03,.97,5);reward=rng.uniform(.1,1,5)
    worlds.append(np.array([np.sum(reward*(-np.expm1(k*np.log1p(-v)))) for k in [1,2,3]]))
for pp in worlds:
    for q in [.001,.1,.7,1,1.13,1.45,1.9,2,2.1,2.5,2.9,3,4,10,20]:
        error=abs(concave_lp(pp,q,20)-concave_upper(pp,q));maxerr=max(maxerr,error)
        assert error<1e-7
        checks.append(error)
# Independently check all q upper interpolation directions against random kernels.
direction_checks=0;max_kernel_violation=0
lower_checks=0;max_lower_violation=0
for pp in worlds:
    r,M,s,mg,MG,ell=params(pp)
    for q in [.0001,.2,.9,1.,1.3,1.99,2.01,2.6,2.99,3.01,6.,20.,100.]:
        left=(1<=q<=2) or q>=3;x=s if left else ell
        h=-np.expm1(q*np.log1p(-x));gv=h/x;gp=(q*x*np.exp((q-1)*np.log1p(-x))-h)/x**2
        if left:gam=(x*gp-gv+q)/x**2;beta=2*(gv-q)/x-gp;alf=q
        else:gam=(1-gv-gp*(1-x))/(1-x)**2;beta=gp-2*gam*x;alf=gv-gp*x+gam*x*x
        vv=np.r_[np.geomspace(1e-8,.01,80),np.linspace(.01,.999999,1500),x]
        kernel=-np.expm1(q*np.log1p(-vv))/vv
        err=float(np.max(kernel-(alf+beta*vv+gam*vv*vv)))
        max_kernel_violation=max(max_kernel_violation,err);assert err<2e-9
        direction_checks+=1
        x=ell if left else s
        h=-np.expm1(q*np.log1p(-x));gv=h/x;gp=(q*x*np.exp((q-1)*np.log1p(-x))-h)/x**2
        if not left:gam=(x*gp-gv+q)/x**2;beta=2*(gv-q)/x-gp;alf=q
        else:gam=(1-gv-gp*(1-x))/(1-x)**2;beta=gp-2*gam*x;alf=gv-gp*x+gam*x*x
        err=float(np.max(alf+beta*vv+gam*vv*vv-kernel))
        max_lower_violation=max(max_lower_violation,err);assert err<2e-9
        lower_checks+=1
# Check the interval for exact means, and for different true means inside each box.
boxchecks=0;boxviol=0
for Q in [4,10,20]:
    for e in [1e-5,3e-5,1e-4]:
        z=refined(p,e,2/15,Q,.08);assert z is not None
        for _ in range(80):
            truep=p+rng.uniform(-e,e,3);value=exact(truep,2/15,Q)['capture_upper']
            boxviol+=not (z[0]-1e-9<=value<=z[1]+1e-9);boxchecks+=1
assert boxviol==0
precision=[]
def upper_coeff(pp,q):
    rr,M,x,mg,MG,ell=params(pp)
    h=-np.expm1(q*np.log1p(-x));g=h/x;gp=(q*x*np.exp((q-1)*np.log1p(-x))-h)/x**2
    gam=(x*gp-g+q)/x**2;beta=2*(g-q)/x-gp
    return np.array([q+2*beta+3*gam,-beta-3*gam,gam])
for Q in [4.,10.,20.,50.]:
    b=upper_coeff(p,Q);direction=np.sign(b);base=exact(p,2/15,Q)['capture_upper']
    # Direct derivative check; gradient agrees with the upper dual coefficients.
    def fixedq(pp):
        rr,mm,ss,*_=params(pp)
        return curve(Q,rr,mm,ss)
    numerical=np.array([(fixedq(p+np.eye(3)[k]*1e-7)-fixedq(p-np.eye(3)[k]*1e-7))/2e-7 for k in range(3)])
    assert np.max(abs(numerical-b))<1e-4,(Q,numerical.tolist(),b.tolist())
    for tolerance in [.01,.02,.05,.08]:
        def delta(e):
            pp=p+e*direction;aa=pp[0];bb=2*pp[0]-pp[1];dd=3*pp[0]-3*pp[1]+pp[2]
            assert aa>bb>dd>0 and aa*dd>bb*bb
            r0,M0,s0,*_=params(pp)
            return curve(Q,r0,M0,s0)-2/15*Q-base-tolerance
        bracket=None
        for e in np.geomspace(1e-9,.01,160):
            try:value=delta(e)
            except AssertionError:break
            if value>0:bracket=e;break
        if bracket is None:continue
        eps=brentq(delta,1e-10,bracket,xtol=1e-14)
        precision.append(dict(Q=Q,tau=tolerance,direction=direction.tolist(),derivative_coefficients=b.tolist(),
            least_favorable_mean_distance=eps,necessary_total_record_budget=4*.05**2*.9**2/eps**2,
            asymptotic_budget_coefficient=4*.9**2*(.05*np.sum(abs(b)))**2))
out=dict(status='passed',concavity_LP_checks=len(checks),max_concavity_LP_error=maxerr,
         independent_upper_kernel_checks=direction_checks,max_upper_kernel_violation=max_kernel_violation,
         independent_lower_kernel_checks=lower_checks,max_lower_kernel_violation=max_lower_violation,
         different_true_points_in_box_checks=boxchecks,box_violations=boxviol,locally_optimized_precision=precision,
         scope='Independent finite concavity LP, interpolation direction and deterministic box checks. Numerical checks do not replace proofs.')
(root/'experiments/second_paper_bounded_independent.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
