"""Finite known reward caps and native bounded aggregate count experiments.

All examples are synthetic. The document-based application mapping is in the paper.
No account, advertising platform, market data or network access is used.
"""
from pathlib import Path
import json, math, itertools
import numpy as np
from scipy.optimize import linprog, brentq
from scipy.stats import chi2, beta

ROOT=Path(__file__).resolve().parents[1]
P0=np.array([4.,6.4,8.16])
COST=.8

def parameters(p):
    a=p[0];b=2*p[0]-p[1];d=3*p[0]-3*p[1]+p[2]
    if not a>b>d>0 or a*d<=b*b:raise ValueError('not strict moment interior')
    ell=(b-d)/(a-b);wg=(a-b)/(1-ell)
    return a,b,d,a-b*b/d,ell,a-wg,wg/ell,a+(a-b)**2/(b-d)

def capped_atoms(p,S):
    a,b,d,r,ell,mg,MG,T=parameters(p)
    if S<=T:raise ValueError('cap must exceed minimum mass')
    den=S*b-a*a
    node_sum=(S*d-a*b)/den
    node_product=(a*d-b*b)/den
    high=(node_sum+np.sqrt(node_sum**2-4*node_product))/2
    low=node_product/high
    wh=(a-S*low)/(high-low);wl=S-wh
    assert 0<low<high<1 and min(wl,wh)>0
    return np.array([low,high]),np.array([wl,wh])

def H(p,S,q):
    v,w=capped_atoms(p,S);q=np.asarray(q)
    return np.sum(w*(-np.expm1(np.expand_dims(q,-1)*np.log1p(-v))),axis=-1)

def G(p,q):
    a,b,d,r,ell,mg,MG,T=parameters(p)
    return mg+MG*(-np.expm1(np.asarray(q)*np.log1p(-ell)))

def capped_values(p,S,q):
    h=H(p,S,q);g=G(p,q)
    return np.minimum(h,g),np.maximum(h,g)

def H_derivative(p,S,q):
    v,w=capped_atoms(p,S);z=-np.log1p(-v)
    return float(np.sum(w*z*np.exp(-q*z)))

def peak_h(p,S,c,lo,hi):
    if H_derivative(p,S,lo)<=c:q=lo
    elif H_derivative(p,S,hi)>=c:q=hi
    else:q=brentq(lambda x:H_derivative(p,S,x)-c,lo,hi)
    return float(q),float(H(p,S,q)-c*q)

def peak_g(p,c,lo,hi):
    a,b,d,r,ell,mg,MG,T=parameters(p);z=-np.log1p(-ell)
    q=float(np.clip(np.log(MG*z/c)/z,lo,hi))
    return q,float(G(p,q)-c*q)

def capped_upper(p,S,c,Q):
    # H is the sharp upper on [1,2] and [3,infinity); G on other intervals.
    options=[(0.,0.)]
    for lo,hi,fun in [(0.,1.,peak_g),(1.,2.,peak_h),(2.,3.,peak_g),(3.,Q,peak_h)]:
        if hi<lo:continue
        options.append(fun(p,S,c,lo,hi) if fun==peak_h else fun(p,c,lo,hi))
    return max(options,key=lambda x:x[1])

def gauss_actual_optimum(p,S,c,Q):return peak_h(p,S,c,0.,Q)

def interpolation_coefficients(p,q):
    # Cubic Hermite majorant/minorant of 1-(1-v)^q, contact at Gauss nodes.
    v,w=capped_atoms(p[0],p[1]);matrix=[];rhs=[]
    for x in v:
        matrix.append([1.,x,x*x,x*x*x]);rhs.append(-np.expm1(q*np.log1p(-x)))
        matrix.append([0.,1.,2*x,3*x*x]);rhs.append(q*(1-x)**(q-1))
    return np.linalg.solve(matrix,np.array(rhs))

def lower_g_coeff(p,q):
    a,b,d,r,x,mg,MG,T=parameters(p)
    value=-np.expm1(q*np.log1p(-x))/x
    prime=(q*x*np.exp((q-1)*np.log1p(-x))-1+(1-x)**q)/x**2
    gam=(1-value-prime*(1-x))/(1-x)**2
    bet=prime-2*gam*x;A=value-prime*x+gam*x*x
    return np.array([A+2*bet+3*gam,-bet-3*gam,gam])

def action_certificate(lo,hi,S,Q,c):
    # Count Chernoff intervals feed a deterministic moment certificate.
    center=(lo+hi)/2;radius=(hi-lo)/2
    vertices=np.array(list(itertools.product(*zip(lo,hi))))
    try:
        if any(parameters(x)[-1]>=S for x in vertices):return None
        # Simplify first: a-B=p2-p1; B-D=-p1+2p2-p3.
        # Expanding intervals before these cancellations needlessly multiplies the error.
        tlo=lo[1]-hi[0];thi=hi[1]-lo[0]
        ulo=-hi[0]+2*lo[1]-hi[2];uhi=-lo[0]+2*hi[1]-lo[2]
        if min(tlo,ulo)<=0:return None
        ell_lo=ulo/thi;ell_hi=uhi/tlo
        if not 0<ell_lo<=ell_hi<1:return None
        def h(q,e):return -np.log1p(-e)/e*(1-e)**(q-1)
        denom_max=.25 if ell_lo<=.5<=ell_hi else max(ell_lo*(1-ell_lo),ell_hi*(1-ell_hi))
        denom_min=min(ell_lo*(1-ell_lo),ell_hi*(1-ell_hi))
        MGlo=tlo/denom_max;MGhi=thi/denom_min
        if tlo*h(3,ell_hi)<=c:return None
        q,_=peak_g(center,c,3.,Q);coef=lower_g_coeff(center,q)
        low=max(0.,float(coef@center-np.abs(coef)@radius-c*q))
        upper=max(peak_g(x,c,3.,Q)[1] for x in vertices)
        zlo=-np.log1p(-ell_lo);zhi=-np.log1p(-ell_hi)
        A_lo=np.log(MGlo*zlo/c);A_hi=np.log(MGhi*zhi/c)
        if A_lo<=0:return None
        qlo=float(np.clip(A_lo/zhi,3.,Q));qhi=float(np.clip(A_hi/zlo,3.,Q))
        curvature=thi*zhi*h(3,ell_lo)
        distance_regret=.5*curvature*max(abs(q-qlo),abs(q-qhi))**2 if qhi<Q else float('inf')
        residual=max(abs(tlo*h(q,ell_hi)-c),abs(thi*h(q,ell_lo)-c))
        strong_curvature=tlo*zlo*h(max(q,qhi),ell_hi)
        gradient_regret=residual*residual/(2*strong_curvature)
        regret=min(distance_regret,gradient_regret)
        return low,upper,q,regret,qlo,qhi
    except (ValueError,FloatingPointError):return None

def main():
    rng=np.random.default_rng(2026101531)
    lp_rows=[];max_lp_error=0.;max_moment_error=0.;max_dual_violation=0.
    qs=[.2,.8,1.4,2.4,3.,4.,8.,20.]
    for case in range(6):
        v0=rng.uniform(.06,.9,7);w0=rng.uniform(.2,2.,7)
        p=np.array([np.sum(w0*(-np.expm1(k*np.log1p(-v0)))) for k in [1,2,3]])
        for factor in [1.1,3.,20.]:
            S=parameters(p)[-1]*factor;v,w=capped_atoms(p,S)
            moments=np.array([sum(w*v**k) for k in [1,2,3]])
            target=np.array(parameters(p)[:3]);max_moment_error=max(max_moment_error,max(abs(moments-target)))
            grid=np.unique(np.r_[np.linspace(0,1,801),v,parameters(p)[4]])
            eq=np.array([grid,grid**2,grid**3])
            for q in qs:
                response=-np.expm1(q*np.log1p(-np.minimum(grid,1-1e-16)));response[grid==1]=1.
                low,high=capped_values(p,S,q)
                for sign,expected in [(1,float(low)),(-1,-float(high))]:
                    sol=linprog(sign*response,A_eq=eq,b_eq=target,A_ub=np.ones((1,len(grid))),b_ub=[S],bounds=(0,None),method='highs')
                    assert sol.success
                    err=abs(sol.fun-expected);max_lp_error=max(max_lp_error,err)
                    assert err<2e-6,(case,factor,q,sol.fun,expected)
                    lp_rows.append(dict(case=case,cap=S,q=q,side='lower' if sign==1 else 'upper',error=err))
                coeff=interpolation_coefficients((p,S),q)
                interp=sum(coeff[j]*grid**j for j in range(4))
                upper_branch=(1<=q<=2) or q>=3
                violation=max(response-interp) if upper_branch else max(interp-response)
                max_dual_violation=max(max_dual_violation,float(violation))
                assert violation<2e-8
                assert coeff[0]>=-2e-8 if upper_branch else coeff[0]<=2e-8
    assert max_moment_error<1e-9

    # A finite fixed cap eventually makes the target constant; before that cap it can be sensitive.
    caps=[100.,400.,1600.];Qgrid=[10.,20.,40.,80.,160.,320.,640.,1280.]
    frontier=[];alpha=beta_error=.05;tau=.5
    for S in caps:
        for Q in Qgrid:
            qopt,U=capped_upper(P0,S,COST,Q)
            frontier.append(dict(cap=S,Q=Q,upper=U,optimum=qopt,formal_cutoff=S/COST))
        cutoff=S/COST
        assert abs(capped_upper(P0,S,COST,cutoff)[1]-capped_upper(P0,S,COST,2*cutoff)[1])<1e-8

    necessity=[];slow_r=1.4;fast_count=8;fast_z=.9
    def native_p(K,e=0):
        z=np.array([slow_r*(1+e)/K,fast_z]);w=np.array([K-fast_count,fast_count])
        return np.array([sum(w*(-np.expm1(-k*z))) for k in [1,2,3]])
    for regime in ['fixed_cap','cap_proportional_to_Q']:
        for Q in [10.,20.,40.,80.,160.,320.,640.]:
            S=100 if regime=='fixed_cap' else int(10*Q);p=native_p(S)
            _,U=capped_upper(p,S,COST,Q)
            qbase,actual=gauss_actual_optimum(p,S,COST,Q)
            assert abs(actual-U)<1e-7
            def difference(e):return gauss_actual_optimum(native_p(S,e),S,COST,Q)[1]-U-tau
            ehi=.001
            while difference(ehi)<0:ehi*=2
            eps=brentq(difference,0.,ehi,xtol=1e-14)*(1+1e-5)
            p1=native_p(S,eps)
            assert gauss_actual_optimum(p1,S,COST,Q)[1]>U+tau
            uniform_KL_bound=eps*eps*3*slow_r*np.exp(2*(1+eps)*slow_r*3/S)
            rigorous_budget=2*(1-alpha-beta_error)**2/uniform_KL_bound
            necessity.append(dict(regime=regime,Q=Q,cap=S,baseline_readouts=p.tolist(),epsilon=eps,baseline_upper=U,alternative_upper=gauss_actual_optimum(p1,S,COST,Q)[1],uniform_KL_bound_for_any_pilot_q_le_3=uniform_KL_bound,necessary_continuous_pilot_records=rigorous_budget))
    moving=[x for x in necessity if x['regime']=='cap_proportional_to_Q']
    slope=float(np.polyfit(np.log([x['Q'] for x in moving[-4:]]),np.log([x['necessary_continuous_pilot_records'] for x in moving[-4:]]),1)[0])
    assert 1.85<slope<2.15

    # Known fixed population, bounded individual reward and bounded per-window count.
    pool_checks=[]
    for S in [100,400]:
        for q in [1.,2.,3.]:
            probs=-np.expm1(-q*np.array([slow_r/S,fast_z]));w=np.array([S-fast_count,fast_count])
            captured=rng.binomial(w,probs,size=(50000,2)).sum(axis=1)
            mean=float(w@probs);variance=float(w@(probs*(1-probs)));se=math.sqrt(variance/len(captured))
            assert abs(captured.mean()-mean)<6*se
            assert abs(captured.var()-variance)<.2 and max(captured)<=S
            pool_checks.append(dict(cap=S,q=q,paths=len(captured),predicted_mean=mean,predicted_variance=variance,observed_mean=float(captured.mean()),observed_variance=float(captured.var()),maximum_observed_count=int(max(captured))))

    # Complete native-count inference/action runs; no Gaussian proxy or monetary survival claim.
    native=[];groups=[];R=400;J=19;delta=.05;action_tolerance=.5
    designs=list(itertools.product([100,400,1600],[20.,80.],[0.,.05,.1]))
    for group,(S,Q,perturbation) in enumerate(designs):
        p=native_p(S,perturbation);rrng=np.random.default_rng(2026101541+group);means=p
        probs=-np.expm1(-np.array([1.,2.,3.])[:,None]*np.array([slow_r*(1+perturbation)/S,fast_z]))
        counts=np.zeros((R,3),dtype=np.int64);alive=np.ones(R,dtype=bool);covered=np.ones(R,dtype=bool);prev=0
        actual_best=peak_g(p,COST,3.,Q)[1]
        for stage in range(1,J+1):
            N=32*2**(stage-1);batch=N-prev;prev=N
            ix=np.flatnonzero(alive)
            counts[ix]+=rrng.binomial(batch*(S-fast_count),probs[:,0],size=(len(ix),3))+rrng.binomial(batch*fast_count,probs[:,1],size=(len(ix),3))
            eta=delta/(3*stage*(stage+1))
            s=counts[ix]
            y=s/N;b=math.log(2/eta)/N;rad=np.sqrt(2*y*b)
            lo=np.maximum(0.,y-rad-b);hi=y+rad+2*b
            covered[ix]&=np.all((means>=lo)&(means<=hi),axis=1)
            for j,index in enumerate(ix):
                cert=action_certificate(lo[j],hi[j],S,Q,COST)
                if cert is None or cert[3]>action_tolerance:continue
                lower,upper,q,regret,qlo,qhi=cert
                actual=min(float(H(p,S,q)),float(G(p,q)))-COST*q
                if covered[index]:assert lower-1e-8<=actual_best<=upper+1e-8 and actual>=lower-1e-8 and actual_best-actual<=regret+1e-8
                success=bool(covered[index] and actual>=actual_best-action_tolerance-1e-8)
                native.append(dict(group=group,cap=S,Q=Q,perturbation=perturbation,rep=int(index),records=3*N,stage=stage,action=q,certified_lower=lower,certified_upper=upper,regret_certificate=regret,optimum_interval=[qlo,qhi],actual_guarantee=actual,actual_best_robust=actual_best,all_intervals_covered=bool(covered[index]),joint_success=success))
                alive[index]=False
            if not alive.any():break
        assert not alive.any()
        rows=[x for x in native if x['group']==group];success=sum(x['joint_success'] for x in rows)
        bound=float(beta.ppf(.05/len(designs),success,R-success+1)) if success else 0.
        groups.append(dict(cap=S,Q=Q,perturbation=perturbation,paths=R,joint_successes=success,joint_success_rate=success/R,simultaneous_one_sided_95_lower=bound,median_total_records=float(np.median([x['records'] for x in rows])),max_stage=max(x['stage'] for x in rows)))

    # Cross-check against the full piecewise robust optimum at other feasible means.
    audit_rng=np.random.default_rng(2026101551);boxes=0;points=0;max_excess=-1e10
    for _ in range(80):
        vv=audit_rng.uniform(.08,.9,5);ww=audit_rng.uniform(.1,2.,5)
        pp=np.array([sum(ww*(-np.expm1(k*np.log1p(-vv)))) for k in [1,2,3]])
        aa,bb,dd,rr,ee,mg,MG,T=parameters(pp);zz=-np.log1p(-ee)
        cc=MG*zz*np.exp(-audit_rng.uniform(3.5,6.)*zz);SS=1.8*T
        radius=float(max(pp))*audit_rng.uniform(.00005,.0005);lo=pp-radius;hi=pp+radius
        cert=action_certificate(lo,hi,SS,10.,cc)
        if cert is None:continue
        boxes+=1
        for _ in range(40):
            truth=audit_rng.uniform(lo,hi);low,upper,q,regret,ql,qh=cert
            best=max(0.,peak_h(truth,SS,cc,0,1)[1],peak_g(truth,cc,1,2)[1],peak_h(truth,SS,cc,2,3)[1],peak_g(truth,cc,3,10)[1])
            actual=float(min(H(truth,SS,q),G(truth,q))-cc*q)
            excess=best-actual-regret;max_excess=max(max_excess,excess)
            assert excess<=2e-8 and low<=actual+2e-8 and best<=upper+2e-8
            points+=1
    assert boxes>=30 and points>=1200
    out=dict(status='passed',scope='Finite known total reward cap; fixed finite population; reward 1 per identity; bounded raw aggregate count, not a Gaussian or Poisson observation proxy. Synthetic, no platform or market calibration.',native_constants=dict(slow_r=slow_r,fast_count=fast_count,fast_z=fast_z,c=COST,pilot_max=3.),known_cap_closed_form_LP_checks=len(lp_rows),max_LP_error=max_lp_error,max_moment_error=max_moment_error,max_dual_grid_violation=max_dual_violation,LP_checks=lp_rows,finite_cap_frontier=frontier,finite_scale_necessary_records=necessity,moving_cap_log_slope_last_four=slope,latent_pool_checks=pool_checks,native_action_groups=groups,native_action_paths=native,native_paths=len(native),native_action_precision=action_tolerance,native_nominal_failure=delta,certificate_random_box_checks=dict(boxes=boxes,different_true_points=points,max_actual_regret_minus_certificate=max_excess),continuous_pilot_lower_bound='analytic uniform Bernoulli KL bound in appendix E; no grid approximation used for displayed necessary records',cash_survival_experiment=False)
    (ROOT/'experiments/application_contract_bridge.json').write_text(json.dumps(out,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['LP_checks','finite_cap_frontier','finite_scale_necessary_records','native_action_paths','latent_pool_checks']},ensure_ascii=True))

if __name__=='__main__':main()
