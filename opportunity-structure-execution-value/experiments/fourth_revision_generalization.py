"""General-parameter certificates, independent discretization, and cash selection.

Analytic certificates are exact rational identities. LP and Monte Carlo are
separate checks, not proofs of the continuous or finite-sample theorems.
"""
from pathlib import Path
from fractions import Fraction as F
import sys, json, math
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / '_local_dependencies'))
sys.path.insert(0, str(ROOT / 'experiments' / '_plot_dependencies'))
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def coefficients(tau, t):
    return (tau+F(1,2)-(tau+1)*t+2*t*t,
            tau/2+F(1,3)-(tau+1)*t*t/2+F(4,3)*t**3)

def independent_lp(a, m, tau, budget, n):
    # Cell averages. Dominate the affine floor on the WHOLE cell.
    # Exact first moments of step functions; no analytic witness in constraints.
    centers=(np.arange(n)+.5)/n
    objective=np.zeros(2*n+1); objective[-1]=-1
    ub=[]; rhs=[]
    row=np.zeros(2*n+1); row[n:2*n]=1
    ub.append(row);rhs.append(budget)
    for i in range(n):
        row=np.zeros(2*n+1);row[i]=1;row[n+i]=-1
        if i:row[i-1]=-1
        ub.append(row);rhs.append(0)
        row=np.zeros(2*n+1);row[i]=-1;row[-1]=tau+(i+1)/n
        ub.append(row);rhs.append(0)
    eq=np.zeros((2,2*n+1));eq[0,:n]=1/n;eq[1,:n]=centers/n
    sol=linprog(objective,A_ub=ub,b_ub=rhs,A_eq=eq,b_eq=[a,m],
                bounds=[(0,1)]*n+[(0,None)]*n+[(0,None)],method='highs')
    assert sol.success,sol.message
    return sol.x[-1]

rows=[]; checks=0
for tau in (F(1,10),F(1,5),F(2,5)):
    for budget in (F(4,5),F(1),F(6,5)):
        for t in (F(3,20),F(1,4)):
            w=F(2,5)
            A,M=coefficients(tau,t)
            a=budget*t+w*A;m=budget*t*t/2+w*M
            d=budget-(tau+1)*w
            early=budget-w*(1-2*t);end=(tau+1)*w
            if not(d>=0 and max(early,end)<=1):continue
            K=M-t*A
            assert K>0 and (m-t*a+budget*t*t/2)/K==w
            assert d+w*(tau+2*t)+w*(1-2*t)==budget
            def primitive(x):
                y=min(x,2*t);return y*y/2-t*y
            for i in range(41):
                for j in range(i,41):
                    left,right=F(i,40),F(j,40)
                    term=(right==1 and left<1)
                    assert primitive(right)-primitive(left)+t*t/2*(1-term)>=0
                    checks+=1
            floors=[independent_lp(float(a),float(m),float(tau),float(budget),n)
                    for n in (80,160,320)]
            assert all(x<=float(w)+1e-8 for x in floors)
            assert floors[-1]>float(w)-.01
            assert floors[-1]>=floors[0]-1e-8
            costs=[float((a-(tau+F(1,2))*w)/(1+k*w)) for k in (0,1,12)]
            assert all(c>0 for c in costs)
            for k in (0,1,12):
                C=(a-(tau+F(1,2))*w)/(1+k*w)
                v=(a-C)/(tau+F(1,2)+k*a)
                assert v/(1-k*v)==w
                assert (1-k*v)*a-v*(tau+F(1,2))==C
            rows.append(dict(tau=float(tau),B=float(budget),a=float(a),m=float(m),
                             t=float(t),floor=float(w),lp_floor=floors,
                             critical_cost=costs))

# Same nonnegative constant quote, same a,m and single-window budget.
# World A = deterministic center window; B = all-time Bernoulli coverage.
# Menu: uniform; .4 uniform + .6 uniform on first quarter; center quarter.
tau=.2; cost=.1; cbar=.1; pi=.25; kappa=0; R=np.array([1,2.8,4.])
wait=np.array([.5,.275,.375]);den=tau+wait
success=np.array([[.5,.2,1.],[.5,.5,.5]])
rates=(success-cost)/den
assert np.all(rates>=0)
oracle=pi*rates[0].max()+(1-pi)*rates[1].max()
uninformed=(pi*rates[0]+(1-pi)*rates[1]).max()
gap=oracle-uninformed
delta=.001;ell=.001;H=1e8;Jmax=1.2
prices=[]
for N in (100000,300000,1000000):
    s=math.sqrt(math.log(4*3/delta)/(2*N))
    eps=float(np.max(R*(1+cbar)*s/den))
    lower=H*(gap-2*eps-delta*(1+cbar)/tau)-(N+1)*Jmax/tau-N*ell
    prices.append(dict(N=N,epsilon=eps,H=H,net_lower=lower,
                       price_cap=(lower+N*ell)/N))

# Independent random logs, no world label supplied to estimator.
rng=np.random.default_rng(20261011)
coverage=0; regret=[]; selections=np.zeros((2,3),dtype=int)
N=2000;reps=128;s=math.sqrt(math.log(12/.05)/(2*N))
eps=R*(1+cbar)*s/den
for world in (0,1):
    for rep in range(reps):
        D=rng.uniform(size=N)
        Z=((D>=.25)&(D<.75)).astype(float) if world==0 else (rng.uniform(size=N)<.5).astype(float)
        weights=np.vstack([np.ones(N),.4+2.4*(D<.25),4*((D>=.25)&(D<.5))])
        estimate=(weights@(Z-cost)/N)/den
        event=np.all(np.abs(estimate-rates[world])<=eps)
        coverage+=event
        lcb=estimate-eps
        selected=int(np.argmax(lcb)) if lcb.max()>0 else -1
        vr=0 if selected<0 else rates[world,selected]
        if selected>=0:selections[world,selected]+=1
        loss=float(rates[world].max()-vr);regret.append(loss)
        if event:assert loss<=2*eps.max()+1e-12

# Nonconstant unknown quotes and success-dependent cycles, including slow ones.
cycle_cases=[]
for k in (0,1,12):
    truth=(success-(.05+.1*wait))/(den+k*success)
    n_cycle=5000;s_cycle=math.sqrt(math.log(12/.05)/(2*n_cycle))
    radii=R*1.15*s_cycle/den+k*R*s_cycle/den**2
    covered=0
    for world in (0,1):
        for rep in range(32):
            D=rng.uniform(size=n_cycle)
            Z=((D>=.25)&(D<.75)).astype(float) if world==0 else (rng.uniform(size=n_cycle)<.5).astype(float)
            weights=np.vstack([np.ones(n_cycle),.4+2.4*(D<.25),4*((D>=.25)&(D<.5))])
            Ahat=weights@(Z-(.05+.1*D))/n_cycle
            zhat=np.clip(weights@Z/n_cycle,0,1)
            estimate=Ahat/(den+k*zhat)
            event=np.all(np.abs(estimate-truth[world])<=radii)
            covered+=event
            if event:
                lower=estimate-radii
                chosen=int(np.argmax(lower)) if lower.max()>0 else -1
                value=0 if chosen<0 else truth[world,chosen]
                assert truth[world].max()-value<=2*radii.max()+1e-12
    cycle_cases.append(dict(kappa=k,paths=64,coverage=covered/64,
                            max_radius=float(radii.max())))

fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
for tau in (.1,.2,.4):
    rr=[r for r in rows if r['tau']==tau and r['t']==.15]
    ax[0].plot([r['B'] for r in rr],[r['critical_cost'][0] for r in rr],'-o',label=f'tau={tau}')
ax[0].set(xlabel='Episode budget B',ylabel='Exact critical cost (kappa=0)',
          title='General moments: certified parameter families')
ax[0].legend();ax[0].grid(alpha=.25)
# Here moments vary along each family; explicitly label to avoid fixed-moment inference.
ax[0].text(.02,.02,'a,m vary along each curve; t=.15, w=.4',transform=ax[0].transAxes,fontsize=8)
x=np.arange(3);width=.35
ax[1].bar(x-width/2,rates[0],width,label='Center-window world')
ax[1].bar(x+width/2,rates[1],width,label='Flat-coverage world')
ax[1].set(xticks=x,xticklabels=['Uniform','Early mixture','Center quarter'],
          ylabel='Net cash / actual time',title='Same moments, different best actions')
ax[1].legend(fontsize=8);ax[1].grid(axis='y',alpha=.25)
figdir=ROOT/'paper'/'figures';figdir.mkdir(parents=True,exist_ok=True)
fig.savefig(figdir/'fourth_revision_generalization.svg')
fig.savefig(figdir/'fourth_revision_generalization.png',dpi=180)
plt.close(fig)
result=dict(status='passed',exact_window_checks=checks,parameter_cases=len(rows),
            independent_LP_runs=len(rows)*3,rows=rows,
            purchase_example=dict(rates=rates.tolist(),prior=pi,oracle_rate=oracle,
                best_uninformed_rate=uninformed,information_gap=gap,certificates=prices),
            selection_simulation=dict(N=N,paths=2*reps,simultaneous_coverage=coverage/(2*reps),
                selected_actions=selections.tolist(),max_observed_regret=max(regret)),
            unknown_quote_and_cycle_checks=cycle_cases,
            scope='Exact algebra + feasible grid lower bounds + synthetic estimation checks; not empirical or a proof by simulation.')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
