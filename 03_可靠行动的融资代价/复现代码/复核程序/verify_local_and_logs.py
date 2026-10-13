"""Reproduce local-bound and independent raw-message checks; no full simulation needed."""
from pathlib import Path
import continuous_closed_loop_20261013 as m
import json,math,numpy as np
from scipy.optimize import brentq
from scipy.stats import beta
from original_local_template import old_build
OUT=Path(__file__).resolve().parent.parent
rows=[]
for eta in [.05,.1,.2,.3]:
    worlds,c,public,res,opts,gap=old_build(eta);z=worlds[0][0];mid=(worlds[0][1]+worlds[1][1])/2;d=(worlds[0][1]-worlds[1][1])/2;Q=max(opts)+1e-13
    kg=min(float(w@(z*z*np.exp(-z))) for _,w in worlds);kq=min(float(w@(z*z*np.exp(-Q*z))) for _,w in worlds)
    Lg=float(abs(d)@z);Lq=abs(float(d@z))+Q*float(abs(d)@(z*z));cg=Lg*Lg/(2*kg);cq=Lq*Lq/(2*kq);ratios=[]
    for t in np.linspace(-1,1,41):
        w=mid+t*d;q=brentq(lambda a:w@(z*np.exp(-a*z))-c,0,1,xtol=1e-15);v=float(w@(-np.expm1(-q*z))-c*q)
        for h in [-1.,-.37,0.,.53,1.]:
            if t==h:continue
            wh=mid+h*d;qh=brentq(lambda a:wh@(z*np.exp(-a*z))-c,0,1,xtol=1e-15);reg=v-float(w@(-np.expm1(-qh*z))-c*qh);ratios.append(reg/(cq*(t-h)**2))
    assert max(ratios)<=1+1e-5
    rows.append({'eta':eta,'Q':Q,'kappa_global':kg,'kappa_local':kq,'L_global':Lg,'L_local':Lq,'old_loss_coefficient':cg,'new_loss_coefficient':cq,'coefficient_reduction':cg/cq,'allowed_parameter_error_multiplier':math.sqrt(cg/cq),'max_actual_to_local_bound':max(ratios),'pairs':len(ratios)})
(OUT/'局部行动界核验.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),'utf-8')
data=json.loads((OUT/'连续闭环案例.json').read_text('utf-8'));rng=np.random.default_rng(72026013);raw=[];n=200000
for rep in [False,True]:
    plan=next(p for p in data['plans'] if p['rho']==1 and p['repeat']==rep);q=plan['q'];ker,_,_=m.kernel(q,rep)
    for theta in [-1,1]:
        w=m.W0+theta*m.D;types=rng.choice(len(m.Z),size=n,p=w/w.sum());lam=q*m.Z[types];first=rng.exponential(1/lam);hit=first<=1;bins=np.minimum((first[hit]*8).astype(int),7);category=np.zeros(n,int)
        if rep:
            extra=rng.poisson(lam[hit]*(1-first[hit]));category[hit]=1+3*bins+np.minimum(extra,2)
        else:category[hit]=1+bins
        counts=np.bincount(category,minlength=ker.shape[1]);truth=w@ker;alpha=.001/(2*(9+25));miss=0
        for num,p in zip(counts,truth):
            lo=0 if num==0 else float(beta.ppf(alpha/2,num,n-num+1));hi=1 if num==n else float(beta.ppf(1-alpha/2,num+1,n-num));miss+=int(p<lo or p>hi)
        raw.append({'repeat':rep,'theta':theta,'identities':n,'categories':len(counts),'simultaneous_interval_misses':miss,'max_frequency_error':float(max(abs(counts/n-truth)))})
assert all(r['simultaneous_interval_misses']==0 for r in raw)
(OUT/'独立消息生成器核验.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2),'utf-8')
normalization=[]
for p in data['plans']:
    model=m.model(p['rho']);design=m.design(model,p['q'],p['repeat']);I=design['score_information_per_identity']
    for theta in [-1.,-.5,0.,.5,1.]:
        prob=(m.W0+theta*p['rho']*m.D)@design['matrix'];err=abs(float(prob@design['score'])-theta*I);assert err<1e-12
    ncap=design['cap'];lg=math.log(2*len(m.CHECKS)/m.STAT_ERROR)
    radius=(math.sqrt(2*design['variance_max']*lg/(m.K*ncap))+2*design['range_bound']*lg/(3*m.K*ncap))/I
    assert abs(radius-design['radius'][ncap])<1e-12
    normalization.append({'rho':p['rho'],'repeat':p['repeat'],'radius_at_cap':radius,'includes_inverse_information_slope':True})
(OUT/'估计器归一化核验.json').write_text(json.dumps(normalization,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps({'local_pairs':sum(r['pairs'] for r in rows),'raw_message_identities':sum(r['identities'] for r in raw),'normalization_checks':len(normalization)}))
