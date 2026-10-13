"""Frozen finite reference case: same public constraints, actual window-by-window cash.
Tuning/pilot and validation seeds are separate. Not an asymptotic exponent test.
"""
from pathlib import Path
import sys,json,math,time,ast
import numpy as np
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'experiments/closure_financing_audit_20261013.py').exists()),Path(__file__).parent.parent)
sys.path.insert(0,str(ROOT/'experiments/_local_dependencies'))
from scipy.optimize import brentq
from scipy.stats import beta
OUT=Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name=='复核程序' else ROOT/'paper/连续闭环版_20261013';OUT.mkdir(exist_ok=True)
K=32;COST=1.5;EPS=.002;CHI=.005;H=4096;STAT_ERROR=.02;CASH_ERROR=.02
CHECKS=np.array([64,128,256,512,768,1024,1536,2048,3072,4096])
Z=np.array([.1,.4,1.,3.,10.,30.]);A=np.array([np.ones(6),-np.expm1(-Z),Z*Z]);PUB=np.array([1.,.5,25.])
def endpoint(free):
    w=np.full(6,1e-4);fixed=[i for i in range(6) if i not in free]
    w[free]=np.linalg.solve(A[:,free],PUB-A[:,fixed]@w[fixed]);return w
WM=endpoint([1,2,5]);WP=endpoint([0,3,4]);W0=(WP+WM)/2;D=(WP-WM)/2
THGRID=np.linspace(-1,1,20001)
def value(w,q):return float(w@(-np.expm1(-q*Z))-COST*q)
def optimum(w):return brentq(lambda q:w@(Z*np.exp(-q*Z))-COST,0,1,xtol=1e-14)

def model(rho):
    ends=[W0-rho*D,W0+rho*D];opts=[optimum(w) for w in ends];Q=max(opts)+1e-12
    curv=min(float(w@(Z*Z*np.exp(-Q*Z))) for w in ends)
    oldcurv=min(float(w@(Z*Z*np.exp(-Z))) for w in ends)
    L=abs(float(rho*D@Z))+Q*float(abs(rho*D)@(Z*Z))
    # A second valid bound: mesh derivative with explicit Lipschitz remainder.
    qgrid=np.linspace(0,Q,2049);gprime=(np.exp(-qgrid[:,None]*Z)*Z)@(rho*D)
    L=min(L,float(max(abs(gprime)))+Q/4096*float(abs(rho*D)@(Z*Z)))
    oldL=float(abs(rho*D)@Z)
    lo=np.zeros(len(THGRID));hi=np.full(len(THGRID),Q)
    weights=W0[None,:]+THGRID[:,None]*rho*D
    for _ in range(48):
        mid=(lo+hi)/2;der=(weights*Z*np.exp(-mid[:,None]*Z)).sum(1)-COST
        lo=np.where(der>0,mid,lo);hi=np.where(der>0,hi,mid)
    actions=(lo+hi)/2
    common=brentq(lambda q:value(ends[0],opts[0])-value(ends[0],q)-value(ends[1],opts[1])+value(ends[1],q),min(opts),max(opts))
    return {'rho':rho,'Q':Q,'kappa_local':curv,'L_local':L,'kappa_global':oldcurv,'L_global':oldL,'local_coefficient':L*L/(2*curv),'global_coefficient':oldL*oldL/(2*oldcurv),'action_table':actions,'common_action_loss':value(ends[0],opts[0])-value(ends[0],common),'opts':opts,'interpolation_gradient_bound':25*(L/curv)*(THGRID[1]-THGRID[0])/2}

def kernel(q,repeated,bins=8):
    lam=q*Z;edges=np.linspace(0,1,bins+1);cols=[np.exp(-lam)];reward=[0.];fees=[0.]
    for a,b in zip(edges[:-1],edges[1:]):
        pall=np.exp(-lam*a)-np.exp(-lam*b)
        if not repeated:cols.append(pall);reward.append(1.);fees.append(0.);continue
        p1=lam*np.exp(-lam)*(b-a)
        p2=lam*lam*np.exp(-lam)*(b-a-(b*b-a*a)/2)
        cols.extend([p1,p2,pall-p1-p2]);reward.extend([1.,1.-CHI,1.-2*CHI]);fees.extend([0.,CHI,2*CHI])
    mat=np.array(cols).T
    assert mat.min()>-2e-15 and max(abs(mat.sum(1)-1))<2e-14
    return np.maximum(mat,0),np.array(reward),np.array(fees)

def design(m,q,rep,horizon=H,bins=8):
    mat,reward,fees=kernel(q,rep,bins);a=W0@mat;b=m['rho']*D@mat;score=b/a;I=float(b@score)
    v0=float(a@(score*score));v1=float(b@(score*score));t=np.clip(v1/(2*I*I),-1,1)
    vmax=v0+t*v1-t*t*I*I;R=2*float(max(abs(score)))
    checks=CHECKS[CHECKS<=horizon];log=math.log(2*len(CHECKS)/STAT_ERROR)
    radius=(np.sqrt(2*vmax*log/(K*checks))+2*R*log/(3*K*checks))/I
    local=(m['L_local']*radius+m['interpolation_gradient_bound'])**2/(2*m['kappa_local'])
    good=np.flatnonzero(local<=EPS*.999)
    if not len(good):return None
    cap=int(checks[good[0]]);used=checks[checks<=cap]
    ends=[(W0+s*m['rho']*D)@mat for s in [-1,1]]
    deficit=max(0.,COST*q-min(float(p@reward) for p in ends))*K
    moment=max(float(p@(reward*reward)) for p in ends)*K
    Bcert=K*COST*q+cap*deficit+math.sqrt(2*cap*moment*math.log(1/CASH_ERROR))
    oldr=(math.sqrt(2*m['kappa_global']*EPS)-m['interpolation_gradient_bound'])/m['L_global'];logold=log
    def oldrad(n):return (math.sqrt(2*vmax*logold/(K*n))+2*R*logold/(3*K*n))/I
    oldN=math.ceil(brentq(lambda n:oldrad(n)-oldr,1,1e12))
    return {'q':float(q),'repeat':rep,'cap':cap,'checks':used.tolist(),'B_certificate':Bcert,'deficit_per_window':deficit,'score_information_per_identity':I,'variance_max':float(vmax),'range_bound':R,'global_bound_windows':oldN,'local_bound_windows':cap,'matrix':mat,'reward':reward,'fees':fees,'score':score,'radius':{int(n):float(r) for n,r in zip(checks,radius)}}

def action(m,t):return np.interp(t,THGRID,m['action_table'])
def arrays_value(rho,t,q):return ((W0+t[:,None]*rho*D)*(-np.expm1(-q[:,None]*Z))).sum(1)-COST*q
def value_upper(m,t):
    q=action(m,t);w=W0+t[:,None]*m['rho']*D
    v=(w*(-np.expm1(-q[:,None]*Z))).sum(1)-COST*q
    grad=(w*Z*np.exp(-q[:,None]*Z)).sum(1)-COST
    return v+grad*grad/(2*m['kappa_local'])

def run(m,p,theta,n,seed,balance=None,fixed=False):
    rng=np.random.default_rng(seed);probs=(W0+m['rho']*theta*D)@p['matrix'];probs=probs/probs.sum()
    wallet=np.full(n,1e100 if balance is None else balance);minimum_wallet=wallet.copy();net=np.zeros(n);score=np.zeros(n);need=np.zeros(n);alive=np.ones(n,bool)
    bankrupt=np.zeros(n,bool);output=np.zeros(n,bool);coverage=np.ones(n,bool);actions=np.full(n,np.nan);time_used=np.zeros(n,int);spending=np.zeros(n);paid=np.zeros(n);income=np.zeros(n);stop_radius=np.full(n,np.nan);hats=np.full(n,np.nan)
    cost=K*COST*p['q'];checkset=set(p['checks'])
    for j in range(1,p['cap']+1):
        ids=np.flatnonzero(alive)
        if not len(ids):break
        need[ids]=np.maximum(need[ids],cost-net[ids])
        bad=ids[wallet[ids]+1e-12<cost];bankrupt[bad]=True;alive[bad]=False
        ids=ids[wallet[ids]+1e-12>=cost]
        if not len(ids):continue
        wallet[ids]-=cost;minimum_wallet[ids]=np.minimum(minimum_wallet[ids],wallet[ids]);spending[ids]+=cost
        # Each draw is THIS window's K independent identity logs, never an aggregate endpoint draw.
        counts=rng.multinomial(K,probs,size=len(ids));cash=counts@p['reward'];logfees=counts@p['fees']
        score[ids]+=counts@p['score'];wallet[ids]+=cash;net[ids]+=cash-cost;paid[ids]+=logfees;income[ids]+=cash+logfees;time_used[ids]=j
        if j not in checkset:continue
        hat=np.clip(score[ids]/(K*j*p['score_information_per_identity']),-1,1);r=p['radius'][j]
        low=np.maximum(-1,hat-r);high=np.minimum(1,hat+r);qh=action(m,hat)
        coverage[ids]&=(low<=theta)&(theta<=high)
        worst=np.maximum(value_upper(m,low)-arrays_value(m['rho'],low,qh),value_upper(m,high)-arrays_value(m['rho'],high,qh))
        done=np.ones(len(ids),bool) if fixed and j==p['cap'] else (np.zeros(len(ids),bool) if fixed else worst<=EPS)
        chosen=ids[done];actions[chosen]=qh[done];hats[chosen]=hat[done];stop_radius[chosen]=r;output[chosen]=True;alive[chosen]=False
    truthw=W0+m['rho']*theta*D;truthq=optimum(truthw);truthv=value(truthw,truthq)
    regret=np.full(n,np.inf);ok=np.flatnonzero(output);regret[ok]=truthv-arrays_value(m['rho'],np.full(len(ok),theta),actions[ok])
    joint=output&(~bankrupt)&(regret<=EPS+1e-11)&coverage
    meta={'theta':theta,'n':n,'seed':seed,'initial_balance':balance,'fixed_stop':fixed,'successes':int(joint.sum()),'joint_success_rate':float(joint.mean()),'bankruptcy_rate':float(bankrupt.mean()),'timeout_rate':float((~output&~bankrupt).mean()),'output_action_failure_rate':float((output&(regret>EPS+1e-11)).mean()),'confidence_ever_missed_rate':float((~coverage).mean()),'windows_mean':float(time_used.mean()),'windows_p95':float(np.quantile(time_used,.95)),'windows_max':int(time_used.max()),'resources_mean':float(spending.mean()),'fees_mean':float(paid.mean()),'gross_income_mean':float(income.mean()),'terminal_cash_mean':None if balance is None else float(wallet.mean()),'min_cash_recorded':None if balance is None else float(minimum_wallet.min()),'parameter_rmse_given_output':float(np.sqrt(np.mean((hats[ok]-theta)**2))) if len(ok) else None,'max_regret_given_output':float(max(regret[ok])) if len(ok) else None}
    return meta,need

def clean(p):return {k:v for k,v in p.items() if k not in ['matrix','reward','fees','score','radius']}

def main():
    models=[model(r) for r in [.6,.8,1.]];menu=np.geomspace(.005,.24,65);plans=[];pilot=[];validation=[];controls=[];frontier=[]
    for mi,m in enumerate(models):
        for rep in [False,True]:
            candidates=[p for q in menu if (p:=design(m,float(q),rep)) is not None]
            assert candidates
            p=min(candidates,key=lambda p:p['B_certificate'])
            needs=[]
            for ti,t in enumerate([-1.,-.5,0.,.5,1.]):
                row,req=run(m,p,t,400,10000+mi*100+int(rep)*10+ti);row.update(rho=m['rho'],permission='buy' if rep else 'first');pilot.append(row)
                needs.append(float(np.quantile(req,.995,method='higher')))
            # Frozen pilot selection, before independent validation draws.
            Btrial=max(needs)*1.2+K*COST*p['q'];p['B_trial']=Btrial
            plans.append({'rho':m['rho'],**clean(p)})
            print(json.dumps({'stage':'design','rho':m['rho'],'repeat':rep,'q':p['q'],'cap':p['cap'],'B_cert':p['B_certificate'],'B_trial':Btrial}),flush=True)
            for ti,t in enumerate(np.linspace(-1,1,9)):
                row,_=run(m,p,float(t),2000,900000+mi*100+int(rep)*10+ti,balance=Btrial)
                row.update(rho=m['rho'],permission='buy' if rep else 'first');validation.append(row)
            if m['rho']==1.:
                for ti,t in enumerate([-1.,0.,1.]):
                    for ci,typ in enumerate(['quarter_cash','fixed_stop']):
                        row,_=run(m,p,t,1000,1900000+int(rep)*100+ti*10+ci,balance=Btrial*(.25 if ci==0 else 1),fixed=ci==1)
                        row.update(rho=m['rho'],permission='buy' if rep else 'first',control=typ);controls.append(row)
                for horizon in [256,512,768,1024,1536,2048,3072,4096,8192,16384]:
                    # Extending the predeclared checkpoint menu is deliberately excluded here.
                    candidates=[p2 for q in menu if (p2:=design(m,float(q),rep,min(horizon,H))) is not None]
                    frontier.append({'H':horizon,'permission':'buy' if rep else 'first','B_certificate':min((x['B_certificate'] for x in candidates),default=None)})
    alpha=.05/len(validation)
    for r in validation:r['simultaneous_95pct_lower']=float(beta.ppf(alpha,r['successes'],r['n']-r['successes']+1)) if r['successes'] else 0.
    payload={'protocol':{'K':K,'M':K,'public_P1':K*.5,'public_B2':K*25,'c':K*COST,'epsilon':K*EPS,'chi':CHI,'H':H,'stat_error':STAT_ERROR,'cash_error':CASH_ERROR,'categories_first':9,'categories_repeat':25,'final_paths':sum(r['n'] for r in validation),'joint_event':'finite legal output, true action accuracy, all computed parameter confidence intervals cover theta','finite_case_not_asymptotic_eta_test':True},'definition':{'speeds':Z.tolist(),'weights_minus':WM.tolist(),'weights_plus':WP.tolist(),'rho_values':[.6,.8,1.],'minimum_weight':float(min(WM.min(),WP.min())),'public_residual':float(max(abs(A@WM-PUB).max(),abs(A@WP-PUB).max()))},'models':[{k:v for k,v in m.items() if k!='action_table'} for m in models],'plans':plans,'pilot':pilot,'validation':validation,'controls':controls,'frontier':frontier,'all_validation_cells_lower_above_95pct':all(r['simultaneous_95pct_lower']>=.95 for r in validation)}
    (OUT/'连续闭环案例.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({'stage':'done','minimum_success':min(r['joint_success_rate'] for r in validation),'minimum_lower':min(r['simultaneous_95pct_lower'] for r in validation),'all_cells_pass':payload['all_validation_cells_lower_above_95pct']}),flush=True)
if __name__=='__main__':main()
