from pathlib import Path
from decimal import Decimal as D,localcontext,ROUND_CEILING
import json
rows=[]
with localcontext() as ctx:
 ctx.prec=90
 delta=D('.1');alpha=D('.05');c=D('.01');tau=D('.2')
 y=(D(6)-D(26).sqrt())/D(10);eta=1/(1-y);q=(D('.2')-y/3)/2;jp=D('.5')-y+y*y
 for k in [0,1,5,12,100,1000,10000]:
  kd=D(k);v=D('.3')/(D('.7')+D('.5')*kd);w=D('.3')/(D('.7')+D('.2')*kd)
  N=int((D(450)*(D(2)/delta).ln()/(w*w)).to_integral_value(rounding=ROUND_CEILING))
  jmax=kd+D('1.2');tmax=D(N)*jmax
  gamma=v*eta*(q-D(7)*jp/36);A=gamma/jmax
  et=D(2)*(D(2)*N*(1/alpha).ln()).sqrt()
  noise=D(2)*(D(2)*(1/alpha).ln()/tau).sqrt()
  B=gamma*(tmax+jmax)/jmax+c*N+et+v*jmax
  h=max(tmax+jmax,2*B/A,4*noise*noise/(A*A),tau)
  H=D(int(h.to_integral_value(rounding=ROUND_CEILING)))
  certificate=A*H-B-noise*H.sqrt()
  assert certificate>=0
  assert H>=tmax+jmax
  lower=c*(D(144)*(1-2*delta)**2/49)/(w*w*(D('.15')/((D('.2')+D('.5')*kd)*(D('.7')+D('.5')*kd))))
  assert H>=lower
  reserve=(D('.2')+c)*N+(D('.2')+D(7)*v/9)*int(H/tau)
  rows.append(dict(kappa=k,uniform_records=N,fixed_deadline=str(H),
   high_probability_relative_cash_lower=str(certificate),
   optimistic_expected_cash_required_deadline=str(lower),
   conservative_cash_reserve_excluding_execution_principal=str(reserve),
   scaled_sufficient_deadline=float(H/(kd**4)) if k else None))
out=dict(status="passed",rows=rows,confidence_world0=".8",
 scope="Analytic 90-digit sufficient fixed-deadline funded cash certificate in world0 while testing separated alternatives with two-sided classification error. Necessary expectation lower, not all-method high-probability lower; no enormous records actually generated.")
Path(__file__).with_suffix(".json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({"status":"passed","designs":len(rows),"last":rows[-1]}))
