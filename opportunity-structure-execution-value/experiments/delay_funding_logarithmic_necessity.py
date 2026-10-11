from fractions import Fraction as F
from pathlib import Path
import json,math
checks=0;rows=[]
for k in [1,12,100,10000,1000000]:
 fast=F(3,10)/(F(1,5)+F(k,2));base=F(3,10)/(F(7,10)+F(k,2))
 assert 1-base/fast==F(1,1)/(F(k)+F(7,5))
 for p in [F(i,40) for i in range(21)]:
  for d in [F(i,20) for i in range(21)]:
   assert p-F(1,5)<=fast*(F(1,5)+d+k*p);checks+=1
 H=F((k+2)**3);jf=F(6,5)
 for m in range(40):
  q=F(1,2**m);upper=fast*(H-q*(H-m*jf))
  if H>=2*m*jf and q>2/(F(k)+F(7,5)):
   assert upper<base*H;checks+=1
 rows.append(dict(kappa=k,uninterrupted_uniform_fee_buffer_order_lower=.2*max(0,math.log2((k+1.4)/2)),
   scope='Finite sufficient horizon for the necessary inequality; external income, borrowing and an equally underfunded comparator excluded.'))
out=dict(status='passed',exact_conditional_checks=checks,rows=rows,
 proof='All-first-m-opportunities-absent event has probability 2^-m; safe fees exhaust after at most floor(B0/.2) failed attempts; conditional cash-rate upper and idle loss give necessary buffer.',
 scope='Universal safe policies in A; complete settlement, minimum fee .2, no outside income, protected principal. Bound compares uninterrupted uniform expected cash; not every replenishable-budget model.')
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
