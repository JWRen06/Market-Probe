"""Publication vector figures from saved analytic and simulation results."""
from pathlib import Path
import json,csv
import numpy as np
from reportlab.graphics.shapes import Drawing,String,Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib import colors
root=Path(__file__).resolve().parents[1];out=root/'paper'/'figures';out.mkdir(exist_ok=True)
blue=colors.HexColor('#156082');orange=colors.HexColor('#cc6b22');green=colors.HexColor('#397e52')
drawings=[]
def plot(d,x,y,w,h,data,title,xmin,xmax,ymin,ymax,xlabel,labels):
    lp=LinePlot();lp.x=x;lp.y=y;lp.width=w;lp.height=h;lp.data=data
    lp.xValueAxis.valueMin=xmin;lp.xValueAxis.valueMax=xmax
    lp.yValueAxis.valueMin=ymin;lp.yValueAxis.valueMax=ymax
    lp.xValueAxis.labels.fontSize=13;lp.yValueAxis.labels.fontSize=13
    lp.xValueAxis.labelTextFormat='%.2f' if xmax<3 else '%.0f'
    lp.yValueAxis.labelTextFormat='%.3f' if ymax<3 else '%.0f'
    for i,c in enumerate([blue,orange,green]):
        if i<len(data):lp.lines[i].strokeColor=c;lp.lines[i].strokeWidth=1.6
    d.add(lp);d.add(String(x,y+h+34,title,fontSize=16))
    d.add(String(x+w*.08,y-34,xlabel,fontSize=13))
    for i,label in enumerate(labels):d.add(String(x+i*w/len(labels),y+h+12,label,fontSize=11,fillColor=[blue,orange,green][i]))
def floor(b):
    lo,hi=0,.5
    for _ in range(65):
        t=(lo+hi)/2;P=40*b*t**4-80*t**3+(96-42*b)*t*t+(52*b-36)*t-5
        if P<0:lo=t
        else:hi=t
    return (.5-b*t)/(.7-1.2*t+2*t*t)
bs=np.linspace(1,1.3,151);fs=[floor(b) for b in bs]
d=Drawing(760,330)
for x,k in [(65,0),(430,12)]:
    vals=[(.5-.7*w)/(1+k*w) for w in fs]
    ref=.11 if k==0 else .015
    plot(d,x,70,275,190,[list(zip(bs,vals)),[(1,ref),(1.3,ref)]],
         f'Fee boundary (kappa={k})',1,1.3,min(vals)-.001,max(vals)+.001,'Expected interval budget B',['Exact boundary','Example fee'])
d.add(String(65,15,'Below the boundary: a bounded waiting rule strictly improves on uniform. Positive-baseline costs only.',fontSize=8))
renderSVG.drawToFile(d,str(out/'revision_structure_boundary.svg'))
renderPDF.drawToFile(d,str(out/'revision_structure_boundary.pdf'))
drawings.append(d)

r=json.loads((root/'experiments'/'episode_budget_quote_frontier.json').read_text())['rows'][0]
t=r['t'];w=r['floor'];u=np.linspace(0,1,1001,endpoint=False)
legal=w*(.2+2*t)+(1-1.2*w)*(u<t)+w*np.maximum(u-2*t,0)
general=(39/70)*(.2+u)+(847/1200)*(u<12/77)
d=Drawing(760,335)
plot(d,65,75,620,180,[list(zip(u,legal)),list(zip(u,general)),list(zip(u,(39/70)*(.2+u)))],
     'Same time moments; different opportunity structure',0,1,0,1,'Private waiting time',['Single-window feasible','Violates single-window','Fee-neutral cash floor'])
d.add(String(65,18,'Both curves: integral f=0.5, integral t*f=0.25. Blue Q=1; orange Q=1.374405.',fontSize=9))
d.add(String(65,3,'The orange world is admissible only after the effective-interval budget is enlarged.',fontSize=8))
renderSVG.drawToFile(d,str(out/'revision_legal_world_comparison.svg'))
renderPDF.drawToFile(d,str(out/'revision_legal_world_comparison.pdf'))
drawings.append(d)

# One reproducible realised ledger, with outgoing fees before income arrives.
M=1011/1600;H=2000;v0=.39/.7
def sample(method,u):
    if method=='uniform':return u
    y=u*M
    if y<=.35:return y
    return .35+(.815-np.sqrt(.815**2-2.2*(y-.35)))/1.1
paths=[]
for method,seed in [('uniform',7301),('structure',7302)]:
    rng=np.random.default_rng(seed);time=0.;cash=0.;data=[(0.,cash)]
    while time+1.2<=H:
        dd=sample(method,rng.random());fee=(.11/.35)*(dd<.35)
        cash-=fee;data.append((time,cash))
        ff=w*(.2+2*t)+(1-1.2*w)*(dd<t)+w*max(dd-2*t,0)
        z=float(rng.random()<ff);time+=.2+dd;cash+=z;data.append((time,cash))
    data.append((H,cash));paths.append(data)
    with(out/f'revision_ledger_{method}.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.writer(fh);writer.writerow(['time','cash','relative_cash','available_fee_balance'])
        writer.writerows([(tt,yy,yy-v0*tt,500+yy) for tt,yy in data])
d=Drawing(760,340)
plot(d,65,75,275,190,[[(tt,yy-v0*tt) for tt,yy in path] for path in paths],
     'Cash relative to benchmark',0,H,-55,100,'Common deadline',['Uniform','Structure rule'])
plot(d,430,75,275,190,[[(tt,500+yy) for tt,yy in path] for path in paths],
     'Available fee balance',0,H,450,1800,'Common deadline',['Uniform','Structure rule'])
d.add(String(65,21,'Two fixed seeds; one synthetic path per rule, kappa=0, initial fee balance500. No trading principal included.',fontSize=8))
d.add(String(65,8,'Income arrives at completion; fee leaves at start. These paths illustrate bookkeeping, not a probability guarantee.',fontSize=8))
renderSVG.drawToFile(d,str(out/'revision_cash_and_payment_ledger.svg'))
renderPDF.drawToFile(d,str(out/'revision_cash_and_payment_ledger.pdf'))
drawings.append(d)

panel=json.loads((root/'experiments'/'revision_unified_cash_panel.json').read_text())
selected=[r for r in panel['rows'] if r['world']=='budget1' and r['quote']=='early' and r['kappa']==0 and r['deadline']==2000]
lines=['# 重大修订：统一合成比较与核心图','',
 '384组环境与方法组合，每组256条独立合成路径。四类机会过程、两种非负中性报价、两个结算速度、两个共同期限和两个初始费用余额；逐笔费用先付、成功后收入到账，所有开始周期在截止前完整结算。',
 '', '即时执行、均匀、两数三角规则、结构规则、付费独立现金验证和完整信息基准放在同一组环境中。完整信息使用无测量费用的网格最优时刻，仅作理想参照；小余额比较基准仍是无限余额持续均匀现金率，故付款停止后的劣势不能解读成同资金策略排名。',
 '', '代表环境：单窗口达到世界、早期集中报价、κ0、H2000。均匀持续现金基准1114.286。',
 '', '|规则|初始费用余额|平均净现金±均值标准误|超过基准路径比例|付款停止比例|', '|---|---:|---:|---:|---:|']
names={'immediate':'立即执行','uniform':'均匀','triangular':'两数三角','structure_rule':'结构规则','paid_validation':'付费现金验证','full_information':'完整信息参照'}
for z in selected:lines.append(f"|{names[z['method']]}|{z['initial_fee_balance']}|{z['mean_cash']:.2f} ± {z['mean_cash_standard_error']:.2f}|{z['above_uniform_fraction']:.3f}|{z['funding_stop_fraction']:.3f}|")
lines+=['','解释：大余额下结构规则的平均现金比均匀更高，但512条直接现金验证在集中报价环境全部拒绝认证。小余额使约12%均匀路径和25%结构规则路径停止付款；长期优势与付款风险在这里明显分开。结算κ12时，H2000不足以保证完成512条训练，验证流程在测量前拒绝，直接返回均匀。',
 '', '全部方法均报现金5/50/95分位数、均值标准误、正现金比例、超过持续基准比例、付款停止、验证完成和批准比例。比例是256次模拟的频率，不是额外解析证书。budget1_3组有真实反复有效结构，单窗口保证不适用；它用于检查错误沿用结构条件时的风险。',
 '', '![结构与收费边界](figures/revision_structure_boundary.svg)', '', '![合法与非单窗口世界](figures/revision_legal_world_comparison.svg)', '', '![同截止现金与付款余额](figures/revision_cash_and_payment_ledger.svg)',
 '', '数据与复现：experiments/revision_unified_cash_panel.py/.json；experiments/revision_core_figures.py；figures中的逐笔账本CSV。所有结果为模型内合成验证。']
(root/'paper'/'重大修订统一比较与核心图.md').write_text('\n'.join(lines),encoding='utf-8')
(root/'experiments'/'revision_core_figures.json').write_text(json.dumps(dict(status='passed',figures=3,analytic_boundary_points=151,ledger_paths=2,unified_cases=panel['cases']),indent=2),encoding='utf-8')
print('Created3 vector figures,2 actual ledger CSVs and unified comparison report.')
