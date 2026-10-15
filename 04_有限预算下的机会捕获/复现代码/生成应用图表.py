from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'图表';OUT.mkdir(exist_ok=True)
RESULT=json.loads((ROOT/'experiments/application_contract_bridge.json').read_text('utf-8'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
fig,ax=plt.subplots(1,2,figsize=(10.2,3.6))
for S in [100,400,1600]:
    rows=[x for x in RESULT['finite_cap_frontier'] if x['cap']==S]
    ax[0].plot([x['Q'] for x in rows],[x['upper'] for x in rows],'o-',label=f'S={S}')
ax[0].set(xscale='log',xlabel='最大部署强度 Q',ylabel='有限S下的兼容利润上端');ax[0].legend(frameon=False)
for regime,label in [('fixed_cap','固定K=100'),('cap_proportional_to_Q','已知K=10Q')]:
    rows=[x for x in RESULT['finite_scale_necessary_records'] if x['regime']==regime]
    ax[1].plot([x['Q'] for x in rows],[x['necessary_continuous_pilot_records'] for x in rows],'o-',label=label)
ax[1].set(xscale='log',yscale='log',xlabel='最大部署强度 Q',ylabel='原始计数的解析必要记录下界');ax[1].legend(frameon=False)
for a in ax:a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(OUT/'known_cap_and_native_counts.png',dpi=300);plt.close(fig)

