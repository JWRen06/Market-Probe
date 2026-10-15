from pathlib import Path
import sys,ast,json
root=Path(__file__).resolve().parent


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
src=ast.parse((root/'experiments/second_paper_bounded_revision.py').read_text(encoding='utf-8-sig'));ns={'np':np}
for node in src.body:
 if isinstance(node,ast.FunctionDef):exec(compile(ast.Module(body=[node],type_ignores=[]),'<f>','exec'),ns)
folder=root.parent/'图表';folder.mkdir(parents=True,exist_ok=True)
p=np.array([.4,.64,.816]);c=.1;Q=np.linspace(3,20,150);rows=[ns['exact'](p,c,q) for q in Q]
fig,ax=plt.subplots(1,2,figsize=(9.1,3.2))
ax[0].fill_between(Q,[x['concave_lower'] for x in rows],[x['concave_upper'] for x in rows],color='#DDE4E9',label='仅凹性')
ax[0].fill_between(Q,[x['capture_lower'] for x in rows],[x['capture_upper'] for x in rows],color='#8DB6C7',alpha=.9,label='固定捕获结构')
ax[0].set(xlabel='最大投入 Q',ylabel='每窗口净收益范围');ax[0].legend(frameon=False)
ax[1].plot(Q,[x['capture_action'] for x in rows],color='#256579',label='捕获结构的保守投入')
ax[1].plot(Q,[x['concave_action'] for x in rows],color='#697982',ls='--',label='凹性基准的保守投入')
ax[1].set(xlabel='最大投入 Q',ylabel='最佳保守投入');ax[1].legend(frameon=False,loc='lower right')
for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(folder/'bounded_profit_and_action.png',dpi=300);fig.savefig(folder/'bounded_profit_and_action.svg');plt.close(fig)
data=json.loads((root/'experiments/second_paper_bounded_revision.json').read_text(encoding='utf-8-sig'))
act=json.loads((root/'experiments/second_paper_bounded_deployment.json').read_text(encoding='utf-8-sig'))
qs=[4,10,20];records=[np.median([x['total_records'] for x in data['unknown_direction_sequential'] if x['Q']==q]) for q in qs]
aqs=[10,20,50];arecords=[x['median_total_records'] for x in act['summary']]
fig,ax=plt.subplots(figsize=(7.7,3.1));ax.plot(qs,records,'o-',color='#256579',label='利润上端精度');ax.plot(aqs,arecords,'s--',color='#AC6A36',label='保守行动证书')
ax.set(xlabel='最大投入 Q',ylabel='总记录数（对数轴）',yscale='log');ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.25)
ax.text(.02,.97,'精度 0.08；噪声 0.05；费用 2/15；每组80路径中位数',transform=ax.transAxes,va='top',fontsize=9)
fig.tight_layout();fig.savefig(folder/'precision_and_action_cost.png',dpi=300);fig.savefig(folder/'precision_and_action_cost.svg');plt.close(fig)
print('two main figures generated')
