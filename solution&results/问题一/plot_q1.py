"""生成问题一三类九幅可复现图；先运行solve_q1.py。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.plot_style import apply_publication_style, export_figure, PALETTE

ROOT=Path(__file__).resolve().parent
d=pd.read_csv(ROOT/'dispatch.csv');s=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
apply_publication_style(language='zh',width='report')
plt.rcParams.update({'svg.fonttype':'none','axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
                     'figure.constrained_layout.use':False,'figure.autolayout':False})
figures=ROOT/'figures';figures.mkdir(exist_ok=True)
def create(title,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(6.3,3.9));fig.set_layout_engine(None)
    fig.subplots_adjust(left=.14,right=.96,bottom=.18,top=.84)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    return fig,ax
def save(fig,name):
    export_figure(fig,figures/name,dpi=300)
    plt.close(fig)
def hours(ax):ax.set_xlim(0,24);ax.set_xticks(np.arange(0,25,4))
t=(d.slot.to_numpy()-.5)/6
fig,ax=create('负载与光伏预测','时刻（h）','功率（kW）')
ax.plot(t,d.load_kwh*6,label='负载',color=PALETTE['primary'])
ax.plot(t,d.pv_kwh*6,label='光伏预测',color=PALETTE['secondary'],ls='--')
ax.legend(frameon=False,loc='upper left');hours(ax);save(fig,'raw_q1_supply')
fig,ax=create('144个时段的电价分布','电价（元/kWh）','时段数')
ax.hist(d.price_yuan_per_kwh,bins=12,color=PALETTE['primary'],edgecolor='white')
save(fig,'raw_q1_price_distribution')
fig,ax=create('电价与净负载的对应关系','净负载功率（kW）','电价（元/kWh）')
ax.scatter((d.load_kwh-d.pv_kwh)*6,d.price_yuan_per_kwh,s=16,color=PALETTE['primary'],alpha=.7)
ax.axvline(0,color=PALETTE['neutral'],ls='--',lw=.8);save(fig,'raw_q1_price_netload')
fig,ax=create('储电量始终保持在允许范围','时刻（h）','内部储电量（kWh）')
ax.plot(np.arange(145)/6,np.r_[6000,d.storage_end_kwh],color=PALETTE['primary'])
ax.axhline(1200,color=PALETTE['neutral'],ls='--',lw=.8);ax.axhline(10800,color=PALETTE['neutral'],ls='--',lw=.8)
ax.set_ylim(0,12000);hours(ax);save(fig,'process_q1_storage')
fig,ax=create('充放电动作互斥','时刻（h）','母线侧功率（kW）')
ax.bar(t,d.charge_kwh*6,width=1/6,color=PALETTE['primary'],label='充电')
ax.bar(t,-d.discharge_kwh*6,width=1/6,color=PALETTE['secondary'],hatch='//',label='放电（负向）')
ax.axhline(0,color='black',lw=.5);ax.legend(frameon=False,loc='upper center',ncol=2);ax.set_ylim(-6000,7500);hours(ax)
save(fig,'process_q1_actions')
fig,ax=create('充放电效率带来的累计损耗','时刻（h）','累计损耗（kWh）')
loss=(d.charge_kwh*.1+d.discharge_kwh*(1/.9-1)).cumsum()
ax.plot(np.arange(145)/6,np.r_[0,loss],color=PALETTE['contrast']);hours(ax);save(fig,'process_q1_losses')
fig,ax=create('最优计划购电量','时刻（h）','购电量（kWh/10分钟）')
ax.stairs(d.grid_kwh,np.arange(145)/6,color=PALETTE['primary']);hours(ax);save(fig,'result_q1_purchase')
fig,ax=create('储能降低全天购电费用','费用（元）','方案')
ax.barh(['无储能','最优储能'],[s['no_storage_cost_yuan'],s['cost_yuan']],color=[PALETTE['neutral'],PALETTE['primary']])
ax.set_xlim(0,60000)
for i,value in enumerate([s['no_storage_cost_yuan'],s['cost_yuan']]):ax.text(value+800,i,f'{value:,.2f}',va='center',fontsize=10)
save(fig,'result_q1_cost')
fig,ax=create('储能改变购电费用的发生时段','时刻（h）','累计购电费用（元）')
base=np.maximum(d.load_kwh-d.pv_kwh,0)*d.price_yuan_per_kwh
ax.plot(np.arange(145)/6,np.r_[0,base.cumsum()],ls='--',color=PALETTE['neutral'],label='无储能')
ax.plot(np.arange(145)/6,np.r_[0,d.cost_yuan.cumsum()],color=PALETTE['primary'],label='最优储能')
ax.legend(frameon=False,loc='upper left');hours(ax);save(fig,'result_q1_cumulative_cost')
print('9 figures exported: PNG + SVG + grayscale previews')
