"""问题三九幅矢量论文插图。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from utils.publication_style import setup,create,save,NAMES,PALETTE as C
R=Path(__file__).resolve().parent; F=R/'figures'
titles=[('raw_q3_forecasts','多时刻光伏预报与实际功率'),('raw_q3_errors','共同目标时段的预报误差分布'),('raw_q3_energy','日前预报与实际光伏日总电量'),('process_q3_purchase','原计划与最终调整购电量'),('process_q3_storage','日内更新与实际储电量'),('process_q3_increments','不同决策时刻的最终增购电量'),('result_q3_monthly','月度购电费用构成'),('result_q3_policies','相同起止库存下的策略费用'),('result_q3_information','预报更新与调整策略的配对费用差')]
NAMES.update({k:f'图{i+19} 问题三{title}' for i,(k,title) in enumerate(titles)})
setup();d=pd.read_csv(R/'dispatch.csv');v=pd.read_csv(R/'schedule_versions.csv');day=d[d.date=='2025-03-20'];vers=v[v.date=='2025-03-20']
def finish(fig,k,ncol=2):
    for ax in fig.axes:
        if ax.get_legend_handles_labels()[1]:ax.legend()
    save(fig,F,k,ncol)
def timeaxis(ax):ax.set_xlim(0,24);ax.set_xticks([0,6,12,18,24])
# 实际值一律纯黑；四个发布时刻固定蓝/橙/绿/红，与全文配色一致。
# 每条预报曲线的起点加一个实心标记：夜间各曲线都归零并互相重叠，没有标记时
# 读者会误以为四条曲线都贯穿0—24时，而0/6/12/18时的实际覆盖是逐级缩短的。
fig,ax=create('','时刻 / h','光伏功率 / kW')
ax.plot((day.slot-.5)/6,day.pv_kwh*6,color=C['observed'],label='实际值',lw=1.2)
for hour,color,ls in [(0,C['f0'],'--'),(6,C['f6'],':'),(12,C['f12'],'-.'),(18,C['f18'],(0,(4,1,1,1)))]:
    z=vers[vers.decision_hour==hour].sort_values('slot')
    ax.plot((z.slot-.5)/6,z.forecast_pv_kwh*6,color=color,ls=ls,label=f'{hour}时预报',
            marker='o',ms=2.2,markevery=[0],mew=0)
timeaxis(ax);finish(fig,titles[0][0],ncol=3)   # 五个图例项，排2行才不会顶出画布
e=pd.read_csv(R/'forecast_diagnostics.csv')
fig,ax=create('','预报发布时间','日平均绝对误差 / kW')
ax.boxplot([e[e.release_hour==h].mae_kw for h in [0,6,12]],tick_labels=['0时','6时','12时'],patch_artist=True,
    boxprops={'facecolor':'#CBDCF0','edgecolor':C['f0'],'linewidth':.8},medianprops={'color':C['f18'],'linewidth':1.2},
    whiskerprops={'linewidth':.8,'color':C['f0']},capprops={'linewidth':.8,'color':C['f0']},
    flierprops={'marker':'.','markersize':2,'markeredgecolor':C['neutral']})
finish(fig,titles[1][0])
f0=v[v.decision_hour==0].groupby('date').forecast_pv_kwh.sum()/1000;obs=d.groupby('date').pv_kwh.sum()/1000
fig,ax=create('','0时预报光伏日总电量 / MWh','实际光伏日总电量 / MWh')
ax.scatter(f0,obs,s=9,color=C['observed'],linewidths=.5,facecolors='none');lim=max(f0.max(),obs.max())*1.05
ax.plot([0,lim],[0,lim],color=C['f6'],ls='--',label='相等参考线');ax.set_xlim(0,lim);ax.set_ylim(0,lim);finish(fig,titles[2][0])
day=d[d.date=='2025-12-21']
fig,ax=create('','时刻 / h','普通购电量 / (kWh/10 min)')
ax.stairs(day.original_grid_kwh,np.arange(145)/6,color=C['f6'],ls='--',label='0时原计划')
ax.stairs(day.adjusted_grid_kwh,np.arange(145)/6,color=C['f0'],label='最终有效调整量')
timeaxis(ax);finish(fig,titles[3][0])
fig,ax=create('','时刻 / h','内部储电量 / kWh')
ax.plot(np.arange(145)/6,np.r_[day.storage_start_kwh.iloc[0],day.storage_end_kwh],color=C['observed'],label='实际储电量')
for h in [6,12,18]:ax.axvline(h,color=C['f0'],ls=':',lw=.6)
for y in [1200,10800]:ax.axhline(y,color=C['f18'],ls='--',lw=.8)
timeaxis(ax);ax.set_ylim(0,12000);finish(fig,titles[4][0])
fig,ax=create('','最终有效版本的决策时刻','评价期增购电量 / MWh')
inc=d.groupby('decision_hour').increase_kwh.sum().reindex([6,12,18],fill_value=0)/1000
ax.bar(['6时','12时','18时'],inc,color=[C['f0'],C['f6'],C['f18']],edgecolor='white',lw=.5)
finish(fig,titles[5][0])
monthly=d.groupby(d.date.str[:7])[['plan_cost_yuan','increase_cost_yuan','decrease_cost_yuan','emergency_cost_yuan']].sum()/10000
fig,ax=create('','月份','购电费用 / 万元');pos=np.arange(11);base=np.zeros(11)
for values,label,color,hatch in [(monthly.plan_cost_yuan,'原计划费用',C['f0'],None),(monthly.increase_cost_yuan+monthly.decrease_cost_yuan,'调整相关费用',C['f6'],'//'),(monthly.emergency_cost_yuan,'紧急购电费用',C['f18'],'..')]:
    ax.bar(pos,values,bottom=base,label=label,color=color,hatch=hatch,edgecolor='white',lw=.5);base+=values.to_numpy()
ax.set_xticks(pos,range(2,13));finish(fig,titles[6][0])
p=pd.read_csv(R/'policy_comparison.csv');p=p[p.fair_boundary].set_index('policy')
keys=['only0','add6','add12','all','stale_all','stale18'];labels=['仅0时','0/6时','0/6/12时','全部更新','仅旧预报','18时旧预报']
fig,ax=create('','评价期购电费用 / 万元','')
# 前四档是同一条递进路径，用蓝→橙→绿→红；两条对照不是这条路径上的点，另用紫和浅蓝区分。
ax.barh(labels,p.loc[keys].total_cost_yuan/10000,color=[C['f0'],C['f6'],C['f12'],C['f18'],C['accent'],C['sky']],edgecolor='white',lw=.5);ax.invert_yaxis();finish(fig,titles[7][0])
b=pd.read_csv(R/'bootstrap_comparisons.csv');b=b[b.block_days==7];pairs=[('only0','add6'),('add6','add12'),('add12','all'),('stale_all','all'),('stale18','all')]
fig,ax=create('','平均每日节省及95%区间 / 元','');ys=np.arange(5)
for i,(before,after) in enumerate(pairs):
    z=b[(b.before==before)&(b.after==after)].iloc[0];color=[C['f0'],C['f6'],C['f12'],C['f18'],C['accent']][i]
    ax.plot([z.ci_low_yuan_per_day,z.ci_high_yuan_per_day],[i,i],color=color,lw=1.)
    ax.plot(z.mean_saving_yuan_per_day,i,'o',color=color,ms=3)
    ax.plot([z.ci_low_yuan_per_day,z.ci_high_yuan_per_day],[i,i],'|',color=color,ms=6)
ax.set_yticks(ys,['加入6时','加入12时','加入18时','全部新预报信息','18时新预报信息']);ax.invert_yaxis();ax.axvline(0,color=C['neutral'],ls='--',lw=.8)
fig.subplots_adjust(left=.25);finish(fig,titles[8][0])
print('9 figures exported: PDF / SVG / 1200 DPI PNG / grayscale PNG')
