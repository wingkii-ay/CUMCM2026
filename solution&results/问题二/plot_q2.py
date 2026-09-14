"""问题二：以实际结果生成三类九图，PDF/SVG与1200 DPI PNG配套导出。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.publication_style import setup, create, save as save_publication, PALETTE


ROOT=Path(__file__).resolve().parent
d=pd.read_csv(ROOT/'dispatch.csv');daily=pd.read_csv(ROOT/'daily.csv')
fair=pd.read_csv(ROOT/'fair_boundary_daily.csv')
setup()
figures=ROOT/'figures';figures.mkdir(exist_ok=True)
def save(fig,name):save_publication(fig,figures,name)

def legend(ax,ncol=2):
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.08),ncol=ncol,fontsize=9)

def hours(ax):
    ax.set_xlim(0,24);ax.set_xticks(np.arange(0,25,4))

def months(ax):
    ax.set_xlim(pd.Timestamp('2025-02-01'),pd.Timestamp('2025-12-31'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2,4,6,8,10,12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))

# 数据三图：季节变化、分布、价格与净需求对应关系。
raw=d.groupby('date')[['load_kwh','pv_kwh']].sum()
fig,ax=create('每日负载与光伏电量','2025年','电量（MWh/日）')
ax.plot(pd.to_datetime(raw.index),raw.load_kwh/1000,color=PALETTE['primary'],label='负载',lw=1)
ax.plot(pd.to_datetime(raw.index),raw.pv_kwh/1000,color=PALETTE['secondary'],ls='--',label='实际光伏',lw=1)
legend(ax);months(ax);save(fig,'raw_q2_daily_energy')

fig,ax=create('净负载包含富余与缺口时段','净负载功率（kW）','10分钟时段数')
ax.hist((d.load_kwh-d.pv_kwh)*6,bins=45,color=PALETTE['primary'],edgecolor='white')
ax.axvline(0,color=PALETTE['neutral'],ls='--',lw=1)
save(fig,'raw_q2_netload_distribution')

average=d.groupby('slot')[['load_kwh','pv_kwh','price_yuan_per_kwh']].mean()
fig,ax=create('固定电价与平均净负载','各时段平均净负载（kW）','电价（元/kWh）')
ax.scatter((average.load_kwh-average.pv_kwh)*6,average.price_yuan_per_kwh,s=18,color=PALETTE['primary'])
ax.axvline(0,color=PALETTE['neutral'],ls='--',lw=.8)
save(fig,'raw_q2_price_netload')

# 过程三图：同一指定日的预测偏差、储能修正与实际动作，分别检验三层机制。
one=d[d.date=='2025-03-20'];t=(one.slot.to_numpy()-.5)/6
fig,ax=create('3月20日净负载预测与实际','时刻（h）','净负载功率（kW）')
ax.plot(t,(one.load_kwh-one.pv_kwh)*6,color=PALETTE['observed'],label='实际')
ax.plot(t,(one.forecast_load_kwh-one.forecast_pv_kwh)*6,color=PALETTE['f0'],ls='--',label='0时历史预测')
legend(ax);hours(ax);save(fig,'process_q2_forecast_error')

fig,ax=create('3月20日储能参考与实际轨迹','时刻（h）','内部储电量（kWh）')
ax.plot(np.arange(145)/6,np.r_[one.storage_start_kwh.iloc[0],one.storage_end_kwh],color=PALETTE['observed'],label='实际反馈')
ax.plot(np.arange(145)/6,np.r_[one.storage_start_kwh.iloc[0],one.reference_storage_kwh],color=PALETTE['f0'],ls='--',label='日前参考')
for y in [1200,10800]:ax.axhline(y,color=PALETTE['neutral'],ls=':',lw=.8)
ax.set_ylim(0,12000);legend(ax);hours(ax);save(fig,'process_q2_storage_feedback')

fig,ax=create('3月20日实际充放电动作','时刻（h）','母线侧功率（kW）')
ax.bar(t,one.charge_kwh*6,width=1/6,color=PALETTE['f0'],label='充电')
ax.bar(t,-one.discharge_kwh*6,width=1/6,color=PALETTE['f6'],label='放电（负向）')
ax.axhline(0,color='black',lw=.6);ax.set_ylim(-5500,5500)
legend(ax);hours(ax);save(fig,'process_q2_actions')

# 结果三图：月度费用构成、应急发生时段、统一库存边界累计成本对照。
daily['month']=pd.to_datetime(daily.date).dt.month
monthly=daily.groupby('month')[['plan_cost_yuan','emergency_cost_yuan']].sum()/10000
fig,ax=create('购电费用的月度构成','2025年月序','费用（万元）')
ax.bar(monthly.index,monthly.plan_cost_yuan,color=PALETTE['f0'],label='计划费用')
ax.bar(monthly.index,monthly.emergency_cost_yuan,bottom=monthly.plan_cost_yuan,color=PALETTE['f18'],label='紧急费用')
ax.set_xticks(range(2,13));legend(ax);save(fig,'result_q2_monthly_cost')

fig,ax=create('紧急购电的日期与时段','时刻（h）','日期（2025年）')
fig.subplots_adjust(right=.81)
matrix=d.emergency_kwh.to_numpy().reshape(-1,144)
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
classes=np.digitize(matrix,[1e-6,50,200,500],right=True)
# 用橙—朱红顺序色阶替代原来的灰阶：亮度单调递减，灰度打印下仍可区分，
# 但不再是整片黑灰，且与全文的 Okabe-Ito 主色同源。
urgent=LinearSegmentedColormap.from_list('urgent',['#FFFFFF','#FCE0BC','#E69F00','#D55E00','#7A2E00'],N=5)
im=ax.pcolormesh(np.arange(145)/6,np.arange(len(matrix)+1),classes,
                 cmap=urgent,
                 norm=BoundaryNorm(np.arange(-.5,5.5,1),5),shading='flat',rasterized=False)
ax.set_xlim(0,24);ax.set_ylim(len(matrix),0)
starts=daily.groupby('month',sort=True).head(1).index.to_numpy()
ax.set_yticks(starts+.5,[f'{m:02d}-01' for m in range(2,13)])
ax.set_xticks(range(0,25,4))
cb=fig.colorbar(im,ax=ax,fraction=.065,pad=.035);cb.set_label('紧急电量（kWh/10分钟）',fontsize=9)
cb.set_ticks(range(5),labels=['0','(0,50]','(50,200]','(200,500]','>500'])
cb.solids.set_rasterized(False)
save(fig,'result_q2_emergency_map')

fig,ax=create('相同起止库存下的累计费用','2025年','累计费用（万元）')
ax.plot(pd.to_datetime(fair.date),fair.baseline_cost_yuan.cumsum()/10000,color=PALETTE['neutral'],ls='--',label='无储能基线')
ax.plot(pd.to_datetime(fair.date),fair.total_cost_yuan.cumsum()/10000,color=PALETTE['f0'],label='储能策略（公平边界）')
months(ax);legend(ax);save(fig,'result_q2_fair_cost')
print('9 logical figures exported with PDF, SVG, 1200 DPI PNG and grayscale',flush=True)
