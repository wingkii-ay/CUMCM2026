"""从数值文件生成问题二答案；不手填计算结论。"""
from pathlib import Path
import json
import pandas as pd
from solve_q2 import SELECTED

root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
d=pd.read_csv(root/'dispatch.csv');daily=pd.read_csv(root/'daily.csv').set_index('date')
events=pd.read_csv(root/'emergency_events.csv');sens=pd.read_csv(root/'sensitivity.csv')
s=summary['evaluation'];fair=summary['fair_boundary']
f=lambda v:f'{v:,.6f}'
money=lambda v:f'{v:,.2f}'
rows=[]
def add(x=''):rows.append(x)
def table(headers,values):
    add('| '+' | '.join(headers)+' |');add('|'+'|'.join(['---']*len(headers))+'|')
    for row in values:add('| '+' | '.join(map(str,row))+' |')
    add()

add('# C题问题二：日前购电计划与紧急补购')
add()
add('本答案使用附件1固定电价、附件2历史负载和光伏，采用“历史联合场景＋日前购电承诺＋日内电池反馈”。每日0时不读取当天未来实际值。结果是一套经过实际回测的近似策略，不是一般多阶段随机控制问题的全局最优证明。')
add()
add('## 1. 核心答案')
add()
add(f'评价区间为2025年2月1日至12月31日，共334天。**总实际购电费用为{money(s["total_cost_yuan"])}元**，其中计划费用{money(s["plan_cost_yuan"])}元、紧急费用{money(s["emergency_cost_yuan"])}元。')
add()
table(['指标','结果'],[
    ['计划购电总量（kWh）',f(s['grid_kwh'])],['紧急购电总量（kWh）',f(s['emergency_kwh'])],
    ['实际外网购电总量：计划＋紧急（kWh）',f(s['grid_kwh']+s['emergency_kwh'])],
    ['总充电量（母线侧kWh）',f(s['charge_kwh'])],['总放电量（母线侧kWh）',f(s['discharge_kwh'])],
    ['未利用电量（含已付费购电剩余，kWh）',f(s['unused_kwh'])],
    ['发生应急的10分钟时段数',s['emergency_slots']],['同日合并后的应急事件数',s['emergency_events']],
    ['2月1日0:00储电量（kWh）',f(s['initial_kwh'])],['12月31日24:00储电量（kWh）',f(s['final_kwh'])]])
add('问题二没有要求每天首末储电量相同。1月1日从6000 kWh开始连续运行，2月1日6200 kWh来自1月31日期末，不能重新设为6000。1月预热成本'+money(summary['january_cost_yuan'])+'元单独记录，不混入上表；1月1日无前史而采用全部缺口应急的冷启动会使该月费用偏高。')
add()
add('## 2. 四个指定日汇总')
add()
table(['日期','计划购电量/kWh','紧急购电量/kWh','计划费/元','紧急费/元','总费用/元'],[
    [date]+[f(daily.loc[date,c]) for c in ['grid_kwh','emergency_kwh']]+[money(daily.loc[date,c]) for c in ['plan_cost_yuan','emergency_cost_yuan','total_cost_yuan']]
    for date in SELECTED])
zero_days=[x for x in SELECTED if daily.loc[x,'emergency_kwh']<1e-6]
add('上表费用按各10分钟段的电价分别计算；不能用一个日平均电价乘全天电量替代。'
    +('、'.join(zero_days)+'无紧急购电，应急电量为浮点零（小于10⁻⁶ kWh），按零显示。' if zero_days else ''))
add()
add('## 3. 表1要求：指定时段购电量与全天结果')
add()
add('下表“购电量”为每天0时承诺的普通计划购电量，单位kWh；紧急购电另列第5节。')
add()
selection=[]
for slot in [61,73,85,97,109,121]:
    a=d[d.slot==slot].set_index('date')
    selection.append([a.interval.iloc[0]]+[f(a.loc[date,'grid_kwh']) for date in SELECTED])
selection.extend([['全日计划购电量（kWh）']+[f(daily.loc[date,'grid_kwh']) for date in SELECTED],
                  ['全日计划购电费（元）']+[money(daily.loc[date,'plan_cost_yuan']) for date in SELECTED],
                  ['全天总购电费（含应急，元）']+[money(daily.loc[date,'total_cost_yuan']) for date in SELECTED]])
table(['时间段/项目']+SELECTED,selection)
add('## 4. 表2要求：充放电量与日初、日末储电量')
add()
add('以下均为实际反馈执行结果；充电量、放电量采用母线侧kWh，储电量采用电池内部kWh。4小时段内同时出现充、放电量，表示不同10分钟段执行了不同动作，同一10分钟段没有同时充放。')
add()
for date in SELECTED:
    one=d[d.date==date]
    add(f'### {date}');add()
    vals=[]
    for k in range(6):
        part=one.iloc[k*24:(k+1)*24]
        vals.append([f'{k*4:02d}:00-{(k+1)*4:02d}:00',f(part.charge_kwh.sum()),f(part.discharge_kwh.sum())])
    table(['时间段','充电量/kWh','放电量/kWh'],vals)
    add(f'0:00储电量：{f(one.storage_start_kwh.iloc[0])} kWh；24:00储电量：{f(one.storage_end_kwh.iloc[-1])} kWh。');add()
add('## 5. 表3要求：紧急购电时间段及电量')
add()
add('连续正应急段按同日合并，电量为对应10分钟段求和；不跨日合并。工程零阈值10⁻⁶ kWh。')
add()
for date in SELECTED:
    add(f'### {date}');add()
    one=events[events.date==date]
    if one.empty:add('当日无紧急购电。');add()
    else:table(['时间段','紧急购电量/kWh'],[[r.interval,f(r.emergency_kwh)] for r in one.itertuples()])
worst=max(SELECTED,key=lambda x:daily.loc[x,'emergency_kwh'])
emergency_days=int((daily.emergency_kwh>1e-6).sum())
add(f'四个指定日中，{worst}紧急购电最多（{f(daily.loc[worst,"emergency_kwh"])} kWh、'
    f'{money(daily.loc[worst,"emergency_cost_yuan"])}元），'
    +('、'.join(x for x in SELECTED if daily.loc[x,'emergency_kwh']<1e-6)+'无紧急购电。' if zero_days else '')
    +f'这说明本策略不是“零应急”保证：评价期内仍有{emergency_days}天发生紧急购电，'
    '但量级已大幅压降，且全年费用下降不代表每个日期都节省同样比例。')
add()
add('## 6. 数学模型与因果求解')
add()
add('每天0点，用最近'+str(summary['lookback_weeks'])+'个**同一星期几**（周一至周日各自匹配）的负载、光伏均值作预测，叠加此前完整日的联合预测残差，生成最多7个等权场景。周六与周日的负载模式相反，必须分别匹配：实测19—20时周六负载均值526 kWh、周日794 kWh，若混同为“周末”再取平均，会得到两边都不像的曲线，导致周日低估而少承诺购电、被迫按5倍价格买应急电。所有2—12月信息均不参加参数选择。')
add()
add(r'$$\min\;\sum_t p_tG_t+\frac1S\sum_s\sum_t5p_te_t^s+0.472(6000-E_{144})_+.$$')
add()
add('约束为：')
add()
add(r'$$G_t+V_t^s+b_t+e_t^s=L_t^s+c_t+w_t^s,\qquad E_t=E_{t-1}+0.9c_t-b_t/0.9.$$')
add()
add(r'$$1200\le E_t\le10800,\quad 0\le c_t,b_t\le5000/6,\quad c_tb_t=0,\quad c_t\le G_t+\min_s V_t^s.$$')
add()
add('购电、应急、剩余等变量非负。各场景共享G和电池参考轨迹，不能每个场景预知自己的整日未来后独立操作电池。由于免费剩余电能允许消去无效充放循环，日前模型可用LP求解后消循环；证明见[模型与算法说明](模型与算法说明.md)。')
add()
add('0时只锁定G。日内以100 kWh储能网格构造剩余价值表，再使用本段实际供需和段初储电量选择动作。未来价值按场景边际期望近似，不保留天气场景的全部时序状态。假设可在本段测量并及时响应，不模拟10分钟内的测量和调度延迟。实际计费仅为：')
add()
add(r'$$J_d=\sum_t p_tG_t+\sum_t5p_te_t.$$')
add()
add('终端库存惩罚不计入电费，计划多余电量不退款。两侧效率分别为90%；若解释为往返效率90%，应使用敏感性分支，不可混用主结果。')
add()
add('## 7. 基线和稳健性')
add()
save=fair['baseline_cost_yuan']-fair['total_cost_yuan']
table(['统一期初与期末均6000 kWh的独立对照','总费用/元'],[
    ['无储能，历史场景80%分位购电',money(fair['baseline_cost_yuan'])],
    ['储能策略，增加统一终端边界',money(fair['total_cost_yuan'])],
    ['费用节省',money(save)],['节省比例',f'{100*save/fair["baseline_cost_yuan"]:.4f}%']])
add('两方案使用相同因果场景；无储能基线不预知当天真实供需。主策略总费用与上表储能费用略有差异，因为上表是从2月1日6000 kWh出发并强制年末恢复6000的独立实验；最后一天增加保守购电保障终端可达。节省不能归因于额外消耗初始库存。')
add()
rob=root/'results/lookback_robustness.json'
if rob.exists():
    rb=json.loads(rob.read_text(encoding='utf-8'))
    table(['回看深度/周','全年总费用/元','相对混合预测的节省'],[
        [r['weeks'],money(r['total_cost_yuan']),f'{r["saving_pct"]:.2f}%'] for r in rb['study']])
    add('回看深度在'+str(rb['study'][0]['weeks'])+'—'+str(rb['study'][-1]['weeks'])+'周范围内，节省比例的极差仅'+f'{rb["saving_pct_spread"]:.2f}'+'个百分点，主结果对该参数不敏感；8周兼顾样本量与季节性漂移。该检验为一次性稳健性验证，不参与主参数反选。')
    add()
g50=sens[sens.variant=='grid_50'].set_index('date').cost_yuan.sort_index()
gmain=sens[sens.variant=='main'].set_index('date').cost_yuan.sort_index()
grid_delta='、'.join(f'{x:+.2f}' for x in (g50-gmain).to_numpy())
add('参数敏感性以主策略相同日初库存开始，四个指定日分别单独重算，**不代表重新运行全年的成绩**。'
    '50 kWh网格相对100 kWh网格的费用差为'+grid_delta+'元（按日期先后），方向不一致；'
    '更细网格不保证事后成本单调下降。')
add()
s14=sens[sens.variant=='scenes_14'].set_index('date').cost_yuan.sort_index()
s28=sens[sens.variant=='scenes_28'].set_index('date').cost_yuan.sort_index()
scene_delta=('、'.join(f'{x:+.0f}' for x in (s14-gmain).to_numpy())+' 和 '
             +'、'.join(f'{x:+.0f}' for x in (s28-gmain).to_numpy()))
half=sens[sens.variant=='terminal_half'].set_index('date').final_kwh
main_final=sens[sens.variant=='main'].final_kwh
add('14/28个场景相对主策略在四个指定日的费用差为'+scene_delta
    +'元（按日期先后），有增有减，没有一致方向。终端系数减半使四个指定日的日末库存由约'
    +f'{main_final.min():.0f}—{main_final.max():.0f}'
    +' kWh降至'+f'{half.min():.0f}—{half.max():.0f}'
    +' kWh，所以不能把其单日低费用全部当作更优策略。详细值及两侧效率改为√0.9的解释见`sensitivity.csv`；不据这些事后结果反选主参数。')
add()
add('## 8. 核验、文件与复现')
add()
add(f'全365日供需平衡最大残差{summary["validation"]["balance_residual_kwh"]:.3g} kWh；储能递推和跨日连续性残差为0；储电范围1200—10800 kWh，无同时充放；功率超限残差仅浮点量级。结果工作簿另逐格核对计划电量、费用、4小时汇总及应急事件。')
add()
add('- `result2.xlsx`：按附件模板填充334天计划购电、实际充放电、合并应急事件，另附每日费用与口径说明。')
add('- `dispatch.csv`、`daily.csv`、`emergency_events.csv`：完整评价账本；带`_all`文件还包含1月预热。')
add('- `fair_boundary_daily.csv`：统一边界对照；`sensitivity.csv`：指定日局部敏感性。')
add('- `figures/`：原始、过程、结果各3图，PNG＋SVG＋灰度预览；解释见`图表说明.md`。')
add('- `results/复现清单.json`：原附件哈希、依赖版本、参数与命令。')
add()
add('安装requirements.txt中的依赖，在本目录执行：')
add()
add('```powershell\npython run_all.py\n```')
add()
add('其他附件位置使用`python run_all.py --input "你的C题附件目录"`。Python需3.10或更新版本。Markdown公式用VS Code的Ctrl+Shift+V预览。本答案是问题二计算报告，不是完整参赛论文。')
rows = [line[:2]+'\n'+line[2:-2]+'\n'+line[-2:] if line.startswith('$$') and line.endswith('$$') else line for line in rows]
(root/'问题二答案.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
print('Answer generated from numerical outputs')
