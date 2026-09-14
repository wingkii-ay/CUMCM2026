"""由真实账本生成答案和指定日期三张表。"""
from pathlib import Path
import json
import pandas as pd
from battery_core import SELECTED
R=Path(__file__).resolve().parent
s=json.loads((R/'summary.json').read_text(encoding='utf-8'));e=s['evaluation']
d=pd.read_csv(R/'dispatch.csv');daily=pd.read_csv(R/'daily.csv').set_index('date');events=pd.read_csv(R/'emergency_events.csv')
p=pd.read_csv(R/'policy_comparison.csv');fair=p[p.fair_boundary].set_index('policy');boot=pd.read_csv(R/'bootstrap_comparisons.csv');sens=pd.read_csv(R/'sensitivity.csv')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
# 金额到分、电量到瓦时，与result3.xlsx的显示格式一致；全精度会给出0.000001元这种无意义位数。
def fy(x):return f'{x:,.2f}'   # 元
def fk(x):return f'{x:,.3f}'   # kWh
labels={'only0':'仅0时','add6':'0/6时','add12':'0/6/12时','all':'0/6/12/18时全部更新','stale_all':'四次决策均用0时预报','stale18':'18时沿用12时预报'}
text=['# C题问题三答案','',
f'采用“0时计划＋6/12/18时滚动调整＋10分钟储能反馈”策略。2025年2月1日至12月31日334天实际总购电费用为 **{e["total_cost_yuan"]:,.2f}元**。这是所构造可行策略的回放结果，不是一般多阶段问题的全局最优值。',
'','## 一、模型与口径','',
'每天144个10分钟区间，t=1,…,144；内部储能E₀,…,E₁₄₄共145个边界状态。功率乘1/6换成区间电量。G是0时原计划，A是执行前最终普通购电总量，u/v是相对G的增/减购量，e是紧急购电，c/b是母线侧充/放电，w是未利用电量。','',
r'$$',r'\min\ \mathbb E[J]=\mathbb E\!\left[\sum_{t=1}^{144}(p_tG_t+1.5p_tu_t+0.5p_tv_t+5p_te_t)\right],',r'$$','',
r'$$',r'u_t=(A_t-G_t)_+,\quad v_t=(G_t-A_t)_+,\quad A_t+V_t+b_t+e_t=L_t+c_t+w_t,',r'$$','',
r'$$',r'E_t=E_{t-1}+0.9c_t-b_t/0.9,\quad1200\le E_t\le10800,',r'$$','',
r'$$',r'0\le c_t\le(5000/6)z_t,\quad0\le b_t\le(5000/6)(1-z_t),\quad z_t\in\{0,1\},',r'$$','',
r'$$',r'G_t,A_t,e_t,w_t\ge0,\qquad c_t\le A_t+V_t.',r'$$','',
'仅1月1日E₀=6000 kWh，之后跨日连续；1月预热不计入费用评价。0时制定全天G，6/12/18时仅调整未执行区间。预报整点值线性插值并积分到10分钟；负载使用最近8个同一星期几（周一至周日各自匹配）的均值，场景使用过去最多7个完整日联合残差。没有使用尚未发布预报或未来实际值。','',
'**主结算假设：原计划全额支付、每段最终A相对原G的净差仅结算一次。** 因允许免费剩余，A<G被A=G支配，因此主结果不主动减少原计划；可以撤回此前的额外增购。减购退还原电费再收50%违约费的另一解释，另列为敏感性口径。','',
'0时的G由**两阶段随机规划**给出：G是全时段共享的第一阶段承诺；可调整时段（6时及以后）的最终购电A^s≥G，超出部分按1.5p补购；0—6时没有发布节点，该段A冻结为G，缺口只能按5p应急。因此0时计划内部已计入“事后可按1.5p补购”这一选择权，而不是把缺口一律按5p定价。**非对称结算下少承诺比多承诺便宜**（计划内多余电量已付p且不退款，缺额只需补0.5p），最优G落在实际购电约1/3分位，而不是点预测的中位数；这正是问题三与问题二在决策上的实质差别。该两阶段模型仍不是完整多阶段情景树，不声称全局最优。','',
'求解采用共享场景线性规划、消除充放电循环和100 kWh网格价值反馈；终端不足惩罚0.472(6000−E₁₄₄)₊元仅用于决策，不计入实际电费。价值表以场景边际期望近似，按各场景的期望实际购电扣减可购电量。当前10分钟平均供需可即时测得属于反馈假设。**两条结构性限制须一并说明**：一是终端惩罚系数κ=p[0]/η=0.472元/kWh与场景和日期都无关（评价期334天的p[0]=0.4248完全相同），跨日仅靠这一个常数传递，明日信息没有接入通道；二是附件3的每个发布时刻本是自发布起24小时的预报，求解时按当日截断存储，18时预报中明日00:00—18:00的18小时不参与任何决策。两者对结果的影响见《优化方案》§4.2。详细推导和边界见《模型与算法说明》，优化前后的诊断与方案记录见《优化方案》。','',
'完美预知（全时段负载/光伏/电价完全已知）联合LP给出同期同期末库存下的下界12,230,108.36元，本策略距该**不可达**下界9.91%；优化前该差距为26.97%，本次优化回收了其中70.2%。下界与主策略的差额完全由1.5p补购费和5p应急费构成，即信息的价格。','',
'## 二、评价期费用','',table(['项目','结果'],[[k,(fy(e[c]) if c.endswith('_yuan') else fk(e[c]))] for k,c in [('原计划费用 / 元','plan_cost_yuan'),('增加购电费用 / 元','increase_cost_yuan'),('减少购电违约费用 / 元','decrease_cost_yuan'),('紧急购电费用 / 元','emergency_cost_yuan'),('总费用 / 元','total_cost_yuan'),('原计划电量 / kWh','original_grid_kwh'),('最终普通购电量 / kWh','adjusted_grid_kwh'),('紧急购电量 / kWh','emergency_kwh'),('评价期初储能 / kWh','initial_kwh'),('评价期末储能 / kWh','final_kwh')]]),'',
'## 三、指定日期结果','',
'金额以元计，电量以kWh计。表1给出0时计划G；调整后的普通购电A另表列出，应急不并入A。完整144段计划、版本和实际执行量见result3.xlsx及CSV。','',
table(['日期','原计划费用','调整相关费用','紧急费用','总费用'],[[date,fy(daily.loc[date,'plan_cost_yuan']),fy(daily.loc[date,'increase_cost_yuan']+daily.loc[date,'decrease_cost_yuan']),fy(daily.loc[date,'emergency_cost_yuan']),fy(daily.loc[date,'total_cost_yuan'])] for date in SELECTED]),'',
table(['日期','全天原计划G','全天最终普通购电A','全天紧急购电e','全天实际外网供电A+e'],[[date,fk(daily.loc[date,'original_grid_kwh']),fk(daily.loc[date,'adjusted_grid_kwh']),fk(daily.loc[date,'emergency_kwh']),fk(daily.loc[date,'adjusted_grid_kwh']+daily.loc[date,'emergency_kwh'])] for date in SELECTED]),'',
'### 表1 指定日期0时计划购电量','',table(['时间段']+SELECTED,[[d[d.date==SELECTED[0]].iloc[t].interval]+[fk(d[d.date==date].iloc[t].original_grid_kwh) for date in SELECTED] for t in range(144)]),'',
'### 表1补充 最终有效普通购电量','',table(['时间段']+SELECTED,[[d[d.date==SELECTED[0]].iloc[t].interval]+[fk(d[d.date==date].iloc[t].adjusted_grid_kwh) for date in SELECTED] for t in range(144)]),'',
'### 表2 指定日期充放电量与储电量','']
rows=[]
for k in range(6):
    row=[f'{4*k:02d}:00—{4*k+4:02d}:00']
    for date in SELECTED:
        z=d[d.date==date].iloc[k*24:(k+1)*24];row += [fk(z.charge_kwh.sum()),fk(z.discharge_kwh.sum())]
    rows.append(row)
text += [table(['时间段']+[date+' '+kind for date in SELECTED for kind in ['充电','放电']],rows),'',table(['边界时刻']+SELECTED,[[name]+[fk(daily.loc[date,c]) for date in SELECTED] for name,c in [('00:00','initial_kwh'),('24:00','final_kwh')]]),'','### 表3 指定日期紧急购电量','']
groups=[events[events.date==date].reset_index(drop=True) for date in SELECTED]
text += [table([date+' '+kind for date in SELECTED for kind in ['时间段','电量']],[[item for g in groups for item in ([g.iloc[i].interval,fk(g.iloc[i].emergency_kwh)] if i<len(g) else ['—','—'])] for i in range(max(1,max(map(len,groups))))]),'','“—”表示该日期无对应应急事件；连续正应急区间合并，电量累计。','',
'## 四、是否需要其他时刻预报','',
'下表单独从2月1日6000 kWh起算，12月31日恢复6000 kWh，消除起止储能差；它不替换上述跨1月连续的主结果。所有策略均每10分钟作储能反馈。','',table(['策略','公平边界总费用 / 元','紧急电量 / kWh'],[[labels[k],fy(row.total_cost_yuan),fk(row.emergency_kwh)] for k,row in fair.iterrows()]),'',
'配对差为“前一策略费用−后一策略费用”，正值表示后一策略省钱。区间为334个日费用差的循环分块bootstrap（7天块、2000次、种子2026）的平均日节省95%区间。','',
table(['比较','全年节省 / 元','平均每日节省 / 元','95%区间 / 元/日'],[[labels[z.before]+' → '+labels[z.after],fy(z.total_saving_yuan),fy(z.mean_saving_yuan_per_day),f'[{z.ci_low_yuan_per_day:.2f}, {z.ci_high_yuan_per_day:.2f}]'] for z in boot[boot.block_days==7].itertuples()]),'']
def _b(before,after):
    return boot[(boot.before==before)&(boot.after==after)&(boot.block_days==7)].iloc[0]
for before,after in [('only0','all'),('stale_all','all'),('stale18','all')]:
    z=_b(before,after)
    text += [f'从“{labels[before]}”到“{labels[after]}”，评价期费用变化对应节省{z.total_saving_yuan:,.2f}元，日均{z.mean_saving_yuan_per_day:,.2f}元；'+('区间完全为正，支持本数据下的节省效果。' if z.ci_low_yuan_per_day>0 else '区间包含零或为负，不能据此宣称稳定的正收益。'),'']
sa=_b('stale_all','all');s18=_b('stale18','all')
text += [f'**回答“是否需要”：本数据支持使用日内发布的更新预报。** 在同样四次决策机会下，四次都用0时预报的对照比全更新多付{sa.total_saving_yuan:,.2f}元，日均{sa.mean_saving_yuan_per_day:,.2f}元，'
         f'95%区间[{sa.ci_low_yuan_per_day:.2f}, {sa.ci_high_yuan_per_day:.2f}]完全为正——这是唯一隔离“新预报信息增益”的比较，因此新预报确有经济价值。'
         f'18时新预报相对12时则没有稳定的额外节省（日均{s18.mean_saving_yuan_per_day:,.2f}元、区间[{s18.ci_low_yuan_per_day:.2f}, {s18.ci_high_yuan_per_day:.2f}]跨零），可据此建议保留6/12时更新、18时视接入成本而定。'
         '但**最后这一条不能读作“18时的预报没有价值”**，它是本模型结构造成的，不是数据性质的结论：附件3给出的每个发布时刻都是自发布起24小时的预报，而求解时把每个发布时刻的预报截断在当日，18时预报里明日00:00—18:00那18小时从不进入任何决策；跨日又只靠常数终端价值传递，明日信息本就没有接入通道。也就是说这个零结果反映的是“模型没有用它的通道”，而非“它没有信息”。取消该截断、把终端价值换成依赖明日预报的值函数，属于另一套模型，本文未做，列为《优化方案》§4.2 登记的结构限制。'
         '仅0时方案没有任何发布节点可供补购，其0时计划只能按5倍应急自给，是被结算规则逼出的保守解，同样不能读作“预报无用”。这里没有另行训练自适应接入规则，也不将事后最低费用策略称为未来最优策略。','',
'前三组嵌套比较同时增加“调整机会”和“新信息”两项，且只加6时相对仅0时的增量区间跨零，不能单独用来证明预报价值；“四次决策均用0时预报”与全更新的比较才隔离日内新预报的信息增益。18时专门旧预报对照则隔离18时新信息，不能把同一时刻重新观察电池状态的收益误称为新光伏预报收益。结论限于本年数据、本策略与所述结算口径；跨年表现未验证。','',
'## 五、敏感性与验证','',table(['日期','设置','总费用 / 元','期末储能 / kWh'],[[z.date,z.variant,fy(z.total_cost_yuan),fk(z.final_kwh)] for z in sens.itertuples()]),'',
'这些是四个日期固定主策略日初库存的局部重算，不是全年新策略结果；期末库存可不同，不能仅凭单日费用给敏感性设置排名。`terminal_half`行把终端价值减半，日末库存被压到1,243—2,656 kWh，其单日低费用是提前耗尽库存换来的，不是更优策略。','',
'`refund`行（式(11)结算）必须与同行的`main_legacyplan`（同为问题二口径0时计划、同为式(10)结算，只差结算公式）配对比较，退款口径相对主口径的差额为 '
 + '、'.join(f"{date} {sens[(sens.date==date)&(sens.variant=='refund')].total_cost_yuan.iloc[0]-sens[(sens.date==date)&(sens.variant=='main_legacyplan')].total_cost_yuan.iloc[0]:,.2f}元" for date in SELECTED)
 + '；退款口径改变合同，不能与主口径直接解释成算法改进。3/7/14天块长的不确定性结果另存bootstrap_comparisons.csv。','',
f'主路径最大电量约束残差为{max(value for key,value in s["validation"].items() if key.endswith("_kwh")):.3g} kWh，功率上限数值超差为{s["validation"]["power_kw"]:.3g} kW。全部48,096条评价期记录、120,240条版本记录及Excel计划、调整、充放电与应急金额由独立脚本逐项核验。','',
'## 六、文件与复现','',
'- result3.xlsx：按附件模板填写的完整结果，另附每日费用与口径说明。\n- dispatch.csv / daily.csv：334天逐段账本和日汇总；带_all文件包括1月预热。\n- schedule_versions.csv：每次决策的剩余计划快照。\n- policy_comparison.csv / fair_*_daily.csv：六种策略和公平边界对照。\n- figures：图19—27，PDF、SVG、1200 DPI PNG；灰度校验版在 figures/_qa/。\n- run_all.py：唯一复现入口；执行 python run_all.py，可用 --input 指定原始附件目录。\n- bound_perfect_foresight.py / results/perfect_foresight_bound.json：完美预知下界及其求解明细。\n- 优化方案.md：优化前后的诊断、已实施方案①（精确星期几预测）、方案②（0时两阶段随机规划）、已否决方案③（平滑G锯齿）与结论翻转的完整记录。\n- results：数值核验和输入哈希、环境、参数的复现清单。']
(R/'问题三答案.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print('问题三答案.md generated from verified data')
