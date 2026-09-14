"""Generate the Chinese answer and nine evidence figures from computed outputs."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from core import INPUT,SELECTED,DATES,read_data,predictions
ROOT=Path(__file__).resolve().parent

def skill_root():
    for p in [Path.home()/'.claude/skills/math-modeling',Path.home()/'.codex/skills/math-modeling',
              Path(r'C:\Users\NGALAM\.codex\skills\math-modeling')]:
        if (p/'tools').is_dir():return p
    raise RuntimeError('未找到 math-modeling skill 目录，请在 skill_root() 中补上本机路径')
SKILL=skill_root()

def table(headers,rows):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:,.6f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(fmt,row))+' |' for row in rows])+'\n\n'

def answer(out):
    r=out/'results';s=json.loads((r/'summary.json').read_text(encoding='utf-8'));bounds=json.loads((r/'bounds.json').read_text(encoding='utf-8'))
    cmp=pd.read_csv(r/'comparisons.csv');sens=pd.read_csv(r/'sensitivity.csv')
    text='# C题问题四答案：波动电价下的微网购电与储能调度\n\n'
    text+='**主口径：未来电价未知，按供电时段实际电价浮动结算。** 每天0时预测并提交普通购电；问题四对应问题三的分支可在6/12/18时调整。原计划不退款，最终调整净差计费一次。本文是可复现的近似策略回测答案，不是全局最优证明。\n\n'
    text+='## 一、直接结果\n\n统计2025年2月1日至12月31日334天；1月预热另外保存。两分支均从1月1日6000kWh连续运行，不每日重置。\n\n'
    rows=[]
    for b in [2,3]:
        d=pd.read_csv(r/f'q4{b}_all_daily.csv')
        rows.append([f'Q4-{b}',d.plan_cost_yuan.sum(),d.increase_cost_yuan.sum(),d.emergency_cost_yuan.sum(),d.total_cost_yuan.sum(),d.emergency_kwh.sum()])
    text+=table(['分支','原计划费用/元','增购费用/元','应急费用/元','实际总费用/元','应急电量/kWh'],rows)
    for b in [2,3]:
        z=s[f'q4{b}_all'];text+=f'Q4-{b}评价期初库存{z["initial"]:,.6f}kWh，期末{z["final"]:,.6f}kWh。\n\n'
    diff=s['q43_all']['cost']-s['q42_all']['cost'];drop=s['q42_all']['emergency']-s['q43_all']['emergency']
    text+=f'本次Q4-3相对Q4-2总费用{("增加" if diff>=0 else "减少")}{abs(diff):,.2f}元，应急电量减少{drop:,.2f}kWh。**减少应急并不必然减少总费用**：0时使用的光伏预测、普通承诺量和增购费用均会影响总账，不能只比较应急费。\n\n'
    if diff>0:text+='Q4-3允许忽略额外信息和调整机会，因此在相同边界下，其真正最优值不应高于可嵌入的Q4-2策略。当前费用更高说明本共享轨迹、保守日前计划近似未充分利用选择权；这是策略性能局限，不是新增信息必然有害的证明。不能事后按全年费用挑选策略，再声称其是预先确定的在线最优解。\n\n'
    else:text+='Q4-3的可行集包含Q4-2（取A=G即可），故其真正最优值不应高于Q4-2，本次结果方向与问题三一致。此前的版本在0时对**所有**时段的场景缺额一律按5p应急计价，等价于把承诺抬到1−1/5=80%分位；而合同下节点后可增购的边际价是1.5p、对应最优分位1/1.5≈67%，于是系统性多承诺，多出的承诺又因"A不得低于G"无法回退，只能计入未利用电量。现按1.5p给**节点之后**的缺额定阶（节点之前的时段无发布节点，仍按5p），0时计划与节点实际面临的价格一致。修正只改0时计划的缺额系数，不动实时层、调整层与账本口径，因果性与物理约束复核不变。\n\n'
    text+='## 二、模型与算法\n\n令Δ=1/6小时，负载和光伏电量为L=ΔPᴸ、V=ΔPⱽ，G为0时普通承诺，A为最终普通购电，e为应急，c/b为母线充/放电，E为内部储能。\n\n'
    text+=r'$$E_t=E_{t-1}+0.9c_t-b_t/0.9,\quad 1200\le E_t\le10800,\quad 0\le c_t,b_t\le5000/6,\quad c_tb_t=0.$$'+'\n\n'
    text+=r'$$A_t+V_t+b_t+e_t=L_t+c_t+w_t,\quad w_t\ge0.$$'+'\n\n'
    text+='Q4-2令A=G，实际结算为\n\n'+r'$$J_{4-2}=\sum_t p_t(G_t+5e_t).$$'+'\n\nQ4-3令u=(A−G)₊、v=(G−A)₊，实际结算为\n\n'+r'$$J_{4-3}=\sum_t p_t[G_t+1.5u_t+0.5v_t+5e_t].$$'+'\n\n'
    text+='其中p来自执行时附件4，规划时用预测场景pˢ。原计划费用使用E[pˢ]G，应急项使用E[5pˢeˢ]，不能用E[pˢ]E[eˢ]替代后者，避免丢失价格与缺口相关性。终端罚κ(6000−E_T)₊只用于控制，不计账单。\n\n'
    text+='负载与Q4-2光伏取最近8个同星期几历史均值；价格取过去28日同段均值，并在6小时节点用刚过去1小时观测偏差作指数衰减修正。Q4-3使用当前已发布光伏整点预报，插值积分为10分钟电量。7个历史同日联合残差构成负载、光伏、价格场景，均只来自此前完整日。\n\n'
    text+='两分支0时求共享普通计划/参考储能轨迹LP；Q4-3节点按新的库存和预报求共享调整曲线。LP可消除无效同时充放循环。其后以100kWh状态网格价值表逐段反馈，当前动作使用实际当段供需及价格，不使用后续实际曲线。0时对节点之后（6时起）的场景缺额按合同增购价1.5p计价，节点之前无发布节点、只能按5p应急，因此缺额系数与节点实际面临的增购价一致。该系数直接决定计划承诺量：系数越高，0时承诺越保守、越容易多买。\n\n'
    text+='## 三、四个指定日的表1、表2、表3\n\n所有电量kWh、费用元；完整10分钟计划见Excel。表1普通购电为G，实际外网电量为A+e，不能混用。每4小时同时出现充与放表示不同10分钟段分别操作。\n\n'
    for b in [2,3]:
        df=pd.read_csv(r/f'q4{b}_dispatch.csv');daily=pd.read_csv(r/f'q4{b}_all_daily.csv');events=pd.read_csv(r/f'q4{b}_emergency.csv')
        text+=f'### Q4-{b}\n\n'
        selected=daily[daily.date.isin(SELECTED)]
        text+=table(['日期','原计划G','最终普通A','应急e','总费用/元'],selected[['date','original_grid_kwh','adjusted_grid_kwh','emergency_kwh','total_cost_yuan']].values.tolist())
        text+='**表1：指定10分钟普通计划购电量G**\n\n'
        rows=[]
        for slot in [61,73,85,97,109,121]:
            v=[df[(df.date==date)&(df.slot==slot)].iloc[0] for date in SELECTED]
            rows.append([v[0]['interval']]+[z.original_grid_kwh for z in v])
        for label,col in [('全天原计划购电量','original_grid_kwh'),('全天原计划费用/元','plan_cost_yuan'),('全天实际总费用/元','total_cost_yuan')]:
            rows.append([label]+[float(selected[selected.date==d][col].iloc[0]) for d in SELECTED])
        text+=table(['时段/项目']+SELECTED,rows)
        if b==3:
            text+='**调整补充：最终普通购电量A（含原计划，不含应急）**\n\n'
            text+=table(['时段']+SELECTED,[[df[(df.date==SELECTED[0])&(df.slot==slot)].interval.iloc[0]]+[float(df[(df.date==d)&(df.slot==slot)].adjusted_grid_kwh.iloc[0]) for d in SELECTED] for slot in [61,73,85,97,109,121]])
        for date in SELECTED:
            day=df[df.date==date];text+=f'**表2：{date}充放电量与库存**\n\n';rows=[]
            for k in range(6):
                part=day.iloc[k*24:(k+1)*24];rows.append([f'{k*4:02d}:00—{(k+1)*4:02d}:00',part.charge_kwh.sum(),part.discharge_kwh.sum()])
            text+=table(['时间段','充电量','放电量'],rows)
            text+=f'0时储电量{day.storage_start_kwh.iloc[0]:,.6f}，24时储电量{day.storage_end_kwh.iloc[-1]:,.6f}。\n\n'
        text+='**表3：指定日紧急购电**\n\n';rows=[]
        for date in SELECTED:
            es=events[events.date==date]
            if not len(es):rows.append([date,'无',0.])
            else:rows.extend(es[['date','interval','emergency_kwh']].values.tolist())
        text+=table(['日期','时间段','应急电量'],rows)
    text+='## 四、是否需要其他时刻的光伏预报\n\n先固定2月1日初值及12月31日期末均为6000kWh，独立回放；价格每6小时更新在所有策略一致。只有新光伏版本及是否允许调整不同。最后一天补库保障约束单列在模型说明中，不覆盖主结果。\n\n'
    text+=table(['策略','公平边界总费用/元','应急电量/kWh'],[[k,s[k]['cost'],s[k]['emergency']] for k in s if k.endswith('_fair')])
    text+=table(['前策略','后策略','评价期节省/元','日均节省/元','日均95%区间'],[[z.before,z.after,z.saving,z.mean,f'[{z.ci_low:.2f}, {z.ci_high:.2f}]'] for z in cmp.itertuples()])
    text+='正值表示后策略便宜。区间使用334个日费用差、7日循环分块、2000次重抽样、种子2026；是单年开发型回测的描述性区间，不是未来保证。\n\n'
    for key,label in [('q43_stale_all_fair','全部日内新预报'),('q43_stale18_fair','18时新预报')]:
        z=cmp[cmp.before==key].iloc[0]
        judgement='本策略回测支持正的费用收益' if z.ci_low>0 else ('本策略回测显示费用上升' if z.ci_high<0 else '未显示稳定的费用收益')
        text+=f'对{label}，固定调整机会的对照日均节省{z["mean"]:.2f}元，95%区间[{z.ci_low:.2f}, {z.ci_high:.2f}]；{judgement}。\n\n'
    text+='all与stale_all仅改变可用光伏版本；all与stale18隔离18时预报。only0与all同时改变信息和调整机会，不能把其全部差值当作预报纯价值。该结论限于本保守规划与反馈结构，不能推断更好的多阶段模型也会得到相同结果。\n\n'
    text+='**须与0时缺额定价一起读**：0时按1.5p给节点后缺额定阶，等于假定节点能用增购把缺额补上；这一假定在光伏预报更新的版本里成立，在stale_all里不成立——节点拿到的光伏预报与0时完全相同，补不上缺额，缺口落到5倍应急。因此stale_all的对照值同时包含了"预报更新"和"该假定是否可执行"两重差异，all与stale_all之差应读作**更新的预报与它使日内调整真正可用的联合价值**，不是预报纯价值。反过来，缺额系数取5p虽在stale_all下更优（见第五节敏感性），却会掩掉这个区别，使两个版本看起来几乎无差。\n\n'
    text+='## 五、下界、敏感性与验证\n\n'
    text+=table(['分支','同边界完美预知下界/元','实际费用/元','差额/元','相对下界差距/%'],[[k,v['lower_bound'],v['main_cost'],v['gap_yuan'],v['gap_over_lower_pct']] for k,v in bounds.items()])
    text+='下界预先知道全部实际价格、负载与光伏，仅作参照，不能作为可执行成绩；共享轨迹近似及预测误差都可能造成差距，不能把全部差距归因为信息。\n\n'
    sc=r/'short_coef_sensitivity.csv'
    if sc.exists():
        z=pd.read_csv(sc);z=z[z.branch==3].sort_values(['policy','short_adj'],ascending=[True,False])
        text+='0时缺额系数的敏感性。主结果取1.5，即合同里节点后增购的边际价，与问题三对可调整时段的定价同源；下表把该系数放开扫描，只用于说明主结果对这一建模选择的敏感程度，**不用于选参**。\n\n'
        text+=table(['日内策略','节点后缺额系数','全年费用/元','应急电量/kWh','未利用电量/kWh'],
                    [[x.policy,x.short_adj,x.cost,x.emergency_kwh,x.unused_kwh] for x in z.itertuples()])
        text+='系数越高，0时承诺越保守、买得越多。对能用上更新光伏预报的all，1.5与2.0相差约0.1%，取合同价1.5；对不更新预报的stale_all则单调偏好高系数，因为节点补不上缺额。两者方向相反说明该系数不是可随手取的经验参数，而必须与"节点实际能执行多少增购"一致；按回测费用挑系数属于事后择优，本交付不采用。该扫描由 run_all.py 中的 `study_shortcoef.py` 步骤产生。\n\n'
    text+=table(['分支','日期','设置','实际费用/元','期末储电量/kWh'],[[f'Q4-{z.branch}',z.date,z.case,z.total_cost_yuan,z.final_kwh] for z in sens.itertuples()])
    text+='以上均从主策略同日初库存单日重算；日末库存可不同，不能据单日低费用给全年策略排名。roundtrip90表示两侧效率均sqrt(0.9)，是另一物理解释，不能混入主账本。\n\n'
    diag=pd.read_csv(r/'price_diagnostics.csv');text+='不同发布时间的价格MAE分别针对各自剩余时段，目标集合不同，不能直接用它们证明预测更新必然提高精度。\n\n'
    text+=table(['发布时间/h','平均后缀价格MAE/元每kWh'],diag.groupby('release_hour').price_mae.mean().reset_index().values.tolist())
    text+='数据、状态、功率、互斥、跨日、费用、版本、Excel及应急事件由verify.py复核；完整证据见results/verification.json。未来扰动测试和年末硬终端测试由solve.py --smoke运行。图和复现门禁状态见验收记录。\n\n'
    lk=r/'leakage_audit.json';ch=r/'leakage_chain.json';lg=r/'lag_sensitivity.json'
    if lk.exists() and ch.exists():
        a1=json.loads(lk.read_text(encoding='utf-8'));a2=json.loads(ch.read_text(encoding='utf-8'));c=a2['chains'][0]
        cut=a2['cut_day']
        text+=(f'**因果性单独送检**。`audit_leakage.py` 不依赖 solve.py 的自检，直接在{len(a1["days_tested"])}个抽样日×{a1["configs"]}种配置上把决策点之后的实际曲线按放大、置零、随机化、打乱四种方式污染，'
               f'要求已发生的决策逐位不变：{a1["perturbation_runs"]}次扰动运行的最大漂移为{a1["worst_control_leak"]:.6f}，判定{a1["status"]}。'
               f'另加整年库存结转测试：把第{cut}天起（{DATES[cut].date()}）全部替换为噪声，此前{cut}天含库存在内逐位一致（最大漂移{c["leak"]:.6f}）；'
               f'被污染运行的期末库存为{c["final_soc_ruined"]:,.0f}kWh、未污染为{c["final_soc_real"]:,.0f}kWh，说明该测试确实改变了后续状态，不是空转。\n\n')
    if lg.exists():
        s2=json.loads(lg.read_text(encoding='utf-8'))
        parts=[]
        for b in ('q42','q43'):
            if b not in s2:continue
            z=s2[b];lags=sorted(int(k[8:]) for k in z if k.startswith('cost_lag'))
            parts.append(f'Q4-{b[-1]}（lag=0复现交付账本误差{z["lag0_vs_delivered_error_yuan"]:.9f}元，'
                         +'，'.join(f'lag={l}全年费用{z[f"cost_delta_pct_lag{l}"]:+.2f}%' for l in lags if l)+'）')
        text+='`audit_lag.py` 把实时观测整体回退做严格因果对照：'+'；'.join(parts)+'。该差值是"段内即时响应"这一已声明理想化的价签，不是把后续时段当已知。\n\n'
        if 'q42' in s2 and 'q43' in s2:
            u,v=s2['q42'],s2['q43'];k=sorted(int(i[8:]) for i in u if i.startswith('cost_lag'))
            for l in [i for i in k if i]:
                text+=(f'滞后对照还给出一个必须说明的限制：在lag={l}（实时观测回退{l}段）下，Q4-2为{u[f"cost_lag{l}"]:,.0f}元、Q4-3为{v[f"cost_lag{l}"]:,.0f}元，'
                       f'Q4-3反而高出{v[f"cost_lag{l}"]-u[f"cost_lag{l}"]:,.0f}元。也就是说Q4-3相对Q4-2的优势小于"段内即时响应"这一理想化的价签，'
                       f'**不能宣称Q4-3在严格因果观测下仍然更便宜**；lag=0下二者的对照也只落在噪声量级（见第四节区间）。\n\n')
        text+='完整结果见results/leakage_audit.json、leakage_chain.json、lag_sensitivity.json。\n\n'
    text+='## 六、文件和复现\n\n- result4-2.xlsx、result4-3.xlsx：题目要求的完整结果表。\n- results：逐时账本、日汇总、版本、应急、对照、敏感性、下界与复现清单。\n- figures：9幅SVG/PNG图与灰度预览；图注见图表说明.md。\n- 题目分析报告.md、术语表格.md：模型及符号合同。\n- 从本目录运行 `python run_all.py --input "原始C题附件目录"` 可重建结果、答案、图及审计。\n\n'
    text+='方法参考：Parisio、Rikos、Glielmo（2016），Journal of Process Control 43:24–37，[出版页](https://www.sciencedirect.com/science/article/pii/S0959152416300324)，DOI 10.1016/j.jprocont.2016.04.008。只作为随机MPC背景，不使用其数值。双引擎检索受网络权限限制，已公开记录，引用通过出版页另行核验。\n'
    (out/'问题四答案.md').write_text(text,encoding='utf-8')

def figures(folder,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.plot_style import apply_publication_style,PALETTE,audit_design,_save_grayscale_preview
    sys.path.insert(0,str(SKILL/'tools/figure/scripts'))
    from visual_qa import audit_layout
    from export_figure import export_figure
    apply_publication_style(language='zh');plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})
    data,fc=read_data(folder);base,prices=predictions(data);r=out/'results';f=out/'figures';f.mkdir(exist_ok=True)
    d2=pd.read_csv(r/'q42_dispatch.csv');d3=pd.read_csv(r/'q43_dispatch.csv');cmp=pd.read_csv(r/'comparisons.csv')
    contracts=[];qa=[];blue=PALETTE['primary'];orange=PALETTE['secondary'];green=PALETTE['positive']
    def save(fig,name,claim,source,kind):
        fig.canvas.draw();issues=audit_layout(fig);design=audit_design(fig)
        if any(level in ('WARN','FAIL') for level,_ in issues) or design:raise RuntimeError((name,issues,design))
        export_figure(fig,str(f/name),formats=['svg','png'],size_inches=tuple(fig.get_size_inches()),dpi=400,grayscale_preview=False,tight=False)
        _save_grayscale_preview(f/(name+'.png'),400)
        contracts.append(dict(name=name,claim=claim,source=source,kind=kind,layout='single quantitative panel',size_inches=list(fig.get_size_inches()),dpi=400))
        qa.append(dict(name=name,layout=issues,design=design));plt.close(fig)
    def axes():return plt.subplots(figsize=(6.3,3.8),layout='constrained')
    fig,ax=axes();ax.hist(data[31:,2].ravel(),bins=50,color=blue,edgecolor='white',linewidth=.3)
    ax.set(xlabel='实际电价（元/kWh）',ylabel='10分钟时段数');save(fig,'raw_q1_price_distribution','电价覆盖显著变化范围，不能按固定单价结算','附件4，2—12月48096时段','histogram')
    fig,ax=axes()
    for months,color,line,name in [([12,1,2],blue,'-','冬季'),([3,4,5],green,'--','春季'),([6,7,8],orange,'-.','夏季'),([9,10,11],PALETTE['contrast'],':','秋季')]:
        ids=np.isin(DATES.month,months);ax.plot(np.arange(144)/6,data[ids,2].mean(0),label=name,color=color,linestyle=line)
    ax.set(xlabel='供电时段起点（时）',ylabel='季节同段平均电价（元/kWh）',xticks=np.arange(0,25,4));ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.18))
    save(fig,'raw_q1_intraday_price','不同季节的日内价格形状与水平存在差异','附件4，365日按冬春夏秋分组，不画推断区间','line')
    fig,ax=axes();net=(data[31:,0]-data[31:,1]).sum(1)/1000;price=data[31:,2].mean(1)
    ax.scatter(net,price,s=12,alpha=.55,color=blue,edgecolors='none');ax.set(xlabel='每日净负载电量（MWh）',ylabel='每日平均电价（元/kWh）')
    save(fig,'raw_q2_load_price_relation','净负载与价格需联合考虑，散点本身不作因果解释','附件2、4，334日均无删除','scatter')
    day=78;t=np.arange(144)/6
    fig,ax=axes();ax.plot(t,data[day,2],color=blue,label='实际',lw=1.2);ax.plot(t,prices[day,0],color=orange,ls='--',label='0时预测')
    ax.plot(t[72:],prices[day,2,72:],color=green,ls='-.',label='12时后缀预测');ax.set(xlabel='3月20日供电时段起点（时）',ylabel='电价（元/kWh）',xticks=np.arange(0,25,4));ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.18))
    save(fig,'process_q1_price_forecast','已发布节点只能依据此前观测修正未来价格','3月20日实际与因果价格预测，144/72个对应时段','line')
    fig,ax=axes();errors=[(fc[31:,j,78:108]-data[31:,1,78:108]).ravel()*6 for j in [0,1,2]]
    ax.boxplot(errors,tick_labels=['0时','6时','12时'],showfliers=False,patch_artist=True,boxprops={'facecolor':'#CEE2EE'},medianprops={'color':blue},whis=1.5)
    ax.axhline(0,color='gray',lw=.7,ls=':');ax.set(xlabel='光伏预报发布时间',ylabel='共同13—18时目标功率误差（kW）')
    save(fig,'process_q2_pv_error','在相同有效时段比较不同发布版本的误差','334日×30段/版本；箱体IQR，中线中位数，须1.5IQR，不显示离群点、不删原数据，不作独立样本显著性检验','boxplot')
    fig,ax=axes();z=d3[d3.date=='2025-03-20'];ax.plot(t,z.storage_end_kwh/1000,color=blue,label='实际段末库存');ax.plot(t,z.reference_storage_kwh/1000,color=orange,ls='--',label='当节点参考轨迹')
    ax.axhline(1.2,color='gray',ls=':',lw=.7);ax.axhline(10.8,color='gray',ls=':',lw=.7);ax.set(xlabel='3月20日供电时段起点（时）',ylabel='内部储能（MWh）',xticks=np.arange(0,25,4));ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.18))
    save(fig,'process_q2_storage','实际库存反馈与计划参考轨迹不同，但受同一物理边界限制','Q4-3账本，3月20日144段','line')
    fig,ax=axes();m=d2.groupby(pd.to_datetime(d2.date).dt.month)[['plan_cost_yuan','emergency_cost_yuan']].sum()/10000
    ax.bar(m.index,m.plan_cost_yuan,color=blue,label='原计划费');ax.bar(m.index,m.emergency_cost_yuan,bottom=m.plan_cost_yuan,color=orange,label='应急费');ax.set(xlabel='月份',ylabel='Q4-2实际费用（万元）',xticks=np.arange(2,13));ax.set_ylim(bottom=0);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.18))
    save(fig,'result_q1_cost','总费用需逐月计入计划和应急两项','Q4-2，334日按月实际加总，非样本均值，无误差棒','stacked_bar')
    fig,ax=axes();cs=cmp[cmp.before.isin(['q43_stale_all_fair','q43_stale18_fair'])].reset_index(drop=True);y=np.arange(len(cs))
    ax.errorbar(cs['mean'],y,xerr=np.vstack([cs['mean']-cs.ci_low,cs.ci_high-cs['mean']]),fmt='o',color=blue,capsize=4);ax.axvline(0,color='gray',ls=':',lw=.8)
    ax.set(yticks=y,yticklabels=['全部新预报','18时新预报'],xlabel='后策略日均节省及95%区间（元/日）',ylabel='同调整机会的版本对照');ax.margins(y=.5)
    save(fig,'result_q2_information_value','信息收益必须固定调整机会并检验费用差','公平6000起终库存；334日，7日循环块bootstrap，2000次，种子2026；正值表示更新便宜','interval')
    fig,ax=axes();m2=d2.groupby(pd.to_datetime(d2.date).dt.month).emergency_kwh.sum()/1000;m3=d3.groupby(pd.to_datetime(d3.date).dt.month).emergency_kwh.sum()/1000
    ax.bar(m2.index-.18,m2.values,width=.36,color=blue,label='Q4-2');ax.bar(m3.index+.18,m3.values,width=.36,color=orange,label='Q4-3');ax.set(xlabel='月份',ylabel='紧急购电量（MWh）',xticks=np.arange(2,13));ax.set_ylim(bottom=0);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.18))
    save(fig,'result_q2_emergency','降低应急电量不等于总成本一定降低','两分支主结果，2—12月实际应急按月加总','grouped_bar')
    (f/'contracts.json').write_text(json.dumps(contracts,ensure_ascii=False,indent=2),encoding='utf-8');(r/'figure_layout.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    legend='# 问题四图表说明\n\n图中q1对应Q4-2，q2对应Q4-3；图均6.3×3.8英寸、SVG与400DPI PNG，同时提供灰度预览。颜色以蓝/橙区分策略，折线另以线型区分。\n\n'
    for z in contracts:legend+=f'## {z["name"]}\n\n{z["claim"]}。\n\n数据及统计口径：{z["source"]}。\n\n![{z["claim"]}]({(f/(z["name"]+".png")).resolve().as_posix()})\n\n'
    (out/'图表说明.md').write_text(legend,encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=INPUT);p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--answer-only',action='store_true');a=p.parse_args()
    answer(a.output)
    if not a.answer_only:figures(a.input,a.output)

if __name__=='__main__':main()
