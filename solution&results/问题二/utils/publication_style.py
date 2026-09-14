"""论文插图样式：无图内标题，Arial/微软雅黑，8/10pt，PDF/SVG/1200DPI PNG。"""
from pathlib import Path
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image
from .plot_style import apply_publication_style, export_figure, PALETTE as OKABE_ITO

# 直接复用 plot_style 内置的 Okabe-Ito 色盲安全配色，只补语义别名。此前这里用同名键把
# primary/secondary/neutral 覆盖成炭黑/青绿/灰，而绘图脚本处处显式传 color=C[...]，
# 默认色环形同虚设，等于把仓库已有的色盲安全配色整条绕开了。
PALETTE={**OKABE_ITO,'observed':'#000000','f0':'#0072B2','f6':'#E69F00','f12':'#009E73','f18':'#D55E00'}
NAMES={'raw_q1_supply': '图1 问题一负载与光伏预测功率', 'raw_q1_price_distribution': '图2 问题一电价分布', 'raw_q1_price_netload': '图3 问题一电价与净负载', 'process_q1_storage': '图4 问题一储电量变化', 'process_q1_actions': '图5 问题一充放电功率', 'process_q1_losses': '图6 问题一累计效率损耗', 'result_q1_purchase': '图7 问题一计划购电量', 'result_q1_cost': '图8 问题一购电费用对比', 'result_q1_cumulative_cost': '图9 问题一累计购电费用', 'raw_q2_daily_energy': '图10 问题二每日负载与光伏电量', 'raw_q2_netload_distribution': '图11 问题二净负载分布', 'raw_q2_price_netload': '图12 问题二固定电价与平均净负载', 'process_q2_forecast_error': '图13 问题二净负载预测与实际值', 'process_q2_storage_feedback': '图14 问题二储能参考与实际轨迹', 'process_q2_actions': '图15 问题二实际充放电功率', 'result_q2_monthly_cost': '图16 问题二月度购电费用构成', 'result_q2_emergency_map': '图17 问题二紧急购电分布', 'result_q2_fair_cost': '图18 问题二相同起止库存下累计费用对比'}

def setup():
    apply_publication_style(language='zh',width='report')
    mpl.rcParams.update({
        'font.family':['Arial','Microsoft YaHei'],'font.sans-serif':['Arial','Microsoft YaHei'],
        'font.size':8,'axes.labelsize':10,'axes.titlesize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.linewidth':.8,'lines.linewidth':1.,'patch.linewidth':.5,
        'xtick.major.width':.6,'ytick.major.width':.6,
        'xtick.minor.width':.5,'ytick.minor.width':.5,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'axes.grid':False,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
        'figure.constrained_layout.use':False,'figure.autolayout':False,
    })

def create(title,xlabel,ylabel):
    fig,ax=plt.subplots(figsize=(6.3,3.9));fig.set_layout_engine(None)
    fig.subplots_adjust(left=.15,right=.96,bottom=.17,top=.87)
    ax.set(xlabel=xlabel,ylabel=ylabel)
    return fig,ax

def save(fig,directory,key):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for ax in fig.axes:
        assert not ax.get_title()
        ax.xaxis.label.set_fontsize(10);ax.yaxis.label.set_fontsize(10)
        ax.tick_params(labelsize=8)
        if ax.get_legend() is not None:
            handles,labels=ax.get_legend_handles_labels()
            ax.legend(handles,labels,frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,fontsize=8)
    fig.canvas.draw()
    text_sizes=[float(t.get_fontsize()) for t in fig.findobj(Text) if t.get_text() and t.get_visible()]
    assert all(6<=s<=10 for s in text_sizes),text_sizes
    stem=directory/NAMES[key]
    export_figure(fig,stem,dpi=1200,grayscale_preview=False)
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':NAMES[key]},facecolor='white')
    gray=directory/'_qa';gray.mkdir(exist_ok=True)
    with Image.open(stem.with_suffix('.png')) as image:
        image.convert('L').save(gray/(stem.name+'_grayscale.png'),dpi=(1200,1200))
    path=directory/'插图映射与规范.json'
    records=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    records[key]={'filename':NAMES[key],'title_inside':False,'size_inches':[6.3,3.9],
                  'raster_dpi':1200,'font_family':['Arial','Microsoft YaHei'],
                  'text_sizes_pt':sorted(set(text_sizes)),
                  'line_width_pt':[.5,1.5],'background':'white',
                  'palette':PALETTE,'formats':['pdf','svg','png']}
    path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)
