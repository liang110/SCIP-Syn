"""Deterministic manuscript Figures 2 and 3 from versioned source data."""
from pathlib import Path
import numpy as np
import pandas as pd

INTERNAL=['SCIP-Syn','ENV-Syn','DeepDDS','GraphTranSynergy','MDNNSyn','Molecular MLP','XGBoost','LinearSVR']
EXTERNAL=['SCIP-Syn','ENV-Syn','DeepDDS','Molecular MLP','XGBoost','LinearSVR']
TRAJECTORY=['SCIP-Syn','ENV-Syn','DeepDDS','Molecular MLP','MDNNSyn','XGBoost','LinearSVR']
COLORS={'SCIP-Syn':'#155A9A','ENV-Syn':'#8066A8','DeepDDS':'#4B9AA1','GraphTranSynergy':'#849CAA',
        'MDNNSyn':'#7A7F84','Molecular MLP':'#B06C8C','XGBoost':'#B77B39','LinearSVR':'#4E4E4E'}

def draw(D,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.linewidth':.65,
        'xtick.labelsize':6.3,'ytick.labelsize':6.3,'axes.labelsize':7,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.facecolor':'white'})
    output=Path(output);root=Path(__file__).resolve().parents[1]
    coords=pd.read_csv(root/'reproduce/figure2_source.csv')
    raw=pd.read_csv(root/'reproduce/strict_metrics.csv')
    def save(fig,n):
        for ext in ['svg','pdf','png','tiff']:
            fig.savefig(output/f'Figure{n}.{ext}',dpi=600)
        plt.close(fig)
    fig=plt.figure(figsize=(7.2,5.45))
    a=fig.add_axes([.075,.19,.32,.72])
    regimes=['Unit-Nonoverlap Random','One-Unseen','Both-Unseen']
    for m,mark,ls in zip(TRAJECTORY,['o','D','^','s','x','p','P'],['-','--','-.','--',':','-.',':']):
        sub=coords[(coords.panel=='a')&(coords.method==m)].set_index('regime').loc[regimes]
        a.plot(range(3),sub['mean'],linestyle=ls,marker=mark,color=COLORS[m],label=m,
               lw=1.7 if m=='SCIP-Syn' else 1,ms=4.3 if m=='SCIP-Syn' else 3.7)
    a.set_xticks(range(3),['Random\nunit-nonoverlap','One-\nUnseen','Both-\nUnseen'])
    a.set_yscale('log',base=2)
    a.set(xlim=(-.12,2.12),ylim=(.05,.9),ylabel='Pearson correlation (PCC; log2 scale)')
    a.set_yticks([.05,.1,.2,.4,.8],['0.05','0.1','0.2','0.4','0.8'])
    a.minorticks_off();a.grid(axis='y',color='#E4E9EC',lw=.5)
    a.set_title('Drug-label exposure settings',loc='left',fontsize=8.3,fontweight='bold',pad=14)
    a.text(-.16,1.04,'(a)',transform=a.transAxes,fontsize=10,fontweight='bold')
    a.legend(loc='upper left',bbox_to_anchor=(-.08,-.12),ncol=2,fontsize=6.3,frameon=False,
             columnspacing=.8,handlelength=2)
    def three(ax,panel,models,title):
        dx,dy=.37,.34
        for j,metric in enumerate(['PCC','Spearman']):
            s=coords[(coords.panel==panel)&(coords.metric==metric)].set_index('method').loc[models]
            x=np.arange(len(models),dtype=float);y=np.full(len(models),j*.58)
            means=s['mean'].to_numpy();sd=s['sd'].to_numpy()
            ax.bar3d(x-dx/2,y-dy/2,np.minimum(means,0),dx,dy,np.abs(means),
                     color=['#2C78B7','#D99550'][j],shade=True,alpha=.94,zsort='average')
            for xi,yi,m,v in zip(x,y,means,sd):
                ax.plot([xi,xi],[yi,yi],[m-v,m+v],color='#394650',lw=.6)
                for z in [m-v,m+v]:ax.plot([xi-.065,xi+.065],[yi,yi],[z,z],color='#394650',lw=.6)
        ax.set_xticks(range(len(models)),models,rotation=30,ha='right',fontsize=5.9)
        ax.set_yticks([])
        ax.set_zlim(-.22,.34);ax.set_zticks([-.2,0,.2,.3])
        ax.set_zlabel('Correlation',fontsize=6.5,labelpad=1)
        ax.set_xlim(-.6,len(models)-.35);ax.set_ylim(-.40,.98)
        ax.view_init(elev=24,azim=-62);ax.set_proj_type('ortho')
        ax.set_box_aspect((max(len(models)*.8,5),1.45,2.5),zoom=1.03)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.pane.set_facecolor((.96,.97,.98,.25))
            axis._axinfo['grid'].update({'color':'#DDE3E7','linewidth':.4})
        ax.tick_params(pad=0)
        ax.text2D(-.045,1.04,f'({panel})',transform=ax.transAxes,fontsize=10,fontweight='bold')
        ax.text2D(.07,1.04,title,transform=ax.transAxes,fontsize=8.1,fontweight='bold')
    b=fig.add_axes([.47,.55,.49,.35],projection='3d')
    c=fig.add_axes([.47,.12,.49,.35],projection='3d')
    three(b,'b',INTERNAL,'ALMANAC strict Both-Unseen')
    three(c,'c',EXTERNAL,'O’Neil strict Both-Unseen')
    fig.legend(handles=[Patch(facecolor='#2C78B7',label='PCC'),Patch(facecolor='#D99550',label='Spearman')],
               loc='upper right',bbox_to_anchor=(.97,.99),ncol=2,fontsize=6.5,frameon=False)
    save(fig,2)
    fig=plt.figure(figsize=(7.2,4.72))
    axes=[fig.add_axes(z) for z in [[.19,.57,.29,.32],[.70,.57,.28,.32],
                                  [.19,.12,.29,.31],[.70,.12,.28,.31]]]
    for i,(reg,models,title) in enumerate([('ALMANAC',INTERNAL,'ALMANAC'),('ONeil',EXTERNAL,'O’Neil')]):
        top,bottom=axes[i],axes[i+2]
        for j,m in enumerate(models):
            v=raw[(raw.regime==reg)&(raw.model==m)&(raw.metric=='EF@5%')].value
            hits=int(round(raw[(raw.regime==reg)&(raw.model==m)&(raw.metric=='Hits@5%')].value.sum()))
            assert len(v)==10
            col='#21689A' if m=='SCIP-Syn' else '#84949F'
            top.errorbar(v.mean(),j,xerr=v.std(ddof=1),fmt='D',ms=4 if m=='SCIP-Syn' else 3,
                         color=col,capsize=2,lw=1)
            bottom.plot([0,hits],[j,j],color=col,lw=1.5)
            bottom.plot(hits,j,'o',color=col,ms=4)
            bottom.text(hits+(0.45 if i==0 else 4),j,str(hits),va='center',fontsize=6,color=col)
        for ax in [top,bottom]:
            ax.set_yticks(range(len(models)),models,fontsize=6.2)
            ax.set_ylim(len(models)-.5,-.5);ax.grid(axis='x',lw=.45,color='#E4E9EC')
            ax.tick_params(axis='y',length=0)
        top.axvline(1,color='#B77B39',ls='--',lw=.9)
        top.set(xlim=(-.8,3.65),xlabel='EF@5% (mean ± SD)')
        bottom.set(xlim=(0,32 if i==0 else 318),xlabel='Cumulative strong-label events')
        if i==0:bottom.axvline(17.084829,color='#B77B39',ls='--',lw=.9)
        for ax,letter,tt in [(top,'AB'[i],title+' enrichment'),(bottom,'CD'[i],title+' selection events')]:
            ax.set_title(tt,loc='left',fontweight='bold',fontsize=8,pad=10)
            ax.text(-.48,1.08,f'({letter.lower()})',transform=ax.transAxes,fontsize=10,fontweight='bold')
    save(fig,3)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).mkdir(parents=True,exist_ok=True)
    draw(None,a.output)
