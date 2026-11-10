import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TD,OUT=sys.argv[1:3]
INK,INK2,SURF,GRID="#1a1a19","#55554f","#fcfcfb","#e4e3df"
d=pd.read_csv(f"{TD}/H1b_annual_series_class.csv"); H=pd.read_csv(f"{TD}/H1_annual_slope_SON_vs_MAM.csv")
fig,axs=plt.subplots(2,2,figsize=(11,8.4),facecolor=SURF,sharex=True,sharey=True)
for i,sen in enumerate(["Terra","Aqua"]):
    g=d[(d.sensor==sen)&(d.cls=="humid|forest")]
    h=H[(H.sensor==sen)&(H.cls=="humid|forest")&(~H.detrended)].iloc[0]
    for j,(s,lab,c) in enumerate([("MAM","Printemps (MAM)","#2c6fb3"),("SON","Automne (SON)","#c2410c")]):
        ax=axs[i,j]; ax.set_facecolor(SURF)
        x=g["spei_"+s].values; y=g["z_"+s].values
        ax.axhline(0,color=GRID,lw=.8); ax.axvline(0,color=GRID,lw=.8)
        ax.scatter(x,y,s=40,color=c,zorder=3)
        for xi,yi,yr in zip(x,y,g.year):
            if yr in (2012,2016,2017,2022,2024,2025): ax.annotate(str(yr),(xi,yi),xytext=(4,4),textcoords="offset points",fontsize=8,color=INK2)
        b=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),20); ax.plot(xs,np.polyval(b,xs),color=c,lw=2)
        ax.set_title(f"{sen} · {lab}\npente {b[0]:.2f} · r = {np.corrcoef(x,y)[0,1]:.2f}",loc="left",fontsize=10.5,fontweight="bold",color=INK)
        ax.tick_params(colors=INK2,labelsize=9)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        for sp in ("left","bottom"): ax.spines[sp].set_color(GRID)
        if j==0: ax.set_ylabel("Anomalie NDVI moyenne (σ)",fontsize=9.5,color=INK2)
        if i==1: ax.set_xlabel("SPEI-3 de fin de saison (moyenne des cellules)",fontsize=9.5,color=INK2)
    axs[i,1].text(.98,.04,f"pente SON − MAM = {h['diff']:+.2f}\nIC 95 % [{h.ci_low:+.2f} ; {h.ci_high:+.2f}]",transform=axs[i,1].transAxes,ha="right",va="bottom",fontsize=9.5,color=INK)
fig.suptitle("Forêt humide : un point par année, anomalie saisonnière du NDVI selon la sécheresse de la même saison",x=.06,ha="left",fontsize=12.5,fontweight="bold",color=INK)
fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(f"{OUT}/Fig7_pentes_annuelles_foret_humide.png",dpi=200,facecolor=SURF); fig.savefig(f"{OUT}/Fig7_pentes_annuelles_foret_humide.svg",facecolor=SURF)
