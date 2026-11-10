"""Figures 5 and 6: Terra vs Aqua SON-MAM gap. Usage: python3 -I aqua_figs.py <study_dir> <tables_aqua_dir> <group_meta.csv> <out_dir>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from gpkg_lines import read_polys
SD, TD, GM, OUT = sys.argv[1:5]
INK, INK2, SURF, GRID = "#1a1a19", "#55554f", "#fcfcfb", "#e4e3df"
ORDER=["arid|steppe","semi-arid|steppe","semi-arid|shrubland","semi-arid|forest","dry sub-humid|forest","humid|forest"]
LAB={"arid|steppe":"Steppe aride","semi-arid|steppe":"Steppe semi-aride","semi-arid|shrubland":"Matorral semi-aride","semi-arid|forest":"Forêt semi-aride","dry sub-humid|forest":"Forêt subhumide sèche","humid|forest":"Forêt humide"}
CA, CT = "#c2410c", "#2c6fb3"   # Aqua / Terra
ge=pd.read_csv(f"{TD}/G2_gap_by_episode_class_terra_vs_aqua.csv")
# ---------------- Figure 5
fig,axs=plt.subplots(2,3,figsize=(13,7.2),facecolor=SURF,sharey=True)
eps=["E06","E07","E08","E09","E10","E11","E12"]
for ax,c in zip(axs.ravel(),ORDER):
    g=ge[ge.cls==c].set_index("episode").reindex(eps)
    x=np.arange(len(eps)); ax.set_facecolor(SURF)
    ax.axhline(0,color=INK2,lw=.8)
    for xi,(a,t) in enumerate(zip(g.gap_aqua,g.gap_terra)):
        if np.isfinite(a) and np.isfinite(t): ax.plot([xi,xi],[a,t],color="#c9c8c2",lw=2,zorder=1)
    ax.scatter(x,g.gap_terra,s=60,facecolor="white",edgecolor=CT,lw=2,zorder=3,label="Terra")
    ax.scatter(x,g.gap_aqua,s=60,color=CA,zorder=4,label="Aqua")
    m=g[["gap_aqua","gap_terra"]].mean()
    ax.set_title(f"{LAB[c]}\nmoyenne Aqua {m.gap_aqua:+.2f} · Terra {m.gap_terra:+.2f} σ",loc="left",fontsize=10.5,color=INK,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(eps,fontsize=9,color=INK2); ax.tick_params(colors=INK2,length=3)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.grid(axis="y",color=GRID,lw=.6,zorder=0)
for ax in axs[:,0]: ax.set_ylabel("Écart SON − MAM (σ)",fontsize=10,color=INK2)
axs[0,0].legend(frameon=False,fontsize=10,loc="upper left",bbox_to_anchor=(0,.98))
fig.suptitle("Écart automne–printemps par épisode : Aqua et Terra, mêmes cellules et même base 2003–2020",x=.06,ha="left",fontsize=13,fontweight="bold",color=INK)
fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(f"{OUT}/Fig5_ecart_Terra_vs_Aqua_par_episode.png",dpi=200,facecolor=SURF); fig.savefig(f"{OUT}/Fig5_ecart_Terra_vs_Aqua_par_episode.svg",facecolor=SURF); plt.close(fig)
# ---------------- Figure 6: maps + scatter, forests
P=pd.read_csv(f"{TD}/G1_cell_episode_gap_terra_aqua.csv")
meta=pd.read_csv(GM); meta=meta[meta.keep]; meta["cls"]=meta.zone+"|"+meta.cover
loc=meta.groupby("tc_cell")[["lon","lat"]].first()
F=P[P.cls.isin(ORDER[3:])]
C=F.groupby("tc_cell")[["gap_aqua","gap_terra"]].mean().join(loc,how="inner")
C["n"]=F.groupby("tc_cell").size()
C.to_csv(f"{TD}/G5_forest_cell_mean_gap_terra_aqua.csv")
r=np.corrcoef(C.gap_aqua,C.gap_terra)[0,1]
polys=read_polys(f"{SD}/S0_Grid/Domain_NordALG_WGS84.gpkg")
cmap=LinearSegmentedColormap.from_list("div",["#a1370b","#e8844a","#ecebe6","#6aa4d8","#1f5a99"])
norm=TwoSlopeNorm(vmin=-1.6,vcenter=0,vmax=0.8)
fig=plt.figure(figsize=(16,4.9),facecolor=SURF)
gs=fig.add_gridspec(2,3,width_ratios=[1.35,1.35,.9],height_ratios=[1,.07],wspace=.16,hspace=.5,left=.045,right=.985,top=.88,bottom=.1)
for k,(col,tt) in enumerate([("gap_terra","a. Terra"),("gap_aqua","b. Aqua")]):
    ax=fig.add_subplot(gs[0,k]); ax.set_facecolor(SURF)
    for rings in polys:
        x,y=rings[0]; ax.fill(x,y,color="#f3f2ee",lw=0,zorder=0); ax.plot(x,y,color="#8a8a85",lw=.6,zorder=3)
    sc=ax.scatter(C.lon,C.lat,c=C[col],cmap=cmap,norm=norm,s=26,marker="s",lw=0,zorder=2)
    ax.set_xlim(-2.4,9.1); ax.set_ylim(34.2,37.4)
    ax.set_aspect(1/np.cos(np.deg2rad(35.5)))
    ax.set_title(tt,loc="left",fontsize=12.5,fontweight="bold",color=INK)
    ax.tick_params(labelsize=9,colors=INK2,length=3)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel("Longitude (°E)",fontsize=9.5,color=INK2)
    if k==0: ax.set_ylabel("Latitude (°N)",fontsize=9.5,color=INK2)
    else: ax.set_yticklabels([])
cax=fig.add_subplot(gs[1,0:2]); cb=fig.colorbar(sc,cax=cax,orientation="horizontal"); cb.set_label("Écart SON − MAM moyen sur les épisodes E07–E12 (σ) ; négatif = automne plus touché",fontsize=9.5,color=INK2); cb.ax.tick_params(labelsize=9,colors=INK2)
ax=fig.add_subplot(gs[0,2]); ax.set_facecolor(SURF)
ax.scatter(C.gap_terra,C.gap_aqua,s=16,color="#4b5d73",alpha=.55,lw=0)
lim=(-2.2,1.4); ax.plot(lim,lim,color=INK2,lw=.9,ls=(0,(4,3))); ax.axhline(0,color=GRID,lw=.8); ax.axvline(0,color=GRID,lw=.8)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Terra (σ)",fontsize=9.5,color=INK2); ax.set_ylabel("Aqua (σ)",fontsize=9.5,color=INK2)
ax.set_title(f"c. Par cellule (r = {r:.2f})",loc="left",fontsize=12.5,fontweight="bold",color=INK)
ax.tick_params(labelsize=9,colors=INK2,length=3)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
fig.suptitle("Écart automne–printemps dans les trois classes forestières : carte Terra, carte Aqua et accord par cellule",x=.04,ha="left",fontsize=13,fontweight="bold",color=INK)
fig.savefig(f"{OUT}/Fig6_carte_ecart_Terra_vs_Aqua_forets.png",dpi=200,facecolor=SURF); fig.savefig(f"{OUT}/Fig6_carte_ecart_Terra_vs_Aqua_forets.svg",facecolor=SURF)
print("cells",len(C),"r",r,"mean A",C.gap_aqua.mean(),"mean T",C.gap_terra.mean(),"share neg A",(C.gap_aqua<0).mean(),"T",(C.gap_terra<0).mean())
