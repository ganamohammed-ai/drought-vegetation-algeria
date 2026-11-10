import sys, pandas as pd, numpy as np
OUT=sys.argv[1]; PIX=0.0625
ORDER=["arid|steppe","semi-arid|steppe","semi-arid|shrubland","semi-arid|forest","dry sub-humid|forest","humid|forest"]
d=pd.read_csv(f"{OUT}/V0_event_seasonal_z_ndvi.csv")
c=d.groupby(["episode","cls","tc_cell"]).agg(SON=("SON","mean"),MAM=("MAM","mean"),n=("n_pixels","first")).reset_index()
c["area"]=c.n*PIX
rows=[]
for (e,k),g in c.groupby(["episode","cls"]):
    gs=g.dropna(subset=["SON"]); gm=g.dropna(subset=["MAM"])
    rows.append(dict(episode=e,cls=k,area_analysed=gs.area.sum(),
        km2_SON_lt_m1=gs.area[gs.SON<-1].sum(), share_SON_lt_m1=gs.area[gs.SON<-1].sum()/gs.area.sum(),
        km2_MAM_lt_m1=gm.area[gm.MAM<-1].sum(), share_MAM_lt_m1=gm.area[gm.MAM<-1].sum()/gm.area.sum(),
        n_cells=len(gs)))
R=pd.DataFrame(rows); R.to_csv(f"{OUT}/P1_area_anomaly_lt_minus1_by_episode_class.csv",index=False)
pd.set_option("display.width",220)
print((R.pivot(index="episode",columns="cls",values="km2_SON_lt_m1")[ORDER]).round(0).to_string())
print((R.pivot(index="episode",columns="cls",values="share_SON_lt_m1")[ORDER]).round(2).to_string())
print((R.pivot(index="episode",columns="cls",values="share_MAM_lt_m1")[ORDER]).round(2).to_string())
print(R.groupby("episode")[["km2_SON_lt_m1","km2_MAM_lt_m1","area_analysed"]].sum().round(0))
