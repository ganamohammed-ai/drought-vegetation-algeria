r"""Anomalies saisonnieres (z, base 2001-2020) de l'humidite du sol et du VPD ERA5-Land, au format de h1_chirps.py (month 5 = MAM, 11 = SON; positif = humide).
Usage: python3 -I soil_index.py <era5_soil_vpd_monthly.csv> <out_dir>"""
import sys, numpy as np, pandas as pd
d=pd.read_csv(sys.argv[1]); out=sys.argv[2]
d["swvl34"]=d[["swvl3","swvl4"]].mean(axis=1); d["neg_vpd"]=-d["vpd_kPa"]
S={5:[3,4,5],11:[9,10,11]}
for var in ("swvl1","swvl34","neg_vpd"):
    R=[]
    for end,ms in S.items():
        s=d[d.month.isin(ms)].groupby(["tc_cell","year"])[var].mean().reset_index()
        b=s[s.year.between(2001,2020)].groupby("tc_cell")[var].agg(["mean","std"]).reset_index()
        s=s.merge(b,on="tc_cell"); s["spei03"]=np.where(s["std"]>0,(s[var]-s["mean"])/s["std"],np.nan); s["month"]=end
        R.append(s[["tc_cell","year","month","spei03"]])
    pd.concat(R).to_csv(f"{out}/soilidx_{var}.csv",index=False)
