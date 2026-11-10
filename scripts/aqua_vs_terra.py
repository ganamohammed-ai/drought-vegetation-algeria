"""Terra (MOD13Q1) vs Aqua (MYD13Q1) replication of the SON-MAM gap. Usage: python3 -I aqua_vs_terra.py <study_dir> <aqua_derived_dir> <group_meta.csv> <out_dir>
Both sensors: z vs the same 2003-2020 monthly climatology, same cell-episode windows (t0..t1+6), same groups (tc_cell x zone x cover)."""
import sys, os, numpy as np, pandas as pd, itertools
ROOT, AD, GM, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
ORDER=["arid|steppe","semi-arid|steppe","semi-arid|shrubland","semi-arid|forest","dry sub-humid|forest","humid|forest"]
SEASON={12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
BASE=(2003,2020); MIN_BASE=10; MODE=int(os.environ.get("AQUA_MODE",0))
meta=pd.read_csv(GM); meta=meta[meta.keep]
# ---- Aqua
rows=[]
for y in range(2002,2026):
    z=np.load(f"{AD}/aqua_monthly_{y}.npz",allow_pickle=True)
    modes=list(z["modes"]); 
    mn=z["means"][MODE,0]            # [month, group] NDVI
    for mo in range(12):
        rows.append(pd.DataFrame({"group":np.arange(mn.shape[1]),"year":y,"month":mo+1,"aqua":mn[mo]}))
A=pd.concat(rows); A=A.merge(meta[["group","tc_cell","zone","cover"]],on="group")
print("aqua mode:",modes[MODE],"groups",A.group.nunique())
# ---- Terra
m=pd.read_csv(f"{ROOT}/S4c_Monthly/S4c_monthly_by_cell_cover.csv.gz")
id_vars=["tc_cell","zone","cover","year"]
t=m.melt(id_vars=id_vars,value_vars=[f"ndvi_{k:02d}" for k in range(1,13)],var_name="vm",value_name="terra")
t["month"]=t.vm.str[-2:].astype(int)
va=m.melt(id_vars=id_vars,value_vars=[f"valid_{k:02d}" for k in range(1,13)],var_name="vm",value_name="valid"); va["month"]=va.vm.str[-2:].astype(int)
t=t.drop(columns="vm").merge(va.drop(columns="vm"),on=id_vars+["month"]); t.loc[t.valid<0.5,"terra"]=np.nan
L=A[["tc_cell","zone","cover","year","month","aqua"]].merge(t[id_vars+["month","terra"]],on=id_vars+["month"],how="inner")
# several Aqua groups may share (tc_cell,zone,cover)? average
L=L.groupby(id_vars+["month"],as_index=False)[["aqua","terra"]].mean()
L["cls"]=L.zone+"|"+L.cover; L=L[L.cls.isin(ORDER)]
# common support: both sensors valid
L.loc[L.aqua.isna()|L.terra.isna(),["aqua","terra"]]=np.nan
key=["tc_cell","zone","cover","month"]
for v in ("aqua","terra"):
    b=L[(L.year>=BASE[0])&(L.year<=BASE[1])].groupby(key)[v].agg(["mean","std","count"]).reset_index()
    b.columns=key+[v+"_mu",v+"_sd",v+"_nb"]; L=L.merge(b,on=key,how="left")
    ok=(L[v+"_nb"]>=MIN_BASE)&(L[v+"_sd"]>0.002)
    L["z_"+v]=np.where(ok,(L[v]-L[v+"_mu"])/L[v+"_sd"],np.nan)
L["t"]=L.year*12+L.month-1; L["season"]=L.month.map(SEASON)
L=L[["tc_cell","zone","cover","t","year","month","season","z_aqua","z_terra"]]
L.to_csv(f"{OUT}/G0_monthly_z_terra_aqua_common.csv.gz",index=False)
# ---- events
ev=pd.read_csv(f"{ROOT}/S4_Response_v2/S4_event_response_by_cell.csv")
ev=ev[(ev.variant=="T80_250m")&(ev.fire_history==0)&(ev.episode!="E00_local")].copy()
ev["cls"]=ev.zone_name+"|"+ev.cover; ev=ev[ev.cls.isin(ORDER)&(~ev.episode.isin(["E01","E02","E03","E04","E05"]))].reset_index(drop=True)
ev["t0"]=ev.onset.str[:4].astype(int)*12+ev.onset.str[5:7].astype(int)-1
ev["t1"]=ev.end.str[:4].astype(int)*12+ev.end.str[5:7].astype(int)-1; ev["eid"]=np.arange(len(ev))
nm=(ev.t1-ev.t0+7).values
X=pd.DataFrame({"eid":np.repeat(ev.eid.values,nm)})
X["t"]=np.repeat(ev.t0.values,nm)+(np.arange(len(X))-np.repeat(np.cumsum(nm)-nm,nm))
X=X.merge(ev[["eid","tc_cell","zone_name","cover","cls","episode"]].rename(columns={"zone_name":"zone"}),on="eid")
X=X.merge(L,on=["tc_cell","zone","cover","t"],how="left")
# pair-valid months only
X.loc[X.z_aqua.isna()|X.z_terra.isna(),["z_aqua","z_terra"]]=np.nan
res={}
for s in ("aqua","terra"):
    S=X.groupby(["eid","season"])["z_"+s].mean().unstack(); S["n"]=X.groupby("eid")["z_"+s].count()
    E=ev[["eid","tc_cell","cls","episode"]].set_index("eid").join(S); E=E[E.n>=3]; E["gap"]=E.SON-E.MAM; res[s]=E
common=res["aqua"].index.intersection(res["terra"].index)
P=pd.DataFrame({"tc_cell":res["aqua"].loc[common,"tc_cell"],"cls":res["aqua"].loc[common,"cls"],"episode":res["aqua"].loc[common,"episode"],
  "gap_aqua":res["aqua"].loc[common,"gap"],"gap_terra":res["terra"].loc[common,"gap"],
  "MAM_aqua":res["aqua"].loc[common,"MAM"],"MAM_terra":res["terra"].loc[common,"MAM"],
  "SON_aqua":res["aqua"].loc[common,"SON"],"SON_terra":res["terra"].loc[common,"SON"]}).dropna()
P.to_csv(f"{OUT}/G1_cell_episode_gap_terra_aqua.csv",index=False)
ge=P.groupby(["cls","episode"])[["gap_aqua","gap_terra","MAM_aqua","MAM_terra","SON_aqua","SON_terra"]].median().reset_index()
n=P.groupby(["cls","episode"]).size().rename("n").reset_index(); ge=ge.merge(n)
ge.to_csv(f"{OUT}/G2_gap_by_episode_class_terra_vs_aqua.csv",index=False)
def sfp(v):
    v=np.array(v); k=len(v); a=[abs(np.sum(v*np.array(s))/k) for s in itertools.product([1,-1],repeat=k)]
    return float(np.mean(np.array(a)>=abs(v.mean())-1e-12))
rows=[]
for c in ORDER:
    g=ge[ge.cls==c]
    if len(g)==0: continue
    r=dict(cls=c,n_episodes=len(g))
    for s in ("aqua","terra"):
        v=g["gap_"+s].values; r[f"mean_gap_{s}"]=v.mean(); r[f"n_neg_{s}"]=int((v<0).sum()); r[f"p_signflip_{s}"]=sfp(v)
    r["corr_gap_by_episode"]=np.corrcoef(g.gap_aqua,g.gap_terra)[0,1] if len(g)>2 else np.nan
    r["per_episode"]="; ".join(f"{e}:A{a:+.2f}/T{t:+.2f}" for e,a,t in zip(g.episode,g.gap_aqua,g.gap_terra))
    rows.append(r)
R=pd.DataFrame(rows); R.to_csv(f"{OUT}/G3_terra_vs_aqua_summary.csv",index=False)
pd.set_option("display.width",250,"display.max_colwidth",160); print(R.round(3).to_string())
# loo-episode for Aqua gap, forests
for c in ORDER[3:]:
    g=ge[ge.cls==c]; v=dict(zip(g.episode,g.gap_aqua)); 
    print(c,"Aqua leave-one-out mean:",{e:round(np.mean([x for k,x in v.items() if k!=e]),2) for e in v})
# drift check: mean (Aqua - Terra) z by year x season, humid and dry sub-humid forest
D=L[L.zone.isin(["humid","dry sub-humid"])&(L.cover=="forest")].copy(); D["d"]=D.z_aqua-D.z_terra
dy=D.groupby(["season","year"]).d.mean().unstack(0).round(2); dy.to_csv(f"{OUT}/G4_aqua_minus_terra_z_by_year_season_forests.csv"); print(dy.to_string())
