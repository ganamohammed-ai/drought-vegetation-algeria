"""Three additions. Usage: python3 -I annual_hier_baseline.py <study_dir> <aqua_derived_dir> <group_meta.csv> <out_dir>
H1  annual slopes of seasonal NDVI z on the SPEI-3 of the same season (MAM->May, SON->Nov), SON minus MAM slope, moving-block bootstrap over years
H2  random-effects meta-analysis (DerSimonian-Laird, Hartung-Knapp CI) of the SON-MAM gap across episodes
H3  sensitivity of the SON-MAM gap to the anomaly baseline period
"""
import sys, os, numpy as np, pandas as pd
ROOT, AD, GM, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
ORDER=["arid|steppe","semi-arid|steppe","semi-arid|shrubland","semi-arid|forest","dry sub-humid|forest","humid|forest"]
SEASON={12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
# ---------- long monthly tables
m=pd.read_csv(f"{ROOT}/S4c_Monthly/S4c_monthly_by_cell_cover.csv.gz")
idv=["tc_cell","zone","cover","year"]
t=m.melt(id_vars=idv,value_vars=[f"ndvi_{k:02d}" for k in range(1,13)],var_name="vm",value_name="v"); t["month"]=t.vm.str[-2:].astype(int)
va=m.melt(id_vars=idv,value_vars=[f"valid_{k:02d}" for k in range(1,13)],var_name="vm",value_name="valid"); va["month"]=va.vm.str[-2:].astype(int)
T=t.drop(columns="vm").merge(va.drop(columns="vm"),on=idv+["month"]); T.loc[T.valid<0.5,"v"]=np.nan; T=T.drop(columns="valid")
meta=pd.read_csv(GM); meta=meta[meta.keep]
rows=[]
for y in range(2002,2026):
    z=np.load(f"{AD}/aqua_monthly_{y}.npz",allow_pickle=True); mn=z["means"][0,0]
    for mo in range(12): rows.append(pd.DataFrame({"group":np.arange(mn.shape[1]),"year":y,"month":mo+1,"v":mn[mo]}))
A=pd.concat(rows).merge(meta[["group","tc_cell","zone","cover"]],on="group").groupby(idv+["month"],as_index=False).v.mean()
for L in (T,A):
    L["cls"]=L.zone+"|"+L.cover
T=T[T.cls.isin(ORDER)].copy(); A=A[A.cls.isin(ORDER)].copy()
sp=pd.read_csv(f"{ROOT}/S4c_Monthly/S4c_spei_by_cell.csv.gz")[["tc_cell","year","month","spei03"]]
ep=pd.read_csv(f"{ROOT}/S2b_Robustness/Drought_episodes_regional.csv")
epm=set()
for _,r in ep.iterrows():
    a=pd.Period(r.start,"M"); b=pd.Period(r.end,"M")
    for p in pd.period_range(a,b,freq="M"): epm.add(p.year*12+p.month-1)
def zscore(L,base_years,drop_event_months=False,min_base=10):
    L=L.copy(); L["t"]=L.year*12+L.month-1
    msk=L.year.between(*base_years)
    if drop_event_months: msk&=~L.t.isin(epm)
    key=["tc_cell","zone","cover","month"]
    b=L[msk].groupby(key).v.agg(["mean","std","count"]).reset_index()
    L=L.merge(b,on=key,how="left")
    ok=(L["count"]>=min_base)&(L["std"]>0.002)
    L["z"]=np.where(ok,(L.v-L["mean"])/L["std"],np.nan)
    L["season"]=L.month.map(SEASON)
    return L[["tc_cell","zone","cover","cls","year","month","t","season","z"]]
# ---------- H1 annual
def block_boot(x,y,z,trend,L=3,B=3000):
    n=len(x); res=[]
    nb=int(np.ceil(n/L))
    for _ in range(B):
        st=rng.integers(0,n-L+1,nb); idx=np.concatenate([np.arange(s,s+L) for s in st])[:n]
        res.append(fit_diff(x[idx],y[idx],z[idx],trend[idx]))
    return np.array(res)
def fit_diff(xm,ym,xs,zs,trend=None): pass
def slope(x,y,tr=None):
    X=np.c_[np.ones(len(x)),x] if tr is None else np.c_[np.ones(len(x)),x,tr]
    return np.linalg.lstsq(X,y,rcond=None)[0][1]
out=[]; SERIES=[]
for sensor,L0,by in (("Terra",T,(2001,2020)),("Aqua",A,(2003,2020))):
    Z=zscore(L0,by)
    S=Z[Z.season.isin(["MAM","SON"])&(Z.year>=by[0])]
    S=S.groupby(["tc_cell","cls","year","season"]).z.agg(["mean","count"]).reset_index(); S=S[S["count"]>=2]
    sp_s=sp[sp.month.isin([5,11])].copy(); sp_s["season"]=np.where(sp_s.month==5,"MAM","SON")
    S=S.merge(sp_s[["tc_cell","year","season","spei03"]],on=["tc_cell","year","season"],how="inner")
    Y=S.groupby(["cls","year","season"])[["mean","spei03"]].mean().reset_index()
    for c in ORDER:
        d=Y[Y.cls==c]; w=d.pivot(index="year",columns="season",values=["mean","spei03"]).dropna()
        yrs=w.index.values; n=len(yrs)
        ww=w.copy(); ww.columns=['z_MAM','z_SON','spei_MAM','spei_SON']; ww['sensor']=sensor; ww['cls']=c; SERIES.append(ww.reset_index())
        zM,zS=w[("mean","MAM")].values,w[("mean","SON")].values; sM,sS=w[("spei03","MAM")].values,w[("spei03","SON")].values
        tr=(yrs-yrs.mean())/10.0
        for ctrl in (False,True):
            T_=tr if ctrl else None
            bM,bS=slope(sM,zM,T_),slope(sS,zS,T_)
            bs=[]
            nb=int(np.ceil(n/3))
            for _ in range(3000):
                st=rng.integers(0,n-2,nb); idx=np.concatenate([np.arange(s,s+3) for s in st])[:n]
                try: bs.append(slope(sS[idx],zS[idx],None if T_ is None else T_[idx])-slope(sM[idx],zM[idx],None if T_ is None else T_[idx]))
                except Exception: pass
            bs=np.array(bs)
            out.append(dict(sensor=sensor,cls=c,years=n,detrended=ctrl,r_MAM=np.corrcoef(sM,zM)[0,1],r_SON=np.corrcoef(sS,zS)[0,1],
              slope_MAM=bM,slope_SON=bS,diff=bS-bM,ci_low=np.percentile(bs,2.5),ci_high=np.percentile(bs,97.5),p_diff_le0=float((bs<=0).mean())))
pd.concat(SERIES).to_csv(f'{OUT}/H1b_annual_series_class.csv',index=False)
H1=pd.DataFrame(out); H1.to_csv(f"{OUT}/H1_annual_slope_SON_vs_MAM.csv",index=False)
pd.set_option("display.width",250); print(H1.round(2).to_string())
# ---------- events (shared by H2, H3)
ev=pd.read_csv(f"{ROOT}/S4_Response_v2/S4_event_response_by_cell.csv")
ev=ev[(ev.variant=="T80_250m")&(ev.fire_history==0)&(ev.episode!="E00_local")].copy()
ev["cls"]=ev.zone_name+"|"+ev.cover; ev=ev[ev.cls.isin(ORDER)].reset_index(drop=True)
ev["t0"]=ev.onset.str[:4].astype(int)*12+ev.onset.str[5:7].astype(int)-1
ev["t1"]=ev.end.str[:4].astype(int)*12+ev.end.str[5:7].astype(int)-1; ev["eid"]=np.arange(len(ev))
nm=(ev.t1-ev.t0+7).values
X0=pd.DataFrame({"eid":np.repeat(ev.eid.values,nm)})
X0["t"]=np.repeat(ev.t0.values,nm)+(np.arange(len(X0))-np.repeat(np.cumsum(nm)-nm,nm))
X0=X0.merge(ev[["eid","tc_cell","zone_name","cover","cls","episode"]].rename(columns={"zone_name":"zone"}),on="eid")
def gaps(Z,episodes=None):
    X=X0.merge(Z[["tc_cell","zone","cover","t","season","z"]],on=["tc_cell","zone","cover","t"],how="left")
    S=X.groupby(["eid","season"]).z.mean().unstack(); S["n"]=X.groupby("eid").z.count()
    E=ev[["eid","tc_cell","cls","episode"]].set_index("eid").join(S); E=E[E.n>=3]; E["gap"]=E.SON-E.MAM
    return E.dropna(subset=["gap"])
# ---------- H2 random-effects across episodes
def re_meta(m,v):
    k=len(m); w=1/v; mu_f=(w*m).sum()/w.sum(); Q=(w*(m-mu_f)**2).sum()
    tau2=max(0,(Q-(k-1))/(w.sum()-(w**2).sum()/w.sum())); ws=1/(v+tau2); mu=(ws*m).sum()/ws.sum()
    from scipy import stats
    q=(ws*(m-mu)**2).sum()/(k-1); se=np.sqrt(max(q,1)/ws.sum()) if False else np.sqrt(q/ws.sum())
    tcrit=stats.t.ppf(.975,k-1); I2=max(0,(Q-(k-1))/Q) if Q>0 else 0
    pi=tcrit*np.sqrt(tau2+se**2)
    return mu,mu-tcrit*se,mu+tcrit*se,np.sqrt(tau2),I2,mu-pi,mu+pi
Z_T=zscore(T,(2001,2020)); E_T=gaps(Z_T)
Z_A=zscore(A,(2003,2020)); E_A=gaps(Z_A[Z_A.year>=2003])
rows=[]
for sensor,E in (("Terra",E_T),("Aqua",E_A)):
    for c in ORDER:
        g=E[E.cls==c].groupby("episode").gap.agg(["mean","var","count"]).dropna()
        if sensor=="Aqua": g=g[~g.index.isin(["E05"])]
        if len(g)<3: continue
        v=(g["var"]/g["count"]).clip(lower=1e-6).values
        mu,lo,hi,tau,I2,plo,phi=re_meta(g["mean"].values,v)
        rows.append(dict(sensor=sensor,cls=c,episodes=len(g),mu=mu,ci_low=lo,ci_high=hi,tau=tau,I2=I2,pred_low=plo,pred_high=phi,equal_weight_mean=g["mean"].mean()))
H2=pd.DataFrame(rows); H2.to_csv(f"{OUT}/H2_random_effects_gap.csv",index=False); print(H2.round(2).to_string())
# ---------- H3 baseline sensitivity
cfg=[("2001-2020 (reference)",(2001,2020),False),("2001-2015",(2001,2015),False),("2006-2025",(2006,2025),False),("2001-2025 (all years)",(2001,2025),False),("2001-2020, episode months excluded",(2001,2020),True)]
rows=[]
for nm_,by,dr in cfg:
    E=gaps(zscore(T,by,dr,min_base=8 if dr else 10))
    for c in ORDER:
        g=E[E.cls==c].groupby("episode").gap.median()
        if len(g): rows.append(dict(sensor="Terra",baseline=nm_,cls=c,episodes=len(g),mean_gap=g.mean(),n_neg=int((g<0).sum())))
for nm_,by,dr in [("2003-2020 (reference)",(2003,2020),False),("2003-2015",(2003,2015),False),("2003-2025 (all years)",(2003,2025),False),("2003-2020, episode months excluded",(2003,2020),True)]:
    E=gaps(zscore(A,by,dr,min_base=8 if dr else 10)); E=E[E.episode!="E05"]
    for c in ORDER:
        g=E[E.cls==c].groupby("episode").gap.median()
        if len(g): rows.append(dict(sensor="Aqua",baseline=nm_,cls=c,episodes=len(g),mean_gap=g.mean(),n_neg=int((g<0).sum())))
H3=pd.DataFrame(rows); H3.to_csv(f"{OUT}/H3_baseline_sensitivity.csv",index=False)
print(H3.pivot_table(index=["sensor","baseline"],columns="cls",values="mean_gap").reindex(columns=ORDER).round(2).to_string())
