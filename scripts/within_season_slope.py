import sys, itertools, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
df['SON'] = df['SON']; 
rows=[]; rng=np.random.default_rng(1)
def slope(x,y):
    m=np.isfinite(x)&np.isfinite(y)
    x,y=x[m],y[m]
    if len(x)<30 or x.std()==0: return np.nan
    return np.polyfit(x,y,1)[0]
def signflip_p(v):
    v=np.array(v); n=len(v)
    allm=[abs(np.sum(v*np.array(s))/n) for s in itertools.product([1,-1],repeat=n)]
    return float(np.mean(np.array(allm)>=abs(v.mean())-1e-12))
for cls,g in df.groupby('cls'):
    per={}
    for ep,h in g.groupby('episode'):
        x=h['peak_spei'].values.astype(float)
        per[ep]={s:slope(x,h[s].values.astype(float)) for s in ['MAM','JJA','SON']}
        per[ep]['n']=len(h)
    eps=[e for e in per if np.isfinite(per[e]['MAM']) and np.isfinite(per[e]['SON'])]
    mam=np.array([per[e]['MAM'] for e in eps]); son=np.array([per[e]['SON'] for e in eps])
    d=son-mam
    rows.append(dict(cls=cls,n_episodes=len(eps),slope_MAM=mam.mean(),slope_SON=son.mean(),
        diff_SON_minus_MAM=d.mean(),n_diff_pos=int((d>0).sum()),n_diff_neg=int((d<0).sum()),
        p_signflip=signflip_p(d) if len(d)<=12 else np.nan,
        per_episode='; '.join(f"{e}:MAM{per[e]['MAM']:+.2f}/SON{per[e]['SON']:+.2f}" for e in eps)))
res=pd.DataFrame(rows); res.to_csv(out+'/F1_within_season_slope_on_peak_spei.csv',index=False)
pd.set_option('display.width',250,'display.max_colwidth',200)
print(res.drop(columns='per_episode').round(3).to_string())
# leave-two-out of the gap (E2 per-episode values)
e2=pd.read_csv('/mnt/user-data/outputs/tables/E2_seasonal_gap_signflip.csv')
lo=[]
for _,r in e2.iterrows():
    vals={kv.split(':')[0]:float(kv.split(':')[1]) for kv in r.per_episode.split('; ')}
    ks=list(vals); res2=[]
    for a,b in itertools.combinations(ks,2):
        m=np.mean([vals[k] for k in ks if k not in(a,b)]); res2.append((m,a,b))
    mx=max(res2); mn=min(res2)
    drop=[ (np.mean([vals[k] for k in ks if k!='E05' and k!='E12']),)]
    lo.append(dict(cls=r.cls,n=len(ks),full=r['mean'],loo2_min=mn[0],loo2_max=mx[0],worst_pair=f"{mn[1]}+{mn[2]}",
        mean_wo_E05_E12=np.mean([vals[k] for k in ks if k not in('E05','E12')]),all_neg_after_any_2=all(m<0 for m,_,_ in res2)))
lo=pd.DataFrame(lo); lo.to_csv(out+'/F2_gap_leave_two_out.csv',index=False); print(lo.round(3).to_string())
