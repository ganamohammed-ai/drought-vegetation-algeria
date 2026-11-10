"""SPI-3 from CHIRPS v3 at the TerraClimate cell centres, for May (Mar-May) and Nov (Sep-Nov); gamma fit on 2001-2020.
Usage: python3 -I chirps_spi.py <chirps_dir> <group_meta.csv> <out_csv>"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from chirps_read import read_chirps
from scipy import stats, special
D, GM, OUT = sys.argv[1:4]
meta=pd.read_csv(GM); meta=meta[meta.keep]; loc=meta.groupby("tc_cell")[["lon","lat"]].first().reset_index()
rd={}
a,sc,tp,nd=read_chirps(f"{D}/chirps-v3.0.2019.05_NordALG.tif"); x0,y0=tp; res=sc[0]; H,W=a.shape
col=np.floor((loc.lon.values-x0)/res).astype(int); row=np.floor((y0-loc.lat.values)/res).astype(int)
ok=(col>=0)&(col<W)&(row>=0)&(row<H); print("cells",len(loc),"inside grid",ok.sum())
P={}
for y in range(2001,2026):
    for m in (3,4,5,9,10,11):
        a,_,_,_=read_chirps(f"{D}/chirps-v3.0.{y}.{m:02d}_NordALG.tif"); a=np.where(a<-100,np.nan,a)
        v=np.full(len(loc),np.nan); v[ok]=a[row[ok],col[ok]]; P[(y,m)]=v
rows=[]
for end,months in ((5,(3,4,5)),(11,(9,10,11))):
    S=np.array([[np.sum([P[(y,m)] for m in months],axis=0)] for y in range(2001,2026)])[:,0,:]  # years x cells
    base=S[:20]; spi=np.full_like(S,np.nan)
    for j in range(S.shape[1]):
        b=base[:,j]; b=b[np.isfinite(b)]
        if len(b)<15 or not np.isfinite(S[:,j]).any(): continue
        q0=(b<=0.5).mean(); pos=b[b>0.5]
        if len(pos)<10: continue
        lm=np.log(pos.mean())-np.mean(np.log(pos)); k=(1+np.sqrt(1+4*lm/3))/(4*lm); th=pos.mean()/k   # Thom estimate
        x=S[:,j]; G=stats.gamma.cdf(np.where(x>0.5,x,np.nan),k,scale=th); Hc=q0+(1-q0)*G; Hc=np.where(x>0.5,Hc,q0/2 if q0>0 else 1e-3)
        spi[:,j]=stats.norm.ppf(np.clip(Hc,1e-4,1-1e-4))
    for iy,y in enumerate(range(2001,2026)):
        rows.append(pd.DataFrame({"tc_cell":loc.tc_cell,"year":y,"month":end,"spei03":spi[iy]}))
R=pd.concat(rows); R.to_csv(OUT,index=False)
print(R.groupby("month").spei03.agg(["mean","std","count"]))
