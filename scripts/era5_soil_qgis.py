# -*- coding: utf-8 -*-
r"""
ERA5-Land mensuel : humidite du sol (swvl1..swvl4) et deficit de pression de vapeur approx. (d2m, t2m) aux centres des cellules
TerraClimate, tous les mois 2001-2025. Console Python de QGIS (ou OSGeo4W Shell). Seulement gdal + numpy.
Sortie : tc_cell,year,month,swvl1..swvl4,vpd_kPa  (colonnes vides si la variable n'existe pas dans les .nc)
"""
import os, csv, glob, numpy as np
from osgeo import gdal
gdal.UseExceptions()
ERA_DIR = r"D:\DATA ALG_CLIP\08_Climate_ERA5Land_Monthly"
GROUP_META = r"E:\MYD13Q1_NordALG_2002_2025\Analysis_Final_2026_11_10\tables\group_metadata.csv"
OUT_CSV = r"C:\Users\User25\Desktop\Articles\Drought_Study\Revision_v2\tables\new\era5_soil_vpd_monthly.csv"
YEARS = range(2001, 2026)
def f(y):
    c = glob.glob(os.path.join(ERA_DIR, "*%d*.nc" % y)); 
    if not c: raise FileNotFoundError(y)
    return c[0]
names = [n.split(":")[-1] for n, d in gdal.Open(f(YEARS[0])).GetSubDatasets()]
print("variables du fichier :", names)
WANT = ["swvl1","swvl2","swvl3","swvl4","t2m","d2m"]; have = [v for v in WANT if v in names]
print("variables retenues :", have, "| absentes :", [v for v in WANT if v not in have])
cells = {}
with open(GROUP_META, newline="", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        if str(r["keep"]).strip().lower() in ("true","1","1.0"): cells.setdefault(int(float(r["tc_cell"])), (float(r["lon"]), float(r["lat"])))
ids = np.array(sorted(cells)); lon = np.array([cells[i][0] for i in ids]); lat = np.array([cells[i][1] for i in ids])
def es(tc): return 0.6108*np.exp(17.27*tc/(tc+237.3))        # kPa
rows = []
for y in YEARS:
    data = {}
    for v in have:
        ds = gdal.Open('NETCDF:"%s":%s' % (f(y), v)); gt = ds.GetGeoTransform()
        if ds.RasterCount != 12: raise RuntimeError("%s %s : %d bandes (expver ?) - me prevenir" % (y, v, ds.RasterCount))
        col = np.floor((lon-gt[0])/gt[1]).astype(int); row = np.floor((lat-gt[3])/gt[5]).astype(int)
        ok = (col>=0)&(col<ds.RasterXSize)&(row>=0)&(row<ds.RasterYSize)
        out = np.full((12, len(ids)), np.nan)
        for k in range(12):
            b = ds.GetRasterBand(k+1); a = b.ReadAsArray().astype(float); nd = b.GetNoDataValue()
            if nd is not None: a[a == nd] = np.nan
            out[k, ok] = a[row[ok], col[ok]]
        data[v] = out
    vpd = es(data["t2m"]-273.15) - es(data["d2m"]-273.15) if ("t2m" in data and "d2m" in data) else None
    for m in range(12):
        for j in range(len(ids)):
            vals = [data[v][m, j] if v in data else np.nan for v in ("swvl1","swvl2","swvl3","swvl4")] + [vpd[m, j] if vpd is not None else np.nan]
            rows.append([int(ids[j]), y, m+1] + ["" if not np.isfinite(x) else round(float(x), 4) for x in vals])
    print("lu", y, flush=True)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["tc_cell","year","month","swvl1","swvl2","swvl3","swvl4","vpd_kPa"]); w.writerows(rows)
print("ecrit :", OUT_CSV)
