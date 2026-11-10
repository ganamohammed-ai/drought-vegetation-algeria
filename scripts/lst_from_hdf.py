# -*- coding: utf-8 -*-
r"""
Table mensuelle de la LST (jour, nuit, amplitude) par cellule TerraClimate, lue DIRECTEMENT dans les HDF originaux
(E:\Drought_Mechanism_Data_NordALG\MYD11A2_061\original_HDF), sans attendre le decoupage. 2003-2025.
OSGeo4W Shell :  python "C:\...\lst_from_hdf.py"      -> lst_monthly_cells.csv
QA strict = produite, bonne qualite (bits 0-1 = 0) et erreur <= 2 K (bits 6-7 <= 1).
"""
import os, re, csv, glob, datetime as dt, numpy as np
from osgeo import gdal
gdal.UseExceptions()
ROOT = r"E:\Drought_Mechanism_Data_NordALG\MYD11A2_061\original_HDF"
GROUP_META = r"E:\MYD13Q1_NordALG_2002_2025\Analysis_Final_2026_11_10\tables\group_metadata.csv"
OUT_CSV = r"C:\Users\User25\Desktop\Articles\Drought_Study\Revision_v2\tables\new\lst_monthly_cells.csv"
MIN_COMPOSITES = 2
cs = 1/24.0; R = 6371007.181
cells = {}
with open(GROUP_META, newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if str(r["keep"]).strip().lower() in ("true", "1", "1.0"): cells.setdefault(int(float(r["tc_cell"])), (float(r["lon"]), float(r["lat"])))
ids = np.array(sorted(cells)); LON = np.array([cells[i][0] for i in ids]); LAT = np.array([cells[i][1] for i in ids]); N = len(ids)
kk = np.array([int(np.floor(a/cs))*100000 + int(np.floor(b/cs)) for a, b in zip(LON, LAT)]); order = np.argsort(kk); kks = kk[order]
cache = {}
def tile_cells(gt, W, H):
    key = (gt, W, H)
    if key not in cache:
        x = gt[0] + (np.arange(W)+0.5)*gt[1]; y = gt[3] + (np.arange(H)+0.5)*gt[5]
        lat = np.degrees(y/R)[:, None]*np.ones((1, W)); lon = np.degrees(x[None, :]/(R*np.cos(np.radians(lat))))
        flat = np.floor(lon/cs).astype(int).ravel()*100000 + np.floor(lat/cs).astype(int).ravel()
        pos = np.searchsorted(kks, flat); pos[pos >= len(kks)] = 0; hit = kks[pos] == flat
        cache[key] = np.where(hit, order[pos], -1).reshape(H, W)
    return cache[key]
def sub(path, name):
    ds = gdal.Open(path); n = next(a for a, b in ds.GetSubDatasets() if a.endswith(":" + name)); d = gdal.Open(n)
    return d.ReadAsArray(), d.GetGeoTransform(), d.RasterXSize, d.RasterYSize
files = sorted(glob.glob(os.path.join(ROOT, "*", "*.hdf"))); periods = {}
for p in files:
    m = re.search(r"\.A(\d{4})(\d{3})\.", os.path.basename(p)); periods.setdefault((int(m.group(1)), int(m.group(2))), []).append(p)
print("fichiers HDF :", len(files), "| periodes :", len(periods))
acc = {}
for n_, ((yy, doy), paths) in enumerate(sorted(periods.items())):
    mid = dt.date(yy, 1, 1) + dt.timedelta(days=doy-1+4)
    for kind, lname, qname in (("day", "LST_Day_1km", "QC_Day"), ("night", "LST_Night_1km", "QC_Night")):
        tot = np.zeros(N); cnt = np.zeros(N)
        for p in paths:
            lst, gt, W, H = sub(p, lname); qc, _, _, _ = sub(p, qname); idx = tile_cells(gt, W, H)
            q = qc.astype(np.int32); good = (lst > 0) & ((q & 3) == 0) & (((q >> 6) & 3) <= 1) & (idx >= 0)
            if not good.any(): continue
            tot += np.bincount(idx[good], weights=lst[good].astype(np.float64)*0.02 - 273.15, minlength=N); cnt += np.bincount(idx[good], minlength=N)
        mean = np.where(cnt > 0, tot/np.maximum(cnt, 1), np.nan)
        a = acc.setdefault((mid.year, mid.month, kind), [np.zeros(N), np.zeros(N)]); ok = np.isfinite(mean); a[0][ok] += mean[ok]; a[1][ok] += 1
    if (n_+1) % 50 == 0: print("LST", n_+1, "/", len(periods), flush=True)
rows = []
for (yy, mm) in sorted({(k[0], k[1]) for k in acc}):
    D = acc.get((yy, mm, "day"), [np.zeros(N), np.zeros(N)]); Nn = acc.get((yy, mm, "night"), [np.zeros(N), np.zeros(N)])
    for j in range(N):
        d_ = D[0][j]/D[1][j] if D[1][j] >= MIN_COMPOSITES else None; n2 = Nn[0][j]/Nn[1][j] if Nn[1][j] >= MIN_COMPOSITES else None
        if d_ is None and n2 is None: continue
        rows.append([int(ids[j]), yy, mm, "" if d_ is None else round(d_, 2), "" if n2 is None else round(n2, 2), "" if (d_ is None or n2 is None) else round(d_-n2, 2), int(D[1][j]), int(Nn[1][j])])
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["tc_cell", "year", "month", "lst_day_C", "lst_night_C", "dtr_C", "n_day", "n_night"]); w.writerows(rows)
print("ecrit :", OUT_CSV, len(rows), "lignes")
