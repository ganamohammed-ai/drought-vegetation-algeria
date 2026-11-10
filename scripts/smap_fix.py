# -*- coding: utf-8 -*-
r"""
SMAP -> tables mensuelles par cellule, avec calibrage AUTOMATIQUE de l'orientation de la grille (corrige le decalage observe).
OSGeo4W Shell :  python "C:\...\smap_fix.py"      Ecrit smap_sm_monthly_cells.csv (remplace l'ancien).
"""
import os, re, csv, glob, json, datetime as dt, numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()
ROOT = r"E:\Drought_Mechanism_Data_NordALG"
GROUP_META = r"E:\MYD13Q1_NordALG_2002_2025\Analysis_Final_2026_11_10\tables\group_metadata.csv"
OUT_CSV = r"C:\Users\User25\Desktop\Articles\Drought_Study\Revision_v2\tables\new\smap_sm_monthly_cells.csv"
GRP = {"AM": ("/Soil_Moisture_Retrieval_Data_AM/", ""), "PM": ("/Soil_Moisture_Retrieval_Data_PM/", "_pm")}
files = sorted(glob.glob(os.path.join(ROOT, "SMAP_SPL3SMP_E_006", "native_daily", "*", "*.nc")))
gj = json.load(open(os.path.join(ROOT, "SMAP_SPL3SMP_E_006", "grid.json"))); T = gj["transform"]
X0, RES, Y0 = T[2], T[0], T[5]; W, H = gj["width"], gj["height"]
cells = {}
with open(GROUP_META, newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if str(r["keep"]).strip().lower() in ("true", "1", "1.0"): cells.setdefault(int(float(r["tc_cell"])), (float(r["lon"]), float(r["lat"])))
ids = np.array(sorted(cells)); LON = np.array([cells[i][0] for i in ids]); LAT = np.array([cells[i][1] for i in ids]); N = len(ids)
def rd(p, name):
    ds = gdal.Open('NETCDF:"%s":%s' % (p, name)); b = ds.GetRasterBand(1); a = b.ReadAsArray().astype(np.float64); nd = b.GetNoDataValue(); ds = None
    if nd is not None: a[a == nd] = np.nan
    return a
def date_of(p): return dt.datetime.strptime(re.search(r"(\d{8})", os.path.basename(p)).group(1), "%Y%m%d").date()
s4326 = osr.SpatialReference(); s4326.ImportFromEPSG(4326); s6933 = osr.SpatialReference(); s6933.ImportFromEPSG(6933)
for s_ in (s4326, s6933): s_.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
fwd = osr.CoordinateTransformation(s4326, s6933); inv = osr.CoordinateTransformation(s6933, s4326)
# ---- 1. orientation : comparer latitude/longitude du fichier a celles que predit la grille
LA = np.full((H, W), np.nan); LO = np.full((H, W), np.nan)
for p in files[:60]:
    la = rd(p, GRP["AM"][0] + "latitude"); lo = rd(p, GRP["AM"][0] + "longitude")
    if la.shape != (H, W): raise SystemExit("Forme inattendue %s au lieu de (%d, %d)" % (la.shape, H, W))
    v = np.isfinite(la) & np.isfinite(lo) & (np.abs(la) < 90) & (np.abs(lo) < 180); LA[v] = la[v]; LO[v] = lo[v]
v = np.isfinite(LA); print("pixels avec latitude/longitude valides :", int(v.sum()), "/", H*W)
if v.sum() < 200: raise SystemExit("Trop peu de coordonnees valides pour calibrer ; envoyez-moi ce message.")
rr, cc = np.where(v)
def predicted(rows):                       # lon/lat predits par la grille pour (rows, cc)
    xy = [(X0+(c+0.5)*RES, Y0-(r+0.5)*RES) for r, c in zip(rows, cc)]; ll = np.array(inv.TransformPoints(xy))[:, :2]; return ll[:, 0], ll[:, 1]
err = {}
for name, rows in (("normale", rr), ("inversee_nord_sud", H-1-rr)):
    lo_p, la_p = predicted(rows); err[name] = float(np.median(np.abs(lo_p-LO[v])) + np.median(np.abs(la_p-LA[v])))
print("ecart median (deg) par orientation :", {k: round(x, 4) for k, x in err.items()})
flip = err["inversee_nord_sud"] < err["normale"]; best = min(err.values())
print("orientation retenue :", "inversee nord-sud" if flip else "normale", "| ecart", round(best, 4))
if best > 0.15: raise SystemExit("Aucune orientation ne colle (ecart %.3f deg) : envoyez-moi ces lignes." % best)
# ---- 2. cellule TerraClimate -> (ligne, colonne) du fichier
xy = np.array([fwd.TransformPoint(float(a), float(b))[:2] for a, b in zip(LON, LAT)])
col = np.floor((xy[:, 0]-X0)/RES).astype(int); row = np.floor((Y0-xy[:, 1])/RES).astype(int)
if flip: row = H-1-row
ok = (col >= 0) & (col < W) & (row >= 0) & (row < H); print("cellules dans la grille :", int(ok.sum()), "/", N)
# ---- 3. moyennes mensuelles (qualite recommandee : bit 0 du drapeau = 0)
acc = {}
for i, p in enumerate(files):
    d = date_of(p); a = acc.setdefault((d.year, d.month), [np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)])
    for k, g in enumerate(("AM", "PM")):
        pre, suf = GRP[g]; sm = rd(p, pre + "soil_moisture" + suf); fl = np.nan_to_num(rd(p, pre + "retrieval_qual_flag" + suf), nan=1)
        c = np.full(N, np.nan); c[ok] = sm[row[ok], col[ok]]; f = np.zeros(N, int); f[ok] = fl[row[ok], col[ok]].astype(int)
        c[(c < 0) | (c > 1) | ((f & 1) != 0)] = np.nan; m = np.isfinite(c); a[2*k][m] += c[m]; a[2*k+1][m] += 1
    if (i+1) % 500 == 0: print("SMAP", i+1, "/", len(files), flush=True)
rows = []
for (yy, mm), a in sorted(acc.items()):
    for j in range(N):
        if a[1][j] + a[3][j] == 0: continue
        rows.append([int(ids[j]), yy, mm, "" if a[1][j] == 0 else round(a[0][j]/a[1][j], 4), "" if a[3][j] == 0 else round(a[2][j]/a[3][j], 4), int(a[1][j]), int(a[3][j])])
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f: w = csv.writer(f); w.writerow(["tc_cell","year","month","sm_smap_am","sm_smap_pm","n_am","n_pm"]); w.writerows(rows)
print("ecrit :", OUT_CSV, len(rows), "lignes")
