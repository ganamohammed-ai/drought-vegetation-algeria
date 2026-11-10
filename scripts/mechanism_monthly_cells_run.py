# -*- coding: utf-8 -*-
r"""
Tables mensuelles par cellule TerraClimate a partir de E:\Drought_Mechanism_Data_NordALG :
  LST  (MYD11A2 .tif 12 bandes)  -> lst_monthly_cells.csv   (jour, nuit, amplitude diurne, nb de composites valides)
  CCI  (ESA CCI SM .nc)          -> cci_sm_monthly_cells.csv (humidite, nb d'obs, part d'obs marquee "vegetation dense")
  SMAP (SPL3SMP_E .nc)           -> smap_sm_monthly_cells.csv (matin, soir, nb d'obs de qualite recommandee)
OSGeo4W Shell :  python "C:\...\mechanism_monthly_cells.py"      (gdal + numpy seulement ; rien a installer)
Commencer par INSPECT_ONLY = True : le script liste les variables des .nc, puis arretez-vous et envoyez-moi cette liste si un nom differe.
"""
import os, re, csv, glob, json, datetime as dt, numpy as np
from osgeo import gdal, osr
gdal.UseExceptions()
# ------------------------------ A MODIFIER SI BESOIN ------------------------------
ROOT = r"E:\Drought_Mechanism_Data_NordALG"
GROUP_META = r"E:\MYD13Q1_NordALG_2002_2025\Analysis_Final_2026_11_10\tables\group_metadata.csv"
OUT_DIR = r"C:\Users\User25\Desktop\Articles\Drought_Study\Revision_v2\tables\new"
INSPECT_ONLY = False
DO_LST, DO_CCI, DO_SMAP = False, True, True
MIN_COMPOSITES = 2                      # LST : composites 8 jours valides minimum par cellule et par mois
# ----------------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
cs = 1/24.0
cells = {}
with open(GROUP_META, newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if str(r["keep"]).strip().lower() in ("true", "1", "1.0"): cells.setdefault(int(float(r["tc_cell"])), (float(r["lon"]), float(r["lat"])))
ids = np.array(sorted(cells)); LON = np.array([cells[i][0] for i in ids]); LAT = np.array([cells[i][1] for i in ids]); N = len(ids)
def date_of(path): m = re.search(r"(\d{8})", os.path.basename(path)); return dt.datetime.strptime(m.group(1), "%Y%m%d").date()
def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f: w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("ecrit :", path, len(rows), "lignes")
def subs(path): return {n.split(":")[-1]: n for n, d in gdal.Open(path).GetSubDatasets()}
def first(pattern): 
    L = sorted(glob.glob(pattern, recursive=True)); 
    if not L: raise FileNotFoundError(pattern)
    return L[0]

# ============================== INSPECTION ==============================
cci_files = sorted(glob.glob(os.path.join(ROOT, "ESA_CCI_SM_COMBINED_09_2", "native_daily", "*", "*.nc")))
smap_files = sorted(glob.glob(os.path.join(ROOT, "SMAP_SPL3SMP_E_006", "native_daily", "*", "*.nc")))
lst_files = sorted(glob.glob(os.path.join(ROOT, "MYD11A2_061", "native_clipped", "*", "*.tif")))
print("fichiers : LST", len(lst_files), "| CCI", len(cci_files), "| SMAP", len(smap_files))
if cci_files: print("CCI variables :", list(subs(cci_files[0]).keys()) or "(un seul jeu)")
if smap_files: print("SMAP variables :", list(subs(smap_files[0]).keys()) or "(un seul jeu)")
if INSPECT_ONLY: raise SystemExit("INSPECT_ONLY = True : mettez False pour lancer le calcul.")

# ============================== LST ==============================
if DO_LST:
    R = 6371007.181
    ds = gdal.Open(lst_files[0]); gt = ds.GetGeoTransform(); W, H = ds.RasterXSize, ds.RasterYSize; ds = None
    x = gt[0] + (np.arange(W)+0.5)*gt[1]; y = gt[3] + (np.arange(H)+0.5)*gt[5]
    latp = np.degrees(y/R)[:, None] * np.ones((1, W)); lonp = np.degrees(x[None, :]/(R*np.cos(np.radians(latp))))   # sinusoidale inverse
    ci = np.floor(lonp/cs).astype(int); cj = np.floor(latp/cs).astype(int)
    key = {(int(np.floor(a/cs)), int(np.floor(b/cs))): k for k, (a, b) in enumerate(zip(LON, LAT))}
    flat = ci.ravel()*100000 + cj.ravel(); kk = np.array([a*100000+b for (a, b) in key]); vv = np.array(list(key.values()))
    srt = np.argsort(kk); pos = np.searchsorted(kk[srt], flat); pos[pos >= len(kk)] = 0
    hit = kk[srt][pos] == flat; ids2d = np.where(hit, vv[srt][pos], -1).reshape(H, W)
    ND = -2147483648; acc = {}
    for i, p in enumerate(lst_files):
        m = re.search(r"_A(\d{4})(\d{3})_", os.path.basename(p)); yy, doy = int(m.group(1)), int(m.group(2))
        mid = dt.date(yy, 1, 1) + dt.timedelta(days=doy-1+4)
        ds = gdal.Open(p); A = ds.ReadAsArray(); ds = None                  # (12, H, W)
        for kind, bl, bq in (("day", 0, 1), ("night", 4, 5)):
            lst = A[bl]; qc = A[bq]
            good = (lst != ND) & (lst > 0) & ((qc & 3) == 0) & (((qc >> 6) & 3) <= 1) & (ids2d >= 0)     # QA strict : produite, bonne qualite, erreur <= 2 K
            if not good.any(): continue
            idx = ids2d[good]; v = lst[good].astype(np.float64)*0.02 - 273.15
            tot = np.bincount(idx, weights=v, minlength=N); cnt = np.bincount(idx, minlength=N)
            mean = np.where(cnt > 0, tot/np.maximum(cnt, 1), np.nan); a = acc.setdefault((mid.year, mid.month, kind), [np.zeros(N), np.zeros(N)])
            ok = np.isfinite(mean); a[0][ok] += mean[ok]; a[1][ok] += 1
        if (i+1) % 100 == 0: print("LST", i+1, "/", len(lst_files), flush=True)
    rows = []
    for (yy, mm) in sorted({(k[0], k[1]) for k in acc}):
        D = acc.get((yy, mm, "day"), [np.zeros(N), np.zeros(N)]); Nn = acc.get((yy, mm, "night"), [np.zeros(N), np.zeros(N)])
        for j in range(N):
            d_ = D[0][j]/D[1][j] if D[1][j] >= MIN_COMPOSITES else None; n_ = Nn[0][j]/Nn[1][j] if Nn[1][j] >= MIN_COMPOSITES else None
            if d_ is None and n_ is None: continue
            rows.append([int(ids[j]), yy, mm, "" if d_ is None else round(d_, 2), "" if n_ is None else round(n_, 2),
                         "" if (d_ is None or n_ is None) else round(d_-n_, 2), int(D[1][j]), int(Nn[1][j])])
    write(os.path.join(OUT_DIR, "lst_monthly_cells.csv"), ["tc_cell","year","month","lst_day_C","lst_night_C","dtr_C","n_day","n_night"], rows)

# ============================== CCI ==============================
def grid_index(ds, lon, lat, fallback_gt=None, srs_epsg=None):
    gt = ds.GetGeoTransform()
    if gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0) and fallback_gt: gt = tuple(fallback_gt)
    x, y = lon, lat
    if srs_epsg:
        s = osr.SpatialReference(); s.ImportFromEPSG(4326); t = osr.SpatialReference(); t.ImportFromEPSG(srs_epsg)
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER); t.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(s, t); xy = np.array([ct.TransformPoint(float(a), float(b))[:2] for a, b in zip(lon, lat)]); x, y = xy[:, 0], xy[:, 1]
    col = np.floor((x-gt[0])/gt[1]).astype(int); row = np.floor((y-gt[3])/gt[5]).astype(int)
    ok = (col >= 0) & (col < ds.RasterXSize) & (row >= 0) & (row < ds.RasterYSize); return row, col, ok

def band(path, name):
    ds = gdal.Open('NETCDF:"%s":%s' % (path, name)); a = ds.GetRasterBand(1).ReadAsArray().astype(np.float64); nd = ds.GetRasterBand(1).GetNoDataValue(); return ds, a, nd

if DO_CCI:
    acc = {}; BAD = 1 | 4 | 8 | 16 | 32            # neige/gel, non-convergence, hors limites, poids faible, donnees non fiables ; vegetation dense (2) et sol nu (64) conserves puis comptes a part
    for i, p in enumerate(cci_files):
        d = date_of(p); ds, sm, nd = band(p, "sm"); _, fl, _ = band(p, "flag")
        if i == 0: row, col, ok = grid_index(ds, LON, LAT)
        sm_c = np.full(N, np.nan); fl_c = np.zeros(N, int); sm_c[ok] = sm[row[ok], col[ok]]; fl_c[ok] = np.nan_to_num(fl[row[ok], col[ok]], nan=0).astype(int)
        if nd is not None: sm_c[sm_c == nd] = np.nan
        sm_c[(sm_c < 0) | (sm_c > 1) | ((fl_c & BAD) != 0)] = np.nan
        a = acc.setdefault((d.year, d.month), [np.zeros(N), np.zeros(N), np.zeros(N)]); v = np.isfinite(sm_c)
        a[0][v] += sm_c[v]; a[1][v] += 1; a[2][v] += ((fl_c[v] & 2) != 0)
        if (i+1) % 1000 == 0: print("CCI", i+1, "/", len(cci_files), flush=True)
    rows = [[int(ids[j]), yy, mm, round(a[0][j]/a[1][j], 4), int(a[1][j]), round(a[2][j]/a[1][j], 3)] for (yy, mm), a in sorted(acc.items()) for j in range(N) if a[1][j] > 0]
    write(os.path.join(OUT_DIR, "cci_sm_monthly_cells.csv"), ["tc_cell","year","month","sm_cci","n_obs","frac_dense_veg_flag"], rows)

# ============================== SMAP ==============================
if DO_SMAP:
    gj = json.load(open(os.path.join(ROOT, "SMAP_SPL3SMP_E_006", "grid.json"))); acc = {}
    names0 = subs(smap_files[0])
    SM, SMP = "/Soil_Moisture_Retrieval_Data_AM/soil_moisture", "/Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm"
    QF, QFP = "/Soil_Moisture_Retrieval_Data_AM/retrieval_qual_flag", "/Soil_Moisture_Retrieval_Data_PM/retrieval_qual_flag_pm"
    for v in (SM, SMP, QF, QFP):
        if v not in names0: raise SystemExit("Variable SMAP introuvable : %s" % v)
    print("SMAP variables utilisees :", SM, SMP, QF, QFP)
    for i, p in enumerate(smap_files):
        d = date_of(p); vals = []
        for v, q in ((SM, QF), (SMP, QFP)):
            if v is None: vals.append(None); continue
            ds, sm, nd = band(p, v); fl = band(p, q)[1] if q else np.zeros_like(sm)
            if i == 0 and v == SM: row, col, ok = grid_index(ds, LON, LAT, fallback_gt=(gj["transform"][2], gj["transform"][0], 0.0, gj["transform"][5], 0.0, gj["transform"][4]), srs_epsg=6933)
            if i == 0 and v == SM:
                lo_ = band(p, "/Soil_Moisture_Retrieval_Data_AM/longitude")[1]; la_ = band(p, "/Soil_Moisture_Retrieval_Data_AM/latitude")[1]
                good_ = ok & np.isfinite(lo_[row.clip(0, lo_.shape[0]-1), col.clip(0, lo_.shape[1]-1)]) & (lo_[row.clip(0, lo_.shape[0]-1), col.clip(0, lo_.shape[1]-1)] > -999)
                dd_ = np.abs(lo_[row[good_], col[good_]]-LON[good_]); print("CONTROLE geolocalisation SMAP : ecart median en longitude = %.3f deg (attendu < 0.1)" % np.median(dd_))
                if np.median(dd_) > 0.2: raise SystemExit("Geolocalisation SMAP douteuse : envoyez-moi ce message.")
            c = np.full(N, np.nan); c[ok] = sm[row[ok], col[ok]]; f = np.zeros(N); f[ok] = np.nan_to_num(fl[row[ok], col[ok]], nan=1)
            c[(c < 0) | (c > 1) | (c == -9999) | ((f.astype(int) & 1) != 0)] = np.nan          # bit 0 = qualite recommandee
            vals.append(c)
        a = acc.setdefault((d.year, d.month), [np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)])
        for k, c in enumerate(vals):
            if c is None: continue
            v_ = np.isfinite(c); a[2*k][v_] += c[v_]; a[2*k+1][v_] += 1
        if (i+1) % 500 == 0: print("SMAP", i+1, "/", len(smap_files), flush=True)
    rows = []
    for (yy, mm), a in sorted(acc.items()):
        for j in range(N):
            if a[1][j] + a[3][j] == 0: continue
            rows.append([int(ids[j]), yy, mm, "" if a[1][j] == 0 else round(a[0][j]/a[1][j], 4), "" if a[3][j] == 0 else round(a[2][j]/a[3][j], 4), int(a[1][j]), int(a[3][j])])
    write(os.path.join(OUT_DIR, "smap_sm_monthly_cells.csv"), ["tc_cell","year","month","sm_smap_am","sm_smap_pm","n_am","n_pm"], rows)
