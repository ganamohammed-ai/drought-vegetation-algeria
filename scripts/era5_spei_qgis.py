# -*- coding: utf-8 -*-
r"""
ERA5-Land SPEI-3 (mai = MAM, novembre = SON) aux centres des cellules TerraClimate.
A coller dans la Console Python de QGIS (Extensions > Console Python > Afficher l'editeur > Ouvrir > Executer),
ou : exec(open(r"C:\...\era5_spei_qgis.py", encoding="utf-8").read())
Necessite seulement osgeo.gdal + numpy (inclus dans QGIS). Aucun paquet a installer.

Sortie : CSV  tc_cell,year,month,spei03   (meme format que chirps_spi3_may_nov.csv)
"""
import os, csv, math, glob, calendar
import numpy as np
from osgeo import gdal
from statistics import NormalDist

# ------------------------------ A MODIFIER SI BESOIN ------------------------------
ERA_DIR = r"D:\DATA ALG_CLIP\08_Climate_ERA5Land_Monthly"
GROUP_META = r"E:\MYD13Q1_NordALG_2002_2025\Analysis_Final_2026_11_10\tables\group_metadata.csv"
OUT_CSV = r"C:\Users\User25\Desktop\Articles\Drought_Study\Revision_v2\tables\new\era5_spei3_may_nov.csv"
YEARS = range(2001, 2026)
BASE = (2001, 2020)          # periode de reference
MONTHS = (5, 11)             # mois de fin des cumuls de 3 mois
# ----------------------------------------------------------------------------------
gdal.UseExceptions()

def files_for(y):
    c = glob.glob(os.path.join(ERA_DIR, "*%d*.nc" % y))
    if not c: raise FileNotFoundError("Pas de fichier ERA5-Land pour %d dans %s" % (y, ERA_DIR))
    return c[0]

def subdatasets(path):
    ds = gdal.Open(path); sd = ds.GetSubDatasets(); ds = None
    return sd   # liste de (nom, description)

# ---------- 1. Inspection du premier fichier : variables disponibles ----------
sd0 = subdatasets(files_for(YEARS[0]))
print("Sous-jeux de donnees de", files_for(YEARS[0]))
for n, d in sd0: print("  ", n, "|", d)
if not sd0:   # fichier a une seule variable
    sd0 = [(files_for(YEARS[0]), "")]

def pick(sdlist, keys):
    for n, d in sdlist:
        tag = n.split(":")[-1].lower()
        if tag in keys: return tag
    return None

VAR_P = pick(sd0, ("tp", "total_precipitation"))
VAR_E = pick(sd0, ("pev", "potential_evaporation"))
VAR_T = pick(sd0, ("t2m", "2m_temperature"))
print("Variables retenues : P =", VAR_P, "| PET(pev) =", VAR_E, "| T2m =", VAR_T)
if VAR_P is None:
    raise RuntimeError("Precipitation (tp) introuvable : verifiez la liste ci-dessus.")
if VAR_E is None and VAR_T is None:
    raise RuntimeError("Ni pev ni t2m : impossible de calculer l'ETP.")

def read_var(y, var):
    """retourne (array [12, H, W] float64, geotransform) ; NaN pour les valeurs manquantes"""
    path = files_for(y)
    ds = gdal.Open('NETCDF:"%s":%s' % (path, var))
    n = ds.RasterCount
    if n != 12:
        raise RuntimeError("%s %s : %d bandes au lieu de 12 (dimension expver ?). Dites-le moi, j'adapte." % (os.path.basename(path), var, n))
    a = np.empty((12, ds.RasterYSize, ds.RasterXSize), np.float64)
    for k in range(12):
        b = ds.GetRasterBand(k + 1); arr = b.ReadAsArray().astype(np.float64)
        nd = b.GetNoDataValue()
        if nd is not None: arr[arr == nd] = np.nan
        a[k] = arr
    gt = ds.GetGeoTransform(); ds = None
    return a, gt

# ---------- 2. Cellules TerraClimate (centres) ----------
cells = {}
with open(GROUP_META, newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if str(r["keep"]).strip().lower() not in ("true", "1", "1.0"): continue
        cells.setdefault(int(float(r["tc_cell"])), (float(r["lon"]), float(r["lat"])))
ids = np.array(sorted(cells)); lon = np.array([cells[i][0] for i in ids]); lat = np.array([cells[i][1] for i in ids])
print("cellules :", len(ids))

# ---------- 3. Serie mensuelle D = P - ETP (mm/mois) a chaque cellule ----------
def days(y): return np.array([calendar.monthrange(y, m)[1] for m in range(1, 13)], float)

D = np.full((len(YEARS) * 12, len(ids)), np.nan)
first = True
for iy, y in enumerate(YEARS):
    P, gt = read_var(y, VAR_P)
    col = np.floor((lon - gt[0]) / gt[1]).astype(int); row = np.floor((lat - gt[3]) / gt[5]).astype(int)
    ok = (col >= 0) & (col < P.shape[2]) & (row >= 0) & (row < P.shape[1])
    if first: print("pixel %.3f deg ; cellules dans la grille : %d / %d" % (gt[1], ok.sum(), len(ids)))
    p = np.full((12, len(ids)), np.nan); p[:, ok] = P[:, row[ok], col[ok]]
    # ERA5-Land mensuel : m par jour (moyenne des cumuls journaliers) -> mm/mois
    p = p * 1000.0 * days(y)[:, None]
    if VAR_E:
        E, _ = read_var(y, VAR_E)
        e = np.full((12, len(ids)), np.nan); e[:, ok] = E[:, row[ok], col[ok]]
        pet = np.abs(e) * 1000.0 * days(y)[:, None]          # pev est negatif (flux vers le haut)
    else:                                                    # Thornthwaite (sans correction de duree du jour)
        T, _ = read_var(y, VAR_T)
        t = np.full((12, len(ids)), np.nan); t[:, ok] = T[:, row[ok], col[ok]] - 273.15
        pet = None; D_T = t                                  # traite apres la boucle
    if pet is not None:
        D[iy * 12:(iy + 1) * 12] = p - pet
    else:
        D[iy * 12:(iy + 1) * 12] = np.nan
        D_T_store = globals().setdefault("D_T_store", {}); D_T_store[y] = (p, D_T)
    if first:
        print("controle : P annuelle moyenne = %.0f mm ; ETP annuelle moyenne = %s" %
              (np.nanmean(p.sum(0)), "%.0f mm" % np.nanmean(pet.sum(0)) if pet is not None else "Thornthwaite"))
        print("  (attendu pour le nord de l'Algerie : P ~ 300-1000 mm, ETP ~ 1000-1500 mm ; sinon verifier les unites)")
        first = False

if not VAR_E:
    # Thornthwaite par cellule
    Tm = np.array([D_T_store[y][1] for y in YEARS])           # annees x 12 x cellules
    Pm = np.array([D_T_store[y][0] for y in YEARS])
    Tc = np.clip(Tm, 0, None); I = ((Tc.mean(0) / 5.0) ** 1.514).sum(0)
    a_ = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
    for iy, y in enumerate(YEARS):
        pet = 16.0 * (10.0 * Tc[iy] / I) ** a_ * days(y)[:, None] / 30.0
        D[iy * 12:(iy + 1) * 12] = Pm[iy] - pet

# ---------- 4. Cumul de 3 mois, ajustement log-logistique (PWM) sur BASE, SPEI ----------
S3 = np.full_like(D, np.nan)
S3[2:] = D[2:] + D[1:-1] + D[:-2]
nd = NormalDist()

def loglogistic_fit(x):
    x = np.sort(x[np.isfinite(x)]); n = len(x)
    if n < 15: return None
    i = np.arange(1, n + 1)
    w0 = x.mean()
    w1 = np.sum((n - i) * x) / (n * (n - 1))
    w2 = np.sum((n - i) * (n - i - 1) * x) / (n * (n - 1) * (n - 2))
    den = 6 * w1 - w0 - 6 * w2
    if den == 0: return None
    beta = (2 * w1 - w0) / den
    if not np.isfinite(beta) or beta <= 1.05: return None
    g = math.gamma(1 + 1 / beta) * math.gamma(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / g
    gam = w0 - alpha * g
    if alpha <= 0: return None
    return alpha, beta, gam

rows = []
yrs = list(YEARS); base_idx = [k for k, y in enumerate(yrs) if BASE[0] <= y <= BASE[1]]
nfail = 0
for m in MONTHS:
    for j, cid in enumerate(ids):
        series = np.array([S3[k * 12 + m - 1, j] for k in range(len(yrs))])
        par = loglogistic_fit(series[base_idx])
        if par is None:
            nfail += 1; sp = np.full(len(yrs), np.nan)
        else:
            alpha, beta, gam = par
            z = series - gam
            with np.errstate(all="ignore"):
                F = 1.0 / (1.0 + (alpha / z) ** beta)
            F = np.where(z > 0, F, 1e-4)                      # sous la borne inferieure : tres sec
            F = np.clip(F, 1e-4, 1 - 1e-4)
            sp = np.array([nd.inv_cdf(float(v)) if np.isfinite(v) else np.nan for v in F])
            sp[~np.isfinite(series)] = np.nan
        for k, y in enumerate(yrs): rows.append((int(cid), y, m, sp[k]))
print("ajustements impossibles :", nfail, "sur", len(ids) * len(MONTHS))

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["tc_cell", "year", "month", "spei03"])
    for r in rows: w.writerow([r[0], r[1], r[2], "" if not np.isfinite(r[3]) else round(r[3], 4)])
v = np.array([r[3] for r in rows]);
print("ecrit :", OUT_CSV, "| moyenne SPEI = %.2f ; ecart-type = %.2f (attendus ~0 et ~1 sur la reference)" % (np.nanmean(v), np.nanstd(v)))
