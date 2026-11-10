# -*- coding: utf-8 -*-
r"""
MYD11A2 v061 (LST jour/nuit, QC, angle de vue, heure d'observation) pour le nord de l'Algerie, 2003-2025.
1) telecharge par annee via AppEEARS (decoupe geographique faite par le service, GeoTIFF),
2) traite : masque QC, moyenne par cellule TerraClimate, moyennes mensuelles -> CSV.
Fonctionne dans la Console Python de QGIS (gele l'interface pendant l'attente) ou dans le "OSGeo4W Shell" : python lst_appeears_qgis.py
Prerequis : un compte gratuit https://urs.earthdata.nasa.gov (identifiant/mot de passe demandes au lancement, jamais ecrits dans le fichier).
Relancable : les annees deja telechargees/traitees sont sautees.
"""
import os, re, sys, json, time, base64, csv, datetime as dt, urllib.request, urllib.error
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
# ------------------------------ A MODIFIER SI BESOIN ------------------------------
OUT_DIR   = r"D:\DATA ALG_CLIP\17_LST_MYD11A2"
GROUP_META= r"E:\MYD13Q1_NordALG_2002_2025\Analysis_Final_2026_11_10\tables\group_metadata.csv"
OUT_CSV   = r"C:\Users\User25\Desktop\Articles\Drought_Study\Revision_v2\tables\new\lst_monthly_by_cell.csv"
YEARS     = range(2003, 2026)
BBOX      = (-2.3, 32.1, 8.8, 37.2)           # lon min, lat min, lon max, lat max
DO_DOWNLOAD, DO_PROCESS = True, True
MIN_COMPOSITES = 2                             # minimum de composites 8 jours valides par mois et par cellule
# ----------------------------------------------------------------------------------
API = "https://appeears.earthdatacloud.nasa.gov/api"
LAYERS = ["LST_Day_1km","LST_Night_1km","QC_Day","QC_Night","Day_view_time","Night_view_time","Day_view_angl","Night_view_angl"]

def credentials():
    try:
        from qgis.PyQt.QtWidgets import QInputDialog, QLineEdit
        u,ok = QInputDialog.getText(None,"Earthdata","Identifiant Earthdata :")
        p,ok2 = QInputDialog.getText(None,"Earthdata","Mot de passe :",QLineEdit.Password)
        if ok and ok2: return u,p
    except Exception: pass
    import getpass; return input("Identifiant Earthdata : "), getpass.getpass("Mot de passe : ")

def req(url, token=None, data=None, basic=None, method=None):
    h = {}
    if token: h["Authorization"] = "Bearer " + token
    if basic: h["Authorization"] = "Basic " + base64.b64encode(("%s:%s" % basic).encode()).decode()
    if data is not None: h["Content-Type"] = "application/json"; data = json.dumps(data).encode()
    r = urllib.request.Request(url, data=data, headers=h, method=method or ("POST" if data is not None or basic else "GET"))
    for k in range(6):                                  # 502/503/504 et coupures reseau : on reessaie avec attente croissante
        try: return urllib.request.urlopen(r, timeout=300)
        except urllib.error.HTTPError as e:
            if e.code not in (500, 502, 503, 504) or k == 5: raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == 5: raise
        print("  serveur occupe, nouvel essai dans %d s" % (30*(k+1)), flush=True); time.sleep(30*(k+1))

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None

def download(url, token, dest):
    """AppEEARS redirige vers S3 : ne pas renvoyer l'en-tete Authorization."""
    op = urllib.request.build_opener(NoRedirect)
    try: r = op.open(urllib.request.Request(url, headers={"Authorization": "Bearer " + token}), timeout=120)
    except urllib.error.HTTPError as e:
        if e.code in (301,302,303,307): r = urllib.request.urlopen(e.headers["Location"], timeout=600)
        else: raise
    with open(dest, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)

def find_task(name, token):
    """Si une soumission precedente a expire (504) mais que la tache existe deja, on la reprend."""
    try:
        for t in json.load(req(API + "/task?limit=200&pretty=true", token)):
            if t.get("task_name") == name and t.get("status") not in ("error", "expired", "deleted"): return t["task_id"]
    except Exception as e: print("  liste des taches indisponible :", e)
    return None

def run_part(y, part, d1, d2, token):
    d = os.path.join(OUT_DIR, str(y)); os.makedirs(d, exist_ok=True)
    flag = os.path.join(d, "_done_%s.txt" % part)
    if os.path.exists(flag): print(y, part, "deja telecharge"); return
    name = "MYD11A2_NordALG_%d_%s" % (y, part); lo, la, hi, ha = BBOX
    task = {"task_type":"area","task_name":name,
            "params":{"dates":[{"startDate":d1,"endDate":d2}],
                      "layers":[{"product":"MYD11A2.061","layer":l} for l in LAYERS],
                      "geo":{"type":"FeatureCollection","fileName":"nordalg","features":[{"type":"Feature","properties":{},
                             "geometry":{"type":"Polygon","coordinates":[[[lo,la],[hi,la],[hi,ha],[lo,ha],[lo,la]]]}}]},
                      "output":{"format":{"type":"geotiff"},"projection":"geographic"}}}
    tid = find_task(name, token)
    if tid is None:
        try: tid = json.load(req(API + "/task", token, task))["task_id"]
        except urllib.error.HTTPError:
            time.sleep(60); tid = find_task(name, token)          # la tache a peut-etre ete creee malgre l'erreur
            if tid is None: raise
    print(y, part, "tache", tid)
    while True:
        st = json.load(req(API + "/task/" + tid, token)).get("status")
        print("  ", y, part, st, time.strftime("%H:%M:%S"), flush=True)
        if st == "done": break
        if st in ("error","expired","deleted"): raise RuntimeError("Tache %s : %s" % (tid, st))
        time.sleep(60)
    files = json.load(req(API + "/bundle/" + tid, token))["files"]
    for f in files:
        nm = os.path.basename(f["file_name"])
        if not nm.lower().endswith(".tif"): continue
        dest = os.path.join(d, nm)
        if os.path.exists(dest) and os.path.getsize(dest) > 0: continue
        download("%s/bundle/%s/%s" % (API, tid, f["file_id"]), token, dest)
    open(flag, "w").write(tid); print(y, part, "telecharge :", len(files), "fichiers")

def run_year(y, token):
    for part, a, b in (("H1", "01-01-%d" % y, "06-30-%d" % y), ("H2", "07-01-%d" % y, "12-31-%d" % y)):   # deux demi-annees : requetes plus legeres
        run_part(y, part, a, b, token)

def get_token():
    """1) jeton Earthdata (variable d'environnement EARTHDATA_TOKEN ou saisi), teste sur AppEEARS ; 2) sinon identifiant/mot de passe."""
    tok = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not tok:
        try:
            from qgis.PyQt.QtWidgets import QInputDialog
            tok, ok = QInputDialog.getText(None, "Earthdata", "Jeton Earthdata (laisser vide pour identifiant/mot de passe) :")
        except Exception:
            tok = input("Jeton Earthdata (vide = identifiant/mot de passe) : ")
        tok = (tok or "").strip()
    if tok:
        try: req(API + "/task?limit=1", tok); print("jeton accepte par AppEEARS"); return tok, None
        except urllib.error.HTTPError as e:
            print("jeton refuse par AppEEARS (HTTP %s) -> identifiant/mot de passe" % e.code)
    for k in range(3):
        u, p = credentials()
        try: return json.load(req(API + "/login", basic=(u, p)))["token"], (u, p)
        except urllib.error.HTTPError as e:
            if e.code not in (401, 403): raise
            print("Connexion refusee (HTTP %s). Utilisez le NOM D'UTILISATEUR Earthdata (pas l'e-mail) et verifiez le mot de passe ;" % e.code)
            print("au premier acces, autorisez aussi l'application AppEEARS sur https://appeears.earthdatacloud.nasa.gov (bouton Sign in).")
    raise SystemExit("Echec d'authentification")

if DO_DOWNLOAD:
    token, cred = get_token()
    for y in YEARS:
        try: run_year(y, token)
        except urllib.error.HTTPError as e:
            if e.code == 401 and cred: token = json.load(req(API + "/login", basic=cred))["token"]; run_year(y, token)
            else: raise

# ----------------------------- TRAITEMENT -----------------------------
if DO_PROCESS:
    cs = 1/24.0
    cells = {}
    with open(GROUP_META, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r["keep"]).strip().lower() in ("true","1","1.0"): cells.setdefault(int(float(r["tc_cell"])), (float(r["lon"]), float(r["lat"])))
    ids = np.array(sorted(cells)); lon = np.array([cells[i][0] for i in ids]); lat = np.array([cells[i][1] for i in ids])
    ci = np.floor(lon/cs).astype(int); cj = np.floor(lat/cs).astype(int); i0, j0 = ci.min(), cj.min(); NI, NJ = ci.max()-i0+1, cj.max()-j0+1
    LUT = np.full((NJ, NI), -1, np.int64); LUT[cj-j0, ci-i0] = np.arange(len(ids))
    cache = {}
    def cell_index(gt, H, W):
        k = (gt, H, W)
        if k not in cache:
            x = gt[0] + (np.arange(W)+0.5)*gt[1]; yv = gt[3] + (np.arange(H)+0.5)*gt[5]
            cx = np.floor(x/cs).astype(int)-i0; cy = np.floor(yv/cs).astype(int)-j0
            okx = (cx>=0)&(cx<NI); oky = (cy>=0)&(cy<NJ)
            m = LUT[np.clip(cy,0,NJ-1)[:,None], np.clip(cx,0,NI-1)[None,:]]; m[~oky,:] = -1; m[:,~okx] = -1; cache[k] = m
        return cache[k]
    def read(path):
        ds = gdal.Open(path); a = ds.GetRasterBand(1).ReadAsArray(); gt = ds.GetGeoTransform(); ds = None; return a, gt
    pat = re.compile(r"MYD11A2\.061_(LST_Day_1km|LST_Night_1km)_doy(\d{4})(\d{3})")
    acc = {}                                              # (year, month, 'Day'|'Night') -> [sum, n]
    n = len(ids)
    for y in YEARS:
        d = os.path.join(OUT_DIR, str(y))
        if not os.path.isdir(d): print("manque", y); continue
        for fn in sorted(os.listdir(d)):
            m = pat.match(fn)
            if not m: continue
            layer, yy, doy = m.group(1), int(m.group(2)), int(m.group(3)); kind = "Day" if "Day" in layer else "Night"
            qcf = os.path.join(d, fn.replace(layer, "QC_" + kind))
            if not os.path.exists(qcf): continue
            lst, gt = read(os.path.join(d, fn)); qc, _ = read(qcf)
            good = (lst > 0) & ((qc & 3) <= 1) & (((qc >> 6) & 3) <= 1)       # LST produite, erreur <= 2 K
            ci_ = cell_index(gt, *lst.shape); s = good & (ci_ >= 0)
            if not s.any(): continue
            idx = ci_[s]; val = lst[s].astype(np.float64)*0.02
            tot = np.bincount(idx, weights=val, minlength=n); cnt = np.bincount(idx, minlength=n)
            mean = np.where(cnt > 0, tot/np.maximum(cnt, 1), np.nan)           # moyenne spatiale de la cellule pour ce composite
            mid = dt.date(yy, 1, 1) + dt.timedelta(days=doy-1+4); key = (mid.year, mid.month, kind)
            a = acc.setdefault(key, [np.zeros(n), np.zeros(n)]); v = np.isfinite(mean)
            a[0][v] += mean[v]; a[1][v] += 1
        print("traite", y, flush=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["tc_cell","year","month","lst_day_C","lst_night_C","n_day","n_night"])
        for (yy, mm) in sorted({(k[0], k[1]) for k in acc}):
            D = acc.get((yy, mm, "Day"), [np.zeros(n), np.zeros(n)]); N = acc.get((yy, mm, "Night"), [np.zeros(n), np.zeros(n)])
            for j in range(n):
                if D[1][j] < MIN_COMPOSITES and N[1][j] < MIN_COMPOSITES: continue
                dd = D[0][j]/D[1][j]-273.15 if D[1][j] >= MIN_COMPOSITES else ""; nn = N[0][j]/N[1][j]-273.15 if N[1][j] >= MIN_COMPOSITES else ""
                w.writerow([int(ids[j]), yy, mm, "" if dd == "" else round(dd, 2), "" if nn == "" else round(nn, 2), int(D[1][j]), int(N[1][j])])
    print("ecrit :", OUT_CSV)
