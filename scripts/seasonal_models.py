"""Task 1: re-estimate the class models on seasonal responses with wild cluster bootstrap by episode.

Usage: python3 -I seasonal_models.py <study_dir> <out_dir>
Responses: spring peak (z_peak, as in S6), MAM, JJA, SON, summer-autumn (mean JJA,SON), all-season mean.
Tests (null imposed, Webb 6-point weights, episode x zone clusters, 9999 draws):
  H4a all six classes equal | H4b cover effect inside semi-arid | H4c zone effect inside forest
Also: marginal class means (equal weight per episode) with episode bootstrap CI and monotonicity share.
"""
import sys, os
import numpy as np
import pandas as pd

SD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)
ORDER = ["arid|steppe", "semi-arid|steppe", "semi-arid|shrubland",
         "semi-arid|forest", "dry sub-humid|forest", "humid|forest"]

# ------------------------------------------------------------------ data
ev = pd.read_csv(f"{SD}/S4_Response_v2/S4_event_response_by_cell.csv")
ev = ev[(ev.variant == "T80_250m") & (ev.episode != "E00_local") & (ev.fire_history == 0)].copy()
ev["cls"] = ev.zone_name + "|" + ev.cover
ev = ev[ev.cls.isin(ORDER)].copy()
mo = pd.read_csv(f"{SD}/S4c_Monthly/Results/Event_monthly_response_by_cell.csv")
mo = mo[["tc_cell", "cls", "episode", "z_DJF", "z_MAM", "z_JJA", "z_SON"]]
cv = pd.read_csv(f"{SD}/S5_Covariates/S5_covariates_by_cell_cover.csv")
ev["zk"] = ev.z_peak.round(4)
mo = pd.read_csv(f"{SD}/S4c_Monthly/Results/Event_monthly_response_by_cell.csv")[["tc_cell", "cls", "episode", "z_peak_spring", "z_DJF", "z_MAM", "z_JJA", "z_SON"]]
mo["zk"] = mo.z_peak_spring.round(4)
mo = mo.drop(columns=["z_peak_spring"]).drop_duplicates(["tc_cell", "cls", "episode", "zk"])
df = ev.merge(mo, on=["tc_cell", "cls", "episode", "zk"], how="inner")
print("rows after merging seasonal responses:", len(df), "of", len(ev))
cvcols = ["elevation_m", "slope_deg", "northness", "tpi_90m", "soil_clay_0_30", "soil_cfvo_0_30",
          "aridity_index", "pop_density_5km"]
df = df.merge(cv[["tc_cell", "cover"] + cvcols], on=["tc_cell", "cover"], how="left")
df["pop_log"] = np.log1p(df.pop_density_5km)
df["aridity_c"] = df.aridity_index - df.groupby("zone_name").aridity_index.transform("mean")
covs = ["elevation_m", "slope_deg", "northness", "tpi_90m", "soil_clay_0_30", "soil_cfvo_0_30", "aridity_c", "pop_log"]
for c in covs:
    df[c] = df[c].fillna(df[c].mean())
    df[c] = (df[c] - df[c].mean()) / df[c].std()
df["I"] = -df.peak_spei
df["I"] = (df.I - df.I.mean()) / df.I.std()
df["D"] = (df.duration_months - df.duration_months.mean()) / df.duration_months.std()
df["z_SA"] = df[["z_JJA", "z_SON"]].mean(axis=1)
df["z_year"] = df[["z_DJF", "z_MAM", "z_JJA", "z_SON"]].mean(axis=1)
df["clu"] = df.episode + "_" + df.zone_name
df["block"] = (np.floor(df.lon / 0.5).astype(int).astype(str) + "_" + np.floor(df.lat / 0.5).astype(int).astype(str))

RESP = {"z_peak": "Printemps, pic (S6)", "z_MAM": "Printemps (MAM)", "z_JJA": "Été (JJA)",
        "z_SON": "Automne (SON)", "z_SA": "Été–automne", "z_year": "Moyenne des 4 saisons"}

# ------------------------------------------------------------------ design
eps = sorted(df.episode.unique())
Xcls = pd.get_dummies(df.cls)[ORDER].astype(float)
Xep = pd.get_dummies(df.episode)[eps[1:]].astype(float)
X = pd.concat([Xcls, Xep, df[["I", "D"] + covs]], axis=1)
cols = list(X.columns)
Xv = X.values
n, k = Xv.shape
XtXi = np.linalg.inv(Xv.T @ Xv)
ci = {c: i for i, c in enumerate(cols)}

def R_rows(pairs):
    R = np.zeros((len(pairs), k))
    for j, (a, b) in enumerate(pairs):
        R[j, ci[a]] = 1; R[j, ci[b]] = -1
    return R

TESTS = {
    "H4a classes égales": R_rows([(ORDER[0], c) for c in ORDER[1:]]),
    "H4b couvert (zone semi-aride)": R_rows([("semi-arid|steppe", "semi-arid|shrubland"), ("semi-arid|steppe", "semi-arid|forest")]),
    "H4c zone (forêts)": R_rows([("semi-arid|forest", "dry sub-humid|forest"), ("semi-arid|forest", "humid|forest")]),
}
# cluster structure by episode
order = np.argsort(df.episode.values, kind="stable")
epi = df.episode.values
groups = {e: np.where(epi == e)[0] for e in eps}
G = len(eps)
Xg = {e: Xv[ix] for e, ix in groups.items()}
XgtXg = {e: Xg[e].T @ Xg[e] for e in eps}

def cr1_cov(y_resid, by="episode"):
    """CR1 covariance with clusters given."""
    if by == "episode":
        meat = np.zeros((k, k))
        for e in eps:
            s = Xg[e].T @ y_resid[groups[e]]
            meat += np.outer(s, s)
        Gc = G
    else:
        cl = df.block.values
        meat = np.zeros((k, k))
        ub = np.unique(cl)
        sc = pd.DataFrame(Xv * y_resid[:, None]).groupby(cl).sum().values
        meat = sc.T @ sc
        Gc = len(ub)
    return Gc / (Gc - 1) * (n - 1) / (n - k) * XtXi @ meat @ XtXi

def wald(b, V, R):
    d = R @ b
    M = R @ V @ R.T
    return float(d @ np.linalg.pinv(M) @ d)

W6 = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
B = 9999
rows_test, rows_mean = [], []
for rcol, rname in RESP.items():
    y = df[rcol].values
    ok = ~np.isnan(y)
    # drop NaN rows consistently
    if (~ok).any():
        print(rcol, "dropping", int((~ok).sum()), "rows")
    # (rows with NaN are rare; refit on complete cases)
    if (~ok).any():
        dd = df[ok]; yy = y[ok]
    else:
        dd = df; yy = y
    Xd = Xv[ok] if (~ok).any() else Xv
    # recompute for this subset (small loss, keep code simple)
    nn = len(yy)
    XtXi_l = np.linalg.inv(Xd.T @ Xd)
    b = XtXi_l @ Xd.T @ yy
    res = yy - Xd @ b
    epl = dd.clu.values
    gl = {e: np.where(epl == e)[0] for e in np.unique(epl)}
    Gl = len(gl)
    Xgl = {e: Xd[ix] for e, ix in gl.items()}
    def cov_ep(resid):
        meat = np.zeros((k, k))
        for e, ix in gl.items():
            s = Xgl[e].T @ resid[ix]; meat += np.outer(s, s)
        return Gl / (Gl - 1) * (nn - 1) / (nn - k) * XtXi_l @ meat @ XtXi_l
    V = cov_ep(res)
    blocks = dd.block.values
    def cov_blk(resid):
        sc = pd.DataFrame(Xd * resid[:, None]).groupby(blocks).sum().values
        Gb = sc.shape[0]
        return Gb / (Gb - 1) * (nn - 1) / (nn - k) * XtXi_l @ (sc.T @ sc) @ XtXi_l, Gb
    Vb, Gb = cov_blk(res)
    for tname, R in TESTS.items():
        # skip tests whose classes are absent
        W = wald(b, V, R)
        Wb = wald(b, Vb, R)
        # null-imposed wild cluster bootstrap by episode
        RXi = R @ XtXi_l
        br = b - XtXi_l @ R.T @ np.linalg.solve(RXi @ R.T, R @ b)
        res_r = yy - Xd @ br
        A = {e: Xgl[e].T @ res_r[ix] for e, ix in gl.items()}
        Mg = {e: Xgl[e].T @ Xgl[e] for e in gl}
        exceed = 0
        for _ in range(B):
            w = {e: W6[rng.integers(0, 6)] for e in gl}
            tot = sum(w[e] * A[e] for e in gl)
            a = XtXi_l @ tot
            meat = np.zeros((k, k))
            for e in gl:
                s = w[e] * A[e] - Mg[e] @ a
                meat += np.outer(s, s)
            Vs = Gl / (Gl - 1) * (nn - 1) / (nn - k) * XtXi_l @ meat @ XtXi_l
            bs = br + a
            Ws = wald(bs, Vs, R)
            exceed += Ws >= W
        rows_test.append(dict(response=rname, test=tname, wald_episode_CR1=W,
                              p_wild_episode=(exceed + 1) / (B + 1), wald_block05=Wb,
                              clusters_block=Gb, n=nn))
    print(rname, "tests done")
    # marginal means: equal weight per episode, bootstrap over episodes
    u = dd.assign(y=yy).groupby(["episode", "cls"]).y.mean().reset_index()
    def cm(uu): return uu.groupby("cls").y.mean().reindex(ORDER)
    base = cm(u)
    bt = np.full((5000, 6), np.nan)
    for i in range(5000):
        pick = rng.choice(eps, len(eps))
        bt[i] = cm(pd.concat([u[u.episode == e] for e in pick])).values
    val = ~np.isnan(bt).any(axis=1)
    mono_asc = np.mean([np.all(np.diff(r) >= 0) for r in bt[val]])
    lo, hi = np.nanpercentile(bt, [2.5, 97.5], axis=0)
    for j, c in enumerate(ORDER):
        rows_mean.append(dict(response=rname, cls=c, marginal_mean=base.iloc[j], ci_low=lo[j], ci_high=hi[j],
                              share_monotone_arid_to_humid=mono_asc))

pd.DataFrame(rows_test).to_csv(f"{OUT}/M1_wild_bootstrap_tests_by_response.csv", index=False)
pd.DataFrame(rows_mean).to_csv(f"{OUT}/M2_marginal_means_by_response.csv", index=False)
pd.set_option("display.width", 220)
t = pd.DataFrame(rows_test)
print("\nTests (Wald, wild-cluster bootstrap p by episode):")
print(t.round(4).to_string(index=False))
m = pd.DataFrame(rows_mean)
print("\nMarginal means by response (equal weight per episode):")
print(m.pivot(index="cls", columns="response", values="marginal_mean").reindex(ORDER).round(2).to_string())
print("\nshare of bootstrap draws with monotone arid->humid ordering:",
      m.groupby("response").share_monotone_arid_to_humid.first().round(3).to_dict())
