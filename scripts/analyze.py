"""Re-analysis of the Drought_Study outputs (tier 1 of the improvement plan).

Usage: python3 -I analyze.py <study_dir> <out_dir>
Produces CSV tables in <out_dir>. All article numbers come from here.
"""
import sys, os
import numpy as np
import pandas as pd

SD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

ORDER = ["arid|steppe", "semi-arid|steppe", "semi-arid|shrubland",
         "semi-arid|forest", "dry sub-humid|forest", "humid|forest"]

# ------------------------------------------------------------------ data
ev = pd.read_csv(f"{SD}/S4_Response_v2/S4_event_response_by_cell.csv")
ev = ev[(ev.variant == "T80_250m") & (ev.episode != "E00_local") & (ev.fire_history == 0)].copy()
ev["cls"] = ev.zone_name + "|" + ev.cover
ev = ev[ev.cls.isin(ORDER)].copy()
print("rows main sample:", len(ev), "(article: 56948)")
assert len(ev) == 56948, "sample does not reproduce S6"

# ------------------------------------------- A. marginal means over episodes
unit = ev.groupby(["episode", "cls"]).agg(
    cells=("tc_cell", "nunique"), z_peak=("z_peak", "mean"),
    z_med=("z_peak", "median"), z_detr=("z_peak_detr", "mean"),
    n_detr=("z_peak_detr", "count"), z_mean_dr=("z_mean_drought", "mean")).reset_index()
unit.to_csv(f"{OUT}/A0_episode_class_means.csv", index=False)

episodes = sorted(unit.episode.unique())

def class_means(u, col="z_peak"):
    """equal weight per episode (marginal mean across episodes)."""
    return u.groupby("cls")[col].mean().reindex(ORDER)

base = class_means(unit)
# bootstrap over episodes
B = 5000
boots = np.full((B, len(ORDER)), np.nan)
for b in range(B):
    pick = rng.choice(episodes, size=len(episodes), replace=True)
    u = pd.concat([unit[unit.episode == e] for e in pick])
    boots[b] = class_means(u).values
ci = np.nanpercentile(boots, [2.5, 97.5], axis=0)
valid = ~np.isnan(boots).any(axis=1)          # draws where every class has >=1 episode
print("bootstrap draws with all 6 classes present:", round(valid.mean(), 3))
bv = boots[valid]
mono = np.mean([np.all(np.diff(r) >= 0) for r in bv])
# probability each class is more negative than humid forest (valid draws only)
p_lt_humid = [np.mean(bv[:, i] < bv[:, -1]) for i in range(len(ORDER))]
tabA = pd.DataFrame({
    "cls": ORDER, "marginal_mean_z_peak": base.values,
    "ci_low": ci[0], "ci_high": ci[1],
    "n_episodes": [unit[unit.cls == c].episode.nunique() for c in ORDER],
    "raw_median_all_rows": [ev[ev.cls == c].z_peak.median() for c in ORDER],
    "P(boot class < humid forest)": p_lt_humid})
tabA.to_csv(f"{OUT}/A1_marginal_means_bootstrap.csv", index=False)
print("\nA1 marginal means (equal weight per episode), bootstrap over episodes")
print(tabA.round(3).to_string(index=False))
print("share of bootstrap draws with fully monotone ordering arid steppe->humid forest:",
      round(float(mono), 3))

# like-for-like: only the episodes where ALL six classes are eligible
common = [e for e in episodes if unit[unit.episode == e].cls.nunique() == len(ORDER)]
print("episodes with all six classes:", common)
uc = unit[unit.episode.isin(common)]
bc = np.full((B, len(ORDER)), np.nan)
for b in range(B):
    pick = rng.choice(common, size=len(common), replace=True)
    bc[b] = class_means(pd.concat([uc[uc.episode == e] for e in pick])).values
tabA2 = pd.DataFrame({"cls": ORDER, "marginal_mean_common_episodes": class_means(uc).values,
                      "ci_low": np.percentile(bc, 2.5, axis=0),
                      "ci_high": np.percentile(bc, 97.5, axis=0)})
tabA2.to_csv(f"{OUT}/A1b_marginal_means_common_episodes.csv", index=False)
print(f"\nA1b like-for-like ({'+'.join(common)}):")
print(tabA2.round(3).to_string(index=False))
print("monotone ordering in", round(float(np.mean([np.all(np.diff(r) >= 0) for r in bc])), 3),
      "of bootstrap draws")

# leave-one-episode-out
rows = []
for e in episodes:
    m = class_means(unit[unit.episode != e])
    rows.append(dict(dropped=e, **{c: m[c] for c in ORDER}))
loo = pd.DataFrame(rows)
loo.to_csv(f"{OUT}/A2_leave_one_episode_out.csv", index=False)
print("\nA2 leave-one-episode-out class means")
print(loo.round(2).to_string(index=False))

# ------------------------------------------- B. detrended vs raw
print("\nB. raw vs detrended z_peak, rows where detr available")
d = ev.dropna(subset=["z_peak_detr"])
cov = d.groupby("episode").size().to_dict()
print("rows with detrended response per episode:", cov)
tb = d.groupby(["episode", "cls"]).agg(raw=("z_peak", "mean"), detr=("z_peak_detr", "mean"),
                                       n=("z_peak", "size")).reset_index()
tb.to_csv(f"{OUT}/B1_raw_vs_detrended_by_episode_class.csv", index=False)
bm = pd.DataFrame({
    "raw_same_rows": class_means(tb.rename(columns={"raw": "v"}).assign(z=lambda x: x.v), "z"),
    "detrended": class_means(tb, "detr")}).reindex(ORDER)
print(bm.round(3).to_string())
bm.to_csv(f"{OUT}/B2_marginal_raw_vs_detrended.csv")

# trend per class (from S4_sensitivity)
sens = pd.read_csv(f"{SD}/S4_Response_v2/S4_sensitivity_by_cell.csv")
sens["cls"] = sens.zone_name + "|" + sens.cover
tr = sens[sens.cls.isin(ORDER)].groupby("cls").trend_z_per_decade.agg(["median", "count"]).reindex(ORDER)
tr.to_csv(f"{OUT}/B3_trend_z_per_decade_by_class.csv")
print("\nmedian trend z per decade (non-drought springs):")
print(tr.round(2).to_string())
print("rank corr between class trend and class marginal z_peak: ",
      round(pd.Series(tr["median"].values).corr(pd.Series(base.values), method="spearman"), 2))

# ------------------------------------------- C. conditional recovery
print("\nC. recovery, all rows vs rows with z_peak < -0.5 (actually affected)")
r = ev[ev.recovery_ok == 1].copy()
r["affected"] = r.z_peak < -0.5
def rec(g):
    return pd.Series({"n": len(g),
                      "share_1y": (g.recovery_time_years <= 1).mean(),
                      "mean_years": g.recovery_time_years.mean(),
                      "share_not_recovered_5y": (g.recovery_time_years >= 5).mean()})
c_all = r.groupby("cls").apply(rec, include_groups=False).reindex(ORDER)
c_aff = r[r.affected].groupby("cls").apply(rec, include_groups=False).reindex(ORDER)
c_all.columns = [f"all_{c}" for c in c_all.columns]
c_aff.columns = [f"affected_{c}" for c in c_aff.columns]
tabC = pd.concat([c_all, c_aff], axis=1)
tabC.to_csv(f"{OUT}/C1_recovery_all_vs_affected.csv")
print(tabC.round(3).to_string())
print("share of recoverable rows already above -0.5 at peak:",
      r.groupby("cls").apply(lambda g: (g.z_peak >= -0.5).mean(), include_groups=False).reindex(ORDER).round(2).to_dict())

# ------------------------------------------- D. H3 within-cell (cell fixed effects)
print("\nD. H3 with cell FE: does a change in antecedent severity change the response of the same cell?")
def within_ols(df, y, x, cluster):
    """cell FE + episode FE via demeaning (one-way cell, episode dummies), CR1 clusters."""
    df = df.dropna(subset=[y, x]).copy()
    cnt = df.groupby("tc_cell")[y].transform("size")
    df = df[cnt >= 2]
    X = pd.get_dummies(df.episode, drop_first=True).astype(float)
    X[x] = df[x].values
    Y = df[y].values
    Xd = X.values.copy().astype(float); Yd = Y.astype(float).copy()
    g = df.tc_cell.values
    for j in range(Xd.shape[1]):
        Xd[:, j] -= pd.Series(Xd[:, j]).groupby(g).transform("mean").values
    Yd -= pd.Series(Yd).groupby(g).transform("mean").values
    beta, *_ = np.linalg.lstsq(Xd, Yd, rcond=None)
    res = Yd - Xd @ beta
    XtXi = np.linalg.pinv(Xd.T @ Xd)
    cl = df[cluster].values
    meat = np.zeros((Xd.shape[1],) * 2)
    ucl = np.unique(cl)
    for c in ucl:
        idx = cl == c
        s = Xd[idx].T @ res[idx]
        meat += np.outer(s, s)
    G, n, k = len(ucl), len(Yd), Xd.shape[1]
    adj = G / (G - 1) * (n - 1) / (n - k - df.tc_cell.nunique())
    V = adj * XtXi @ meat @ XtXi
    se = np.sqrt(V[-1, -1])
    from scipy import stats
    t = beta[-1] / se
    p = 2 * stats.t.sf(abs(t), G - 1)
    return beta[-1], se, p, n, df.tc_cell.nunique(), G

ev["block"] = (np.floor(ev.lon / 0.5).astype(int).astype(str) + "_" +
               np.floor(ev.lat / 0.5).astype(int).astype(str))
sd_ant = ev.antecedent_severity_36m.std()
ev["ant_z"] = ev.antecedent_severity_36m / sd_ant
rowsD = []
for c in ORDER:
    sub = ev[ev.cls == c]
    try:
        b, se, p, n, ncell, G = within_ols(sub, "z_peak", "ant_z", "block")
    except Exception as ex:
        b = se = p = np.nan; n = ncell = G = 0
    rowsD.append(dict(cls=c, coef_per_SD=b, se=se, p=p, rows=n, cells=ncell, clusters=G))
tabD = pd.DataFrame(rowsD)
tabD.to_csv(f"{OUT}/D1_H3_cell_fixed_effects.csv", index=False)
print(tabD.round(3).to_string(index=False))
print("(S6 cross-sectional slopes for comparison: arid steppe -0.161, semi-arid steppe -0.092, "
      "humid forest +0.126, dry sub-humid forest +0.105)")

print("\nDone. Tables in", OUT)
