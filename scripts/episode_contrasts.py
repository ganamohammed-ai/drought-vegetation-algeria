"""Episode-level inference: the episode is the unit of replication.

Usage: python3 -I episode_contrasts.py <study_dir> <out_dir>
For each response (spring peak, MAM, JJA, SON, summer-autumn, 4-season mean) and each class contrast,
we compute the contrast inside every episode where both classes are eligible, then
summarise over episodes: mean, t-interval (n_ep - 1 df), number of episodes with the same sign and the
exact two-sided sign-flip p-value (2^n sign patterns). Also SON - MAM gap per class.
"""
import sys, os, itertools
import numpy as np
import pandas as pd
from scipy import stats

SD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ORDER = ["arid|steppe", "semi-arid|steppe", "semi-arid|shrubland",
         "semi-arid|forest", "dry sub-humid|forest", "humid|forest"]

ev = pd.read_csv(f"{SD}/S4_Response_v2/S4_event_response_by_cell.csv")
ev = ev[(ev.variant == "T80_250m") & (ev.episode != "E00_local") & (ev.fire_history == 0)].copy()
ev["cls"] = ev.zone_name + "|" + ev.cover
ev = ev[ev.cls.isin(ORDER)].copy()
ev["zk"] = ev.z_peak.round(4)
mo = pd.read_csv(f"{SD}/S4c_Monthly/Results/Event_monthly_response_by_cell.csv")
mo["zk"] = mo.z_peak_spring.round(4)
mo = mo[["tc_cell", "cls", "episode", "zk", "z_DJF", "z_MAM", "z_JJA", "z_SON"]].drop_duplicates(
    ["tc_cell", "cls", "episode", "zk"])
df = ev.merge(mo, on=["tc_cell", "cls", "episode", "zk"], how="inner")
df["z_SA"] = df[["z_JJA", "z_SON"]].mean(axis=1)
df["z_year"] = df[["z_DJF", "z_MAM", "z_JJA", "z_SON"]].mean(axis=1)
df["gap"] = df.z_SON - df.z_MAM
print("rows:", len(df))

RESP = {"z_peak": "Printemps, pic", "z_MAM": "MAM", "z_JJA": "JJA", "z_SON": "SON",
        "z_SA": "Été-automne", "z_year": "Moyenne 4 saisons", "gap": "SON - MAM"}
unit = df.groupby(["episode", "cls"])[list(RESP)].mean()

def signflip_p(x):
    x = np.asarray(x, float)
    n = len(x)
    obs = abs(x.sum())
    cnt = 0
    for s in itertools.product([-1, 1], repeat=n):
        if abs((np.array(s) * x).sum()) >= obs - 1e-12:
            cnt += 1
    return cnt / 2 ** n

def summarise(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n > 1:
        se = x.std(ddof=1) / np.sqrt(n)
        h = stats.t.ppf(.975, n - 1) * se
    else:
        h = np.nan
    return dict(n_episodes=n, mean=m, ci_low=m - h, ci_high=m + h,
                n_same_sign=int((np.sign(x) == np.sign(m)).sum()), p_signflip=signflip_p(x))

CONTR = {
    "forêt humide − steppe aride": ("humid|forest", "arid|steppe"),
    "forêt subhumide sèche − steppe aride": ("dry sub-humid|forest", "arid|steppe"),
    "forêt semi-aride − steppe aride": ("semi-arid|forest", "arid|steppe"),
    "steppe semi-aride − steppe aride": ("semi-arid|steppe", "arid|steppe"),
    "forêt semi-aride − steppe semi-aride (couvert)": ("semi-arid|forest", "semi-arid|steppe"),
    "matorral semi-aride − steppe semi-aride (couvert)": ("semi-arid|shrubland", "semi-arid|steppe"),
    "forêt humide − forêt semi-aride (zone)": ("humid|forest", "semi-arid|forest"),
    "forêt subhumide sèche − forêt semi-aride (zone)": ("dry sub-humid|forest", "semi-arid|forest"),
}
rows = []
for rcol, rname in RESP.items():
    U = unit[rcol].unstack("cls")
    for cname, (a, b) in CONTR.items():
        d = (U[a] - U[b]).dropna()
        rows.append(dict(response=rname, contrast=cname, **summarise(d.values),
                         per_episode="; ".join(f"{e}:{v:+.2f}" for e, v in d.items())))
E1 = pd.DataFrame(rows)
E1.to_csv(f"{OUT}/E1_episode_contrasts.csv", index=False)

rows = []
Ug = unit["gap"].unstack("cls")
for c in ORDER:
    d = Ug[c].dropna()
    rows.append(dict(cls=c, **summarise(d.values), per_episode="; ".join(f"{e}:{v:+.2f}" for e, v in d.items())))
E2 = pd.DataFrame(rows)
E2.to_csv(f"{OUT}/E2_seasonal_gap_signflip.csv", index=False)

pd.set_option("display.width", 250, "display.max_colwidth", 70)
print("\nEpisode-level contrasts (spring peak and SON; mean over episodes, sign-flip p):")
sel = E1[E1.response.isin(["Printemps, pic", "SON", "Moyenne 4 saisons"])]
print(E1[E1.response=="SON - MAM"].drop(columns="per_episode").round(3).to_string(index=False))
print(sel.drop(columns="per_episode").round(3).to_string(index=False))
print("\nSON - MAM gap by class, over episodes:")
print(E2.drop(columns="per_episode").round(3).to_string(index=False))
