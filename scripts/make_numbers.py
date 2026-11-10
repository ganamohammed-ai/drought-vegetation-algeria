"""Single source of truth for the numbers quoted in the manuscript.

Usage: python3 -I make_numbers.py <study_dir> <out_dir>
Writes claims_check.csv (article claim vs value recomputed from the CSV files)
and seasonal_by_episode.csv (SON-MAM gap per episode and class).
"""
import sys, os
import numpy as np
import pandas as pd

SD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ORDER = ["arid|steppe", "semi-arid|steppe", "semi-arid|shrubland",
         "semi-arid|forest", "dry sub-humid|forest", "humid|forest"]
rows = []

def add(item, claimed, computed, note=""):
    rows.append(dict(item=item, article_says=claimed, recomputed=computed, note=note))

# -- episodes inside the MODIS period (series starts 2000-03)
ep = pd.read_csv(f"{SD}/S2b_Robustness/Drought_episodes_regional.csv")
n_modis = int((ep.end >= "2000-03").sum())
add("episodes in MODIS period", "9", n_modis, "E05..E12")
add("regional episodes 1988-2025", "12", len(ep))

# -- share of rows confirmed by the three alternative indices
rb = pd.read_csv(f"{SD}/S6_Models/Robustness_checks.csv")
main = rb.iloc[0].rows
rob = rb[rb.check.str.startswith("robust events")].rows.iloc[0]
add("rows confirmed by all indices", "67 %", f"{100 * rob / main:.1f} %", f"{rob}/{main}")

# -- agreement between indices (domain level)
ag = pd.read_csv(f"{SD}/S2b_Robustness/Event_agreement_vs_SPEI12_TC.csv")
dom = ag[ag.zone == "domain"].set_index("alternative_index").share_SPEI12_TC_events_matched
add("event agreement range", "79-87 %",
    f"{100 * dom.min():.0f}-{100 * dom.max():.0f} % (min = SPI12_CHIRPS)",
    dom.round(3).to_dict().__str__())

# -- SPEI-SPI gap 2019-2025, scale 12
g = pd.read_csv(f"{SD}/S2b_Robustness/Zone_SPEI_minus_SPI_by_period.csv")
g = g[(g.scale == 12) & (g.period == "2019-2025")]
share = (g.mean_SPEI_minus_SPI / g.mean_SPEI).round(2)
add("share of 2019-25 intensity from evaporative demand", "about 30 %",
    f"{100 * share.min():.0f}-{100 * share.max():.0f} % by zone", dict(zip(g.zone_name, share)).__str__())

# -- trends
t = pd.read_csv(f"{SD}/S2b_Robustness/Zone_trends_PET_rain.csv")
pet = t[t.variable == "PET_TC"].median_sen_per_decade
add("PET trend mm/decade (TerraClimate)", "+21 to +43", f"+{pet.min():.0f} to +{pet.max():.0f}")
r = t[t.variable == "Rain_TC_hydroyear"].set_index("zone_name")
add("rain decline significant only in arid zone", "yes (-14 mm/dec)",
    f"arid: {r.loc['arid','share_cells_p05_corrected_pct']} % of cells significant (corrected)",
    "humid CHIRPS trend "
    f"{t[(t.variable=='Rain_CHIRPS_hydroyear')&(t.zone_name=='humid')].median_sen_per_decade.iloc[0]} mm/dec vs TC "
    f"{r.loc['humid','median_sen_per_decade']}")

# -- monthly correlation, humid forest Jun-Oct
s = pd.read_csv(f"{SD}/S4c_Monthly/Results/Sensitivity_median_r_class_month.csv")
hf = s[(s.cls == "humid|forest") & s.month.between(6, 10)].set_index("month").best
add("humid forest r, Jun-Oct", "0.53-0.63", f"{hf.min():.2f}-{hf.max():.2f}",
    "Sep = " + f"{hf.loc[9]:.2f}")
hw = s[(s.cls == "humid|forest") & s.month.isin([2, 3, 4])].best
add("humid forest r, Feb-Apr", "nil", f"{hw.min():.2f} to {hw.max():.2f}")

# how much does the 'best' time scale win by? (selection / over-interpretation check)
s["second"] = s[["spei03", "spei06", "spei12"]].apply(lambda r: np.sort(r.values)[-2], axis=1)
s["margin"] = s.best - s.second
marg = s.groupby("cls").margin.agg(["median", "max"]).reindex(ORDER)
marg.to_csv(f"{OUT}/timescale_margin_best_vs_second.csv")
add("time-scale choice: median margin best vs 2nd best r", "steppes=SPEI-12, forests=SPEI-3/6",
    f"median margin {marg['median'].min():.3f}-{marg['median'].max():.3f}",
    "margins below ~0.05 do not discriminate scales")
# per-class mean r by scale over the year
sc = s.groupby("cls")[["spei03", "spei06", "spei12"]].mean().reindex(ORDER)
sc.to_csv(f"{OUT}/mean_r_by_scale_class.csv")

# -- seasonal gap per episode (cell-level monthly response file)
m = pd.read_csv(f"{SD}/S4c_Monthly/Results/Event_monthly_response_by_cell.csv")
m["gap"] = m.z_SON - m.z_MAM
by = (m.groupby(["cls", "episode"]).agg(n=("gap", "size"), z_MAM=("z_MAM", "median"),
        z_SON=("z_SON", "median"), gap_median=("gap", "median"),
        share_SON_lower=("gap", lambda x: (x < 0).mean())).reset_index())
by.to_csv(f"{OUT}/seasonal_by_episode.csv", index=False)
piv = by.pivot(index="cls", columns="episode", values="gap_median").reindex(ORDER)
print("\nmedian (z_SON - z_MAM) per class and episode (negative = worse in autumn):")
print(piv.round(2).to_string())
pooled = m.groupby("cls").gap.median().reindex(ORDER)
add("SON-MAM gap, humid forest (pooled)", "-0.52", f"{pooled['humid|forest']:.2f}")
hfe = piv.loc["humid|forest"].dropna()
add("SON-MAM gap by episode, humid forest", "(not reported)",
    "; ".join(f"{e}:{v:.2f}" for e, v in hfe.items()),
    "episode heterogeneity")
# episode-level clustered CI for the pooled gap in forests
for c in ["humid|forest", "dry sub-humid|forest", "semi-arid|forest"]:
    v = by[by.cls == c].gap_median.values
    add(f"gap {c}: mean of episode medians [min, max]", "-", f"{v.mean():.2f} [{v.min():.2f}, {v.max():.2f}] n_ep={len(v)}")

out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/claims_check.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 90)
print("\n", out.to_string(index=False))
