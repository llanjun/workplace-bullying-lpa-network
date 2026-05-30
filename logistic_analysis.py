# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from tableone import TableOne

# ── Settings ──────────────────────────────────────────────
MPLUS_DIR = "MPLUS_result"
BEST_CLASS = 3
OUT_DIR = "Logistic_result"
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORICAL_CANDIDATES = [
    "Gender", "Rural", "LeftBy", "Religion", "Married", "Level", "Education",
    "Exposure", "WorkTime", "SleepQuality", "Alcohol", "Smoke", "Exercise",
    "ChronicDisease", "Cluster",
]

# ── 1. Load and Merge Data ────────────────────────────────
data = pd.read_excel(os.path.join(MPLUS_DIR, "abuse_pos_only_demographic.xlsx"))
cluster_label = pd.read_csv(
    os.path.join(MPLUS_DIR, f"dataLPA{BEST_CLASS}.TXT"),
    sep=r"[\s,]+", header=None, engine="python",
)
data["Cluster"] = cluster_label.iloc[:, -1].astype(int).values

missing_counts = data.isna().sum()
missing_counts = missing_counts[missing_counts > 0]
if not missing_counts.empty:
    print("Missing values per variable (before listwise deletion):")
    for var, cnt in missing_counts.items():
        print(f"  {var}: {cnt} / {len(data)} ({cnt / len(data) * 100:.1f}%)")

n_before = len(data)
data = data.dropna(axis=0, how="any").copy()
n_dropped = n_before - len(data)
if n_dropped > 0:
    print(f"Listwise deletion: dropped {n_dropped} cases ({len(data)} of {n_before} retained)")

# ── 2. Align LPA Items with Complete Rows ────────────────
lpa_items_all = pd.read_excel(os.path.join(MPLUS_DIR, "abuse_pos_only_PID_PSR_ADP.xlsx"))
lpa_items_complete = lpa_items_all.iloc[data.index].reset_index(drop=True)
data = data.reset_index(drop=True)

# ── 3. Variable Types ─────────────────────────────────────
categorical = [c for c in CATEGORICAL_CANDIDATES if c in data.columns]
feature_categorical = [c for c in categorical if c != "Cluster"]
continuous = [c for c in data.columns if c not in categorical]

# ── 4. TableOne ───────────────────────────────────────────
nonnormal = [
    col for col in continuous
    if pd.api.types.is_numeric_dtype(data[col])
    and len(data[col].dropna()) >= 8
    and stats.normaltest(data[col].dropna()).pvalue < 0.05
]
TableOne(
    data, columns=data.columns.tolist(),
    categorical=categorical, continuous=continuous,
    groupby="Cluster", nonnormal=nonnormal,
    missing=False, pval=True,
).to_excel(os.path.join(OUT_DIR, "tableone_by_cluster.xlsx"))


# ── 5. Custom Descriptive Stats ───────────────────────────
def fmt_p(p):
    return "<0.001" if p < 0.001 else round(float(p), 4)


groups = sorted(data["Cluster"].dropna().unique().tolist())
rows = []

for col in continuous:
    if not pd.api.types.is_numeric_dtype(data[col]):
        continue
    gv = [data.loc[data["Cluster"] == g, col].dropna() for g in groups]
    np_val = [stats.normaltest(x).pvalue if len(x) >= 8 else np.nan for x in gv]
    all_normal = all((not pd.isna(p)) and p > 0.05 for p in np_val)

    if all_normal:
        t = stats.f_oneway(*gv)
        method, st_name = "ANOVA", "F"
        overall = f"{data[col].mean():.2f} +/- {data[col].std(ddof=1):.2f}"
        gs = {f"Class {g}": f"{x.mean():.2f} +/- {x.std(ddof=1):.2f}"
              for g, x in zip(groups, gv)}
    else:
        t = stats.kruskal(*gv)
        method, st_name = "Kruskal-Wallis", "H"
        q25, q75 = data[col].quantile(0.25), data[col].quantile(0.75)
        overall = f"{data[col].median():.2f} ({q25:.2f}, {q75:.2f})"
        gs = {f"Class {g}": f"{x.median():.2f} ({x.quantile(0.25):.2f}, {x.quantile(0.75):.2f})"
              for g, x in zip(groups, gv)}

    row = {"Variable": col, "Type": "continuous", "Level": "",
           "Overall": overall, "Test": method, "Statistic": st_name,
           "Statistic_value": round(float(t.statistic), 4), "P_value": fmt_p(t.pvalue)}
    row.update(gs)
    for g, pv in zip(groups, np_val):
        row[f"Normaltest_P_Class{g}"] = "" if pd.isna(pv) else round(float(pv), 4)
    rows.append(row)

for col in feature_categorical:
    tmp = data[["Cluster", col]].dropna()
    if tmp.empty:
        continue
    ct = pd.crosstab(tmp[col], tmp["Cluster"])
    ct = ct[[g for g in groups if g in ct.columns]]

    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, pv, _, _ = stats.chi2_contingency(ct)
        method, st_v, pv_out = "Chi-square", round(float(chi2), 4), fmt_p(pv)
    else:
        method, st_v, pv_out = "Not tested", "", ""

    rows.append({"Variable": col, "Type": "categorical", "Level": "",
                 "Overall": "", "Test": method, "Statistic": "Chi-square" if method == "Chi-square" else "",
                 "Statistic_value": st_v, "P_value": pv_out})
    for level in sorted(tmp[col].dropna().unique()):
        row = {"Variable": "", "Type": "", "Level": level,
               "Overall": f"{(tmp[col] == level).sum()} ({(tmp[col] == level).mean() * 100:.1f}%)",
               "Test": "", "Statistic": "", "Statistic_value": "", "P_value": ""}
        for g in groups:
            s = tmp.loc[tmp["Cluster"] == g, col]
            n = (s == level).sum()
            row[f"Class {g}"] = f"{n} ({n / len(s) * 100:.1f}%)" if len(s) else ""
        rows.append(row)

pd.DataFrame(rows).to_excel(
    os.path.join(OUT_DIR, "custom_descriptive_stats_by_cluster.xlsx"), index=False
)

# ── 6. Prepare Regression Matrix ─────────────────────────
def prepare_regression_matrix(data, feature_categorical, continuous):
    x = data.drop(columns=["Cluster"]).copy()
    for col in feature_categorical:
        if col in x.columns:
            x[col] = x[col].astype("category")
    x = pd.get_dummies(x, drop_first=True).astype(float)
    x = x.loc[:, x.nunique(dropna=True) > 1]

    continuous_x = [c for c in continuous if c in x.columns]
    if continuous_x:
        sd = x[continuous_x].std(ddof=1).replace(0, np.nan)
        x[continuous_x] = (x[continuous_x] - x[continuous_x].mean()) / sd
        x = x.dropna(axis=1, how="any")
    return x


x = prepare_regression_matrix(data, feature_categorical, continuous)

# y for ordinal logistic: C1 → high risk (2), C3 → low risk (0)
y_risk = 3 - data["Cluster"].astype(int)
# y for multinomial logistic: 0‑based labels
y_label = data["Cluster"].astype(int) - 1

# ── 7. Ordered Logistic Model ─────────────────────────────
model_ord = OrderedModel(y_risk, x, distr="logit")
result_ord = model_ord.fit(method="bfgs", disp=False)

with open(os.path.join(OUT_DIR, "ordered_logistic_summary.txt"), "w", encoding="utf-8") as f:
    f.write(result_ord.summary().as_text())

ci = result_ord.conf_int()
or_table = pd.DataFrame({
    "Coef": result_ord.params,
    "SE": result_ord.bse,
    "Wald_chi_square": (result_ord.params / result_ord.bse) ** 2,
    "OR": np.exp(result_ord.params),
    "2.5%": np.exp(ci[0]),
    "97.5%": np.exp(ci[1]),
    "P_value": result_ord.pvalues,
})
or_table = or_table[~or_table.index.astype(str).str.contains("/")]
or_table["OR_95CI"] = or_table.apply(
    lambda r: f"{r['OR']:.3f} ({r['2.5%']:.3f}, {r['97.5%']:.3f})", axis=1
)
or_table["P_display"] = or_table["P_value"].apply(
    lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
)
or_table.to_excel(os.path.join(OUT_DIR, "ordered_logistic_OR.xlsx"))

ordered_display = or_table.reset_index().rename(columns={"index": "Variable"})
ordered_display = ordered_display[
    ["Variable", "Coef", "SE", "Wald_chi_square", "OR_95CI", "P_display"]
]
ordered_display.to_excel(
    os.path.join(OUT_DIR, "ordered_logistic_common_results.xlsx"), index=False
)

print("\n=== Ordered logistic regression (C1 = high risk) ===")
print(f"Log-likelihood: {result_ord.llf:.2f}")
print(or_table)

# ── 8. Ordered Logistic Forest Plot ───────────────────────
n_vars = len(or_table)
height = max(6, n_vars * 0.35)
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, height), gridspec_kw={"width_ratios": [1, 0.55]}
)
y_pos = np.arange(n_vars)[::-1]
y_lo, y_hi = -1, n_vars

for i, (var, row) in enumerate(or_table.iterrows()):
    ax1.hlines(y=y_pos[i], xmin=row["2.5%"], xmax=row["97.5%"], color="black", linewidth=1)
    ax1.plot(row["OR"], y_pos[i], "o", markersize=4, color="black")
    pl = "<0.001" if row["P_value"] < 0.001 else f"{row['P_value']:.3f}"
    ax2.text(
        0,
        y_pos[i],
        f"{row['OR']:.3f} ({row['2.5%']:.3f}, {row['97.5%']:.3f})",
        va="center",
    )
    ax2.text(0.65, y_pos[i], pl, va="center",
             fontweight="bold" if row["P_value"] < 0.05 else "normal")

ax1.axvline(x=1, color="gray", linestyle="--", linewidth=1)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(or_table.index)
ax1.set_xlabel("OR (per 1 SD for continuous)")
ax1.set_ylim(y_lo, y_hi)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.text(0, 1.01, "OR (95% CI)", transform=ax2.transAxes, fontweight="bold", va="bottom")
ax2.text(0.65, 1.01, "P-value", transform=ax2.transAxes, fontweight="bold", va="bottom")
ax2.set_xlim(0, 1)
ax2.set_ylim(y_lo, y_hi)
ax2.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ordered_logistic_OR_plot.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

# ── 9. Brant‑style Parallelism Test ───────────────────────
x_const = sm.add_constant(x, has_constant="add")
cut1 = (y_risk > 0).astype(int)   # separates C3 | C1+C2
cut2 = (y_risk > 1).astype(int)   # separates C3+C2 | C1

b1 = sm.Logit(cut1, x_const).fit(method="bfgs", maxiter=200, disp=False)
b2 = sm.Logit(cut2, x_const).fit(method="bfgs", maxiter=200, disp=False)

coef_diff = b1.params[x.columns] - b2.params[x.columns]

s1 = b1.model.score_obs(b1.params)
s2 = b2.model.score_obs(b2.params)
h1_inv = np.linalg.pinv(-b1.model.hessian(b1.params))
h2_inv = np.linalg.pinv(-b2.model.hessian(b2.params))
cov_cross = h1_inv @ (s1.T @ s2) @ h2_inv
cov_cross = cov_cross[1:, 1:]

cov1 = b1.cov_params().loc[x.columns, x.columns].to_numpy()
cov2 = b2.cov_params().loc[x.columns, x.columns].to_numpy()
cov_diff = cov1 + cov2 - cov_cross - cov_cross.T

wald_stat = float(coef_diff.T @ np.linalg.pinv(cov_diff) @ coef_diff)
wald_df = len(coef_diff)
wald_p = float(stats.chi2.sf(wald_stat, wald_df))

p_rows = [{"Test": "Global parallelism test", "Chi_square": wald_stat, "df": wald_df, "P_value": wald_p}]
for idx, var in enumerate(x.columns):
    d = float(coef_diff[var])
    v = float(cov_diff[idx, idx])
    c2 = d ** 2 / v if v > 0 else np.nan
    p_rows.append({"Test": var, "Chi_square": c2, "df": 1,
                   "P_value": stats.chi2.sf(c2, 1) if pd.notna(c2) else np.nan})

pd.DataFrame(p_rows).to_excel(os.path.join(OUT_DIR, "parallelism_test.xlsx"), index=False)
with open(os.path.join(OUT_DIR, "parallelism_test_note.txt"), "w", encoding="utf-8") as f:
    f.write(
        "Brant‑style Wald test comparing cumulative logit coefficients "
        "from the high‑risk direction (C1+C2 vs C3; C1 vs C2+C3). "
        "Small P‑values suggest violation of proportional‑odds assumption.\n"
    )

# ── 10. Multinomial Logistic Model ────────────────────────
x_const = sm.add_constant(x, has_constant="add")
model_mn = sm.MNLogit(y_label, x_const)
result_mn = model_mn.fit(method="newton", maxiter=200, disp=False)

with open(os.path.join(OUT_DIR, "multinomial_logistic_summary.txt"), "w", encoding="utf-8") as f:
    f.write(result_mn.summary().as_text())

params_mn = result_mn.params
pvs_mn = result_mn.pvalues
se_mn = result_mn.bse
ci_mn = result_mn.conf_int()

nonref_y = sorted(set(int(v) for v in result_mn.model.endog))[1:]

# Detect which MultiIndex level holds outcome category values
# Format varies across statsmodels versions:
#   (outcome_int, variable_str) or (variable_str, outcome_int) or (outcome_str, variable_str)
y_str_set = set(str(v) for v in nonref_y)
for level in range(ci_mn.index.nlevels):
    sample = set(str(v) for v in ci_mn.index.get_level_values(level).unique())
    if y_str_set.issubset(sample):
        outcome_level = level
        break

mn_rows = []
for i, outcome in enumerate(params_mn.columns):
    actual_y = nonref_y[i]
    outcome_label = actual_y + 1
    ci_by_var = ci_mn.xs(str(actual_y), level=outcome_level)
    for var in params_mn.index:
        mn_rows.append({
            "Comparison": f"Cluster {outcome_label} vs Cluster 1",
            "Variable": var,
            "Coef": params_mn.loc[var, outcome],
            "SE": se_mn.loc[var, outcome],
            "Wald_chi_square": (params_mn.loc[var, outcome] / se_mn.loc[var, outcome]) ** 2,
            "OR": np.exp(params_mn.loc[var, outcome]),
            "2.5%": np.exp(ci_by_var.loc[var, "lower"]),
            "97.5%": np.exp(ci_by_var.loc[var, "upper"]),
            "P_value": pvs_mn.loc[var, outcome],
        })

mn_table = pd.DataFrame(mn_rows)
mn_table["OR_95CI"] = mn_table.apply(
    lambda r: f"{r['OR']:.3f} ({r['2.5%']:.3f}, {r['97.5%']:.3f})", axis=1
)
mn_table["P_display"] = mn_table["P_value"].apply(
    lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
)
mn_table.to_excel(os.path.join(OUT_DIR, "multinomial_logistic_OR.xlsx"), index=False)
mn_table[
    ["Comparison", "Variable", "Coef", "SE", "Wald_chi_square", "OR_95CI", "P_display"]
].to_excel(os.path.join(OUT_DIR, "multinomial_logistic_common_results.xlsx"), index=False)

# ── 11. Export Significant Controls & Aligned Items ───────
sig_vars = or_table[or_table["P_value"] < 0.05].index.tolist()
if sig_vars:
    out = x[sig_vars].assign(Cluster=data["Cluster"].values)
else:
    out = pd.DataFrame({"Cluster": data["Cluster"].values})
out.to_csv(os.path.join(OUT_DIR, "significant_controls_with_cluster.csv"), index=False)

if lpa_items_complete is not None:
    lpa_items_complete.to_csv(
        os.path.join(OUT_DIR, "lpa_items_complete_case.csv"), index=False
    )

# ── 12. Regression Note ───────────────────────────────────
with open(os.path.join(OUT_DIR, "regression_variable_note.txt"), "w", encoding="utf-8") as f:
    f.write(
        "Continuous predictors are standardised (OR per 1 SD increase).\n"
        "Categorical predictors are dummy‑coded (first category = reference).\n"
        "Ordered logistic: Y = C1(high risk)–C2–C3(low risk), OR > 1 = higher risk.\n"
        "Multinomial logistic: reference = C1.\n"
    )

print(f"\nDone. Results saved to {OUT_DIR}")
