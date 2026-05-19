# -*- coding: utf-8 -*-
import os
import re
import shutil
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


# Basic settings
MPLUS_EXE = os.environ.get("MPLUS_EXE", "mplus")  # or set to full path of mplus.exe
MAX_CLASSES = 6
BEST_CLASS = 3  # Manually selected based on entropy peaks, class interpretability,
                 # and class proportions > 5%; see lpa_model_fit_summary.csv for all fit indices.
                 # Note: LMR/BLRT remain significant (p<0.001) through 6 classes,
                 # and AIC/BIC/aBIC continue to decrease — the 3-class choice is driven
                 # by parsimony and theoretical interpretability rather than fit heuristics alone.
STARTS = "1000 250"
PROCESSORS = 8
MPLUS_DIR = "MPLUS_result"
os.makedirs(MPLUS_DIR, exist_ok=True)

ABUSE_ITEMS = ["ABU1", "ABU2", "ABU3", "ABU4", "ABU5", "ABU6"]
ABUSE_EXPOSURE_CUTOFF = 3  # 3 = Sometimes occurs; used to identify exposure to workplace bullying.
LPA_VARIABLES = (
    [f"PID{i}" for i in range(1, 10)]
    + [f"PSR{i}" for i in range(1, 15)]
    + [f"ADP{i}" for i in range(1, 14)]
)


# 1. Check Mplus executable availability
if shutil.which(MPLUS_EXE) is None and not os.path.exists(MPLUS_EXE):
    raise FileNotFoundError(
        f"Cannot find Mplus executable: {MPLUS_EXE}. "
        "Add it to PATH or set environment variable MPLUS_EXE to the full mplus.exe path."
    )

# 2. Prepare data
data = pd.read_excel("Original_data.xlsx")
data["abuse_total_score"] = data[ABUSE_ITEMS].sum(axis=1)
data["abuse_exposed"] = data[ABUSE_ITEMS].ge(ABUSE_EXPOSURE_CUTOFF).any(axis=1)

abuse_pos_data = data[data["abuse_exposed"]].copy()
lpa_data = abuse_pos_data[LPA_VARIABLES].copy()
demographic_cols = [
    col
    for col in abuse_pos_data.columns
    if col not in LPA_VARIABLES
    and not col.startswith("ABU")
    and col != "abuse_total_score"
    and col != "abuse_exposed"
]
demographic_data = abuse_pos_data[demographic_cols].copy()

lpa_data.to_excel(os.path.join(MPLUS_DIR, "abuse_pos_only_PID_PSR_ADP.xlsx"), index=False)
lpa_data.to_csv(os.path.join(MPLUS_DIR, "abuse_pos_data_used.dat"), index=False, sep=",", header=False)
demographic_data.to_excel(os.path.join(MPLUS_DIR, "abuse_pos_only_demographic.xlsx"), index=False)


# 3. Run Mplus LPA models

names_text = "\n    ".join(
    [
        " ".join(LPA_VARIABLES[0:17]),
        " ".join(LPA_VARIABLES[17:32]),
        " ".join(LPA_VARIABLES[32:]),
    ]
)

for k in range(1, MAX_CLASSES + 1):
    inp_text = f"""TITLE: Workplace violence LPA - {k} classes;
DATA:
    FILE IS abuse_pos_data_used.dat;
VARIABLE:
    NAMES ARE
    {names_text};
    MISSING ARE ALL (99);
    USEVARIABLES ARE
    {names_text};
    CLASSES = c({k});
ANALYSIS:
    TYPE = MIXTURE;
    STARTS = {STARTS};
    STITERATIONS = 20;
    PROCESSORS = {PROCESSORS};
OUTPUT:
    TECH11 TECH14;
SAVEDATA:
    FILE IS dataLPA{k}.TXT;
    SAVE = CPROB;
PLOT:
    TYPE IS PLOT3;
    SERIES = PID1-PID9 PSR1-PSR14 ADP1-ADP13 (*);
"""

    inp_file = f"lpa_{k}class.inp"
    with open(os.path.join(MPLUS_DIR, inp_file), "w", encoding="utf-8") as f:
        f.write(inp_text)

    print(f"Running Mplus: {inp_file}")
    subprocess.run([MPLUS_EXE, inp_file], cwd=MPLUS_DIR, check=True)

    # Check Mplus output for warnings
    out_file_path = os.path.join(MPLUS_DIR, f"lpa_{k}class.out")
    if os.path.exists(out_file_path):
        with open(out_file_path, "r", encoding="utf-8", errors="ignore") as f:
            out_content = f.read()
        warnings_found = re.findall(r"\*\*\*\s*WARNING\s*(.*?)(?=\n\n|\*\*\*|\Z)", out_content, re.DOTALL | re.IGNORECASE)
        for w in warnings_found:
            w_clean = " ".join(w.strip().split())
            print(f"  Mplus warning ({k}-class): {w_clean}")


# 4. Extract model fit table
fit_rows = []
for k in range(1, MAX_CLASSES + 1):
    out_file = os.path.join(MPLUS_DIR, f"lpa_{k}class.out")
    if not os.path.exists(out_file):
        print(f"Warning: missing {out_file}")
        continue

    with open(out_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    row = {"Class": k}
    for col, pattern in {
        "AIC": r"Akaike \(AIC\)\s+(-?[\d.]+)",
        "BIC": r"Bayesian \(BIC\)\s+(-?[\d.]+)",
        "aBIC": r"Sample-Size Adjusted BIC\s+(-?[\d.]+)",
        "Entropy": r"Entropy\s+(-?[\d.]+)",
    }.items():
        m = re.search(pattern, content)
        row[col] = float(m.group(1)) if m else None

    for col, pattern in {
        "VLMR_P": r"VUONG-LO-MENDELL-RUBIN LIKELIHOOD RATIO TEST.*?P-Value\s+([\d.]+)",
        "LMR_P": r"LO-MENDELL-RUBIN ADJUSTED LRT TEST.*?P-Value\s+([\d.]+)",
        "BLRT_P": r"PARAMETRIC BOOTSTRAPPED LIKELIHOOD RATIO TEST.*?(?:Approximate\s+)?P-Value\s+([\d.]+)",
    }.items():
        m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        row[col] = float(m.group(1)) if m else None

    block = re.search(
        r"FINAL CLASS COUNTS AND PROPORTIONS FOR THE LATENT CLASSES\s+"
        r"BASED ON THEIR MOST LIKELY LATENT CLASS MEMBERSHIP(.*?)(?:CLASSIFICATION QUALITY|Entropy)",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if block:
        for line in block.group(1).splitlines():
            m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                row[f"Class{m.group(1)}_N"] = float(m.group(2))
                row[f"Class{m.group(1)}_Prop"] = float(m.group(3))

    # Check for BLRT warning: unreliable p-value due to local maxima
    blrt_warning = re.search(
        r"P-VALUE MAY NOT BE TRUSTWORTHY DUE TO LOCAL MAXIMA",
        content,
        re.IGNORECASE,
    )
    row["BLRT_Warning"] = "Yes" if blrt_warning else ""

    # Check for other Mplus warnings in this output
    model_warnings = re.findall(r"\*\*\* WARNING.*?(?:\n\s*\*\*\*|\Z)", content, re.DOTALL | re.IGNORECASE)
    row["Model_Warnings"] = "; ".join(
        " ".join(w.replace("*** WARNING", "").strip().split())
        for w in model_warnings
    ) if model_warnings else ""

    fit_rows.append(row)

fit_table = pd.DataFrame(fit_rows)
fit_table.to_excel(os.path.join(MPLUS_DIR, "lpa_model_fit_summary.xlsx"), index=False)
print("\n=== LPA model fit summary ===")
print(fit_table)


# 5. Read selected class solution and make trajectory tables
cprob_cols = [f"CPROB{i}" for i in range(1, BEST_CLASS + 1)]
lpa_savedata = pd.read_csv(
    os.path.join(MPLUS_DIR, f"dataLPA{BEST_CLASS}.TXT"),
    sep=r"\s+",
    header=None,
    names=LPA_VARIABLES + cprob_cols + ["Cluster"],
    engine="python",
)

lpa_savedata.to_excel(os.path.join(MPLUS_DIR, f"lpa_classification_{BEST_CLASS}class.xlsx"), index=False)

observed_means = lpa_savedata.groupby("Cluster")[LPA_VARIABLES].mean().sort_index()
class_counts = lpa_savedata["Cluster"].value_counts().sort_index()
observed_means.insert(0, "N", class_counts)
observed_means.insert(1, "Percent", class_counts / len(lpa_savedata))
observed_means.to_excel(os.path.join(MPLUS_DIR, f"trajectory_observed_means_{BEST_CLASS}class.xlsx"))


# 6. Extract Mplus estimated class means from selected .out
with open(os.path.join(MPLUS_DIR, f"lpa_{BEST_CLASS}class.out"), "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

mplus_means = {}
for class_block in re.finditer(
    r"Latent Class\s+(\d+).*?\n\s*Means\s*\n(.*?)(?:\n\s*Variances|\n\s*Latent Class|\n\s*Categorical Latent Variables)",
    content,
    re.DOTALL | re.IGNORECASE,
):
    cls = int(class_block.group(1))
    mplus_means[cls] = {}
    for line in class_block.group(2).splitlines():
        m = re.match(r"\s*([A-Z]+\d+)\s+(-?[\d.]+)", line)
        if m and m.group(1) in LPA_VARIABLES:
            mplus_means[cls][m.group(1)] = float(m.group(2))

if mplus_means:
    pd.DataFrame.from_dict(mplus_means, orient="index").sort_index().to_excel(
        os.path.join(MPLUS_DIR, f"trajectory_mplus_estimated_means_{BEST_CLASS}class.xlsx")
    )


# 7. Plot trajectory figure
colors = ["#E41A1C", "#4DAF4A", "#377EB8", "#984EA3", "#FF7F00", "#A65628"]
plt.figure(figsize=(12, 6))

for i, (cluster, row) in enumerate(observed_means[LPA_VARIABLES].iterrows()):
    pct = observed_means.loc[cluster, "Percent"] * 100
    n = int(observed_means.loc[cluster, "N"])
    plt.plot(
        LPA_VARIABLES,
        row.values,
        marker=".",
        linewidth=1.8,
        color=colors[i % len(colors)],
        label=f"Class {int(cluster)} (n={n}, {pct:.1f}%)",
    )

plt.xticks(rotation="vertical", fontsize=10)
plt.xlabel(None)
plt.ylabel("Mean Values", fontsize=12)
plt.legend(loc="lower right", frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(MPLUS_DIR, f"cluster_plot_{BEST_CLASS}class.png"), dpi=600, bbox_inches="tight", pad_inches=0.5)
plt.show()

print(f"\nDone. Used {BEST_CLASS}-class solution for classification table and plot.")
