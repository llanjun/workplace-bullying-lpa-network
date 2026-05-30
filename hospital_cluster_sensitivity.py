# -*- coding: utf-8 -*-
import os
import re
from difflib import get_close_matches

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel


MPLUS_DIR = "MPLUS_result"
OUT_DIR = os.path.join("Logistic_result", "hospital_cluster_sensitivity")
BEST_CLASS = 3
os.makedirs(OUT_DIR, exist_ok=True)


# 1. Check data and prepare hospital clusters
old = pd.read_excel("Original_data.xlsx")
new = pd.read_excel("Original_data_with_hospital_name.xlsx")

common_cols = [c for c in old.columns if c in new.columns]
if old.shape[0] != new.shape[0] or set(new.columns) - set(old.columns) != {"Q3"}:
    raise ValueError("Original_data_with_hospital_name.xlsx should only add Q3 to Original_data.xlsx.")
for col in common_cols:
    if not (old[col].equals(new[col])):
        raise ValueError(f"Common column differs between datasets: {col}")

abuse_items = [f"ABU{i}" for i in range(1, 7)]
exposed = new[abuse_items].ge(3).any(axis=1)
hospital_raw = new.loc[exposed, "Q3"].astype(str).str.strip().reset_index(drop=True)

mapping = pd.DataFrame({"raw_hospital": hospital_raw.value_counts().index})
mapping["n"] = mapping["raw_hospital"].map(hospital_raw.value_counts())
mapping["clean"] = mapping["raw_hospital"].astype(str).str.strip()
mapping["clean"] = mapping["clean"].str.replace(r"[ \t　?？.。，,·]", "", regex=True)
mapping["clean"] = mapping["clean"].str.replace(r"[（(【\[].*?[）)】\]]", "", regex=True)

replace_dict = {
    "人名医院": "人民医院", "醫院": "医院", "醫": "医", "医院医院": "医院",
    "攀刚": "攀钢", "攀枝花是": "攀枝花市",
    "西南医科大": "西南医科大学", "西南医科大学学": "西南医科大学",
    "附第一属": "第一附属", "附属第一": "第一附属",
    "九0三": "九〇三", "九零三": "九〇三", "903": "九〇三", "9O3": "九〇三",
    "四0四": "四〇四", "四零四": "四〇四", "404": "四〇四", "○": "〇",
    "凉山州": "凉山彝族自治州", "甘孜州": "甘孜藏族自治州",
}
for old_text, new_text in replace_dict.items():
    mapping["clean"] = mapping["clean"].str.replace(old_text, new_text, regex=False)

alias = {
    "华西医院": "四川大学华西医院",
    "华西口腔": "四川大学华西口腔医院",
    "华西口腔医院": "四川大学华西口腔医院",
    "华西第二医院": "四川大学华西第二医院",
    "四川大学华西附二院": "四川大学华西第二医院",
    "成都中医大附院": "成都中医药大学附属医院",
    "省中医": "成都中医药大学附属医院",
    "四川中医药大学附属医院": "成都中医药大学附属医院",
    "成都中西医结合医院": "成都市中西医结合医院",
    "成都市第一人民医院中西医结合医院": "成都市中西医结合医院",
    "绵阳四〇四": "四川绵阳四〇四医院",
    "绵阳四〇四医院": "四川绵阳四〇四医院",
    "绵阳市四〇四医院": "四川绵阳四〇四医院",
    "九〇三": "四川省江油市九〇三医院",
    "九〇三医院": "四川省江油市九〇三医院",
    "江油九〇三医院": "四川省江油市九〇三医院",
    "江油市九〇三医院": "四川省江油市九〇三医院",
    "363医院": "三六三医院",
    "成都三六三医院": "三六三医院",
    "成都市三六三医院": "三六三医院",
    "中航三六三医院": "三六三医院",
    "中航工业363医院": "三六三医院",
    "中航工业三六三医院": "三六三医院",
    "成都医学院附院": "成都医学院第一附属医院",
    "川北医学院": "川北医学院附属医院",
    "川贝医学院附属医院": "川北医学院附属医院",
    "泸州医学院附属医院": "西南医科大学附属医院",
    "西南医科大学第一附属医院": "西南医科大学附属医院",
    "西南医科大学学第一附属医院": "西南医科大学附属医院",
    "西南医科医院附属医院": "西南医科大学附属医院",
    "医学科学院四川省人民医院": "四川省人民医院",
    "四川省医学科学院四川省人民医院": "四川省人民医院",
    "四川省医学科学院四川省人民医院": "四川省人民医院",
    "科学研究院四川省人民医院": "四川省人民医院",
    "社会科学院四川省人民医院": "四川省人民医院",
    "人民医院医学科学院": "四川省人民医院",
    "省人民医院": "四川省人民医院",
    "人民医院": "四川省人民医院",
    "肿瘤医院": "四川省肿瘤医院",
    "科学城医院": "四川省科学城医院",
    "第四人民医院": "四川省第四人民医院",
    "南充中心医院": "南充市中心医院",
    "达州中心医院": "达州市中心医院",
    "遂宁中心医院": "遂宁市中心医院",
    "攀枝花中心医院": "攀枝花市中心医院",
    "广元市第一人民": "广元市第一人民医院",
    "广元市中医院": "广元市中医医院",
    "德阳市人民": "德阳市人民医院",
    "绵阳市第三人民": "绵阳市第三人民医院",
    "绵阳中医院": "绵阳市中医医院",
    "内江第一人民医院": "内江市第一人民医院",
    "安岳中医院": "安岳县中医医院",
    "安岳中医医院": "安岳县中医医院",
    "安岳县中医院": "安岳县中医医院",
    "眉山市中医院": "眉山市中医医院",
    "三台中医院": "三台县中医院",
    "宜宾市一医院": "宜宾市第一人民医院",
    "宜宾二医院": "宜宾市第二人民医院",
    "宜宾市二医院": "宜宾市第二人民医院",
    "自贡第四人民医院": "自贡市第四人民医院",
    "自贡市第五人人民医院": "自贡市第五人民医院",
    "阿坝州人民医院": "阿坝州人民医院",
    "阿坝州人人民医院": "阿坝州人民医院",
}
mapping["clean"] = mapping["clean"].replace(alias)

invalid = {"", "医院", "中医院", "护士", "日常", "邓仕轩", "彭薪", "2017季规培护士", "18级护士规培", "第一人民医院", "第二人民医院", "一医院"}
mapping.loc[mapping["clean"].isin(invalid), "clean"] = np.nan

for idx, value in mapping["clean"].dropna().items():
    if "四川大学华西医院" in value and not any(k in value for k in ["宜宾", "广安", "雅安", "资阳"]):
        mapping.loc[idx, "clean"] = "四川大学华西医院"
    elif "华西医院雅安" in value:
        mapping.loc[idx, "clean"] = "四川大学华西医院雅安医院"
    elif "华西医院宜宾" in value or "宜宾华西" in value:
        mapping.loc[idx, "clean"] = "四川大学华西医院宜宾医院"
    elif "华西医院资阳" in value:
        mapping.loc[idx, "clean"] = "四川大学华西医院资阳医院"
    elif "华西广安" in value or "广安华西" in value:
        mapping.loc[idx, "clean"] = "四川大学华西广安医院"
    elif "攀钢" in value:
        mapping.loc[idx, "clean"] = "攀钢集团总医院"
    elif "绵阳四〇四" in value:
        mapping.loc[idx, "clean"] = "四川绵阳四〇四医院"
    elif "成都中医药大学" in value or value == "四川省中医院":
        mapping.loc[idx, "clean"] = "成都中医药大学附属医院"

clean_counts = mapping.dropna(subset=["clean"]).groupby("clean")["n"].sum().sort_values(ascending=False)
canonical = clean_counts[clean_counts >= 5].index.tolist()
mapping["match_method"] = np.where(mapping["clean"].isin(canonical), "direct_or_alias", "low_confidence")

for idx, value in mapping.loc[mapping["match_method"] == "low_confidence", "clean"].dropna().items():
    hit = get_close_matches(value, canonical, n=1, cutoff=0.90)
    if hit:
        mapping.loc[idx, "clean"] = hit[0]
        mapping.loc[idx, "match_method"] = "fuzzy"

mapping["use_for_analysis"] = mapping["match_method"].isin(["direct_or_alias", "fuzzy"])
mapping.loc[~mapping["use_for_analysis"], "clean"] = np.nan
mapping.to_excel(os.path.join(OUT_DIR, "hospital_name_mapping_used.xlsx"), index=False)

hospital_map = dict(zip(mapping["raw_hospital"], mapping["clean"]))
hospital = hospital_raw.map(hospital_map)


# 2. Fit ordered logistic model with conventional and hospital-cluster robust SEs
data = pd.read_excel(os.path.join(MPLUS_DIR, "abuse_pos_only_demographic.xlsx"))
cluster_label = pd.read_csv(
    os.path.join(MPLUS_DIR, f"dataLPA{BEST_CLASS}.TXT"),
    sep=r"[\s,]+",
    header=None,
    engine="python",
)
data["Cluster"] = cluster_label.iloc[:, -1].astype(int).values
data["Hospital"] = hospital.values

categorical_candidates = [
    "Gender", "Rural", "LeftBy", "Religion", "Married", "Level", "Education",
    "Exposure", "WorkTime", "SleepQuality", "Alcohol", "Smoke", "Exercise",
    "ChronicDisease", "Cluster",
]
categorical = [c for c in categorical_candidates if c in data.columns]
feature_categorical = [c for c in categorical if c != "Cluster"]
model_data = data.dropna(axis=0, how="any").copy()

y = 3 - model_data["Cluster"].astype(int)
groups = model_data["Hospital"].astype(str)
x = model_data.drop(columns=["Cluster", "Hospital"]).copy()
for col in feature_categorical:
    if col in x.columns:
        x[col] = x[col].astype("category")
x = pd.get_dummies(x, drop_first=True).astype(float)
x = x.loc[:, x.nunique(dropna=True) > 1]

continuous = [c for c in x.columns if c in model_data.columns and c not in categorical]
if continuous:
    x[continuous] = (x[continuous] - x[continuous].mean()) / x[continuous].std(ddof=1)

standard = OrderedModel(y, x, distr="logit").fit(method="bfgs", disp=False)
clustered = OrderedModel(y, x, distr="logit").fit(
    method="bfgs",
    disp=False,
    cov_type="cluster",
    cov_kwds={"groups": groups},
)

standard_ci = standard.conf_int().loc[x.columns]
cluster_ci = clustered.conf_int().loc[x.columns]

result = pd.DataFrame({
    "Variable": x.columns,
    "OR": np.exp(standard.params.loc[x.columns]).values,
    "CI_low_standard": np.exp(standard_ci.iloc[:, 0]).values,
    "CI_high_standard": np.exp(standard_ci.iloc[:, 1]).values,
    "P_standard": standard.pvalues.loc[x.columns].values,
    "CI_low_cluster": np.exp(cluster_ci.iloc[:, 0]).values,
    "CI_high_cluster": np.exp(cluster_ci.iloc[:, 1]).values,
    "P_cluster": clustered.pvalues.loc[x.columns].values,
})
result["Standard_significant"] = result["P_standard"] < 0.05
result["Cluster_significant"] = result["P_cluster"] < 0.05
result["Significance_changed"] = result["Standard_significant"] != result["Cluster_significant"]
result_display = pd.DataFrame({
    "Variable": result["Variable"],
    "Standard OR (95% CI)": result.apply(
        lambda r: f"{r['OR']:.3f} ({r['CI_low_standard']:.3f}, {r['CI_high_standard']:.3f})",
        axis=1,
    ),
    "Standard P": result["P_standard"].map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"),
    "Hospital-cluster robust OR (95% CI)": result.apply(
        lambda r: f"{r['OR']:.3f} ({r['CI_low_cluster']:.3f}, {r['CI_high_cluster']:.3f})",
        axis=1,
    ),
    "Hospital-cluster robust P": result["P_cluster"].map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"),
    "Significance changed": result["Significance_changed"].map({True: "Yes", False: "No"}),
})
result_display.to_excel(os.path.join(OUT_DIR, "hospital_cluster_OR_comparison.xlsx"), index=False)

summary = pd.DataFrame([{
    "N_original": len(old),
    "N_exposed": int(exposed.sum()),
    "Raw_hospital_names": int(hospital_raw.nunique()),
    "Clean_hospital_clusters": int(groups.nunique()),
    "N_analysis": int(len(model_data)),
    "N_excluded_low_confidence_hospital": int(hospital.isna().sum()),
    "Standard_significant_predictors": int(result["Standard_significant"].sum()),
    "Cluster_significant_predictors": int(result["Cluster_significant"].sum()),
    "Changed_significance_predictors": int(result["Significance_changed"].sum()),
    "Changed_variables": ", ".join(result.loc[result["Significance_changed"], "Variable"]),
}])
summary.to_excel(os.path.join(OUT_DIR, "hospital_cluster_sensitivity_summary.xlsx"), index=False)

print(summary.to_string(index=False))
