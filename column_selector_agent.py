#!/usr/bin/env python3
"""
column_selector_agent.py
Final — Combined Clinical Mode (Domain-aware, life-sciences friendly aligned table)

Usage:
    python column_selector_agent.py --input .\datasets\cirrhosis.csv --output .\out_cirrhosis --keep_fraction 0.6

Outputs:
 - selected_columns.json
 - metadata.json (now includes "__target__")
 - report.txt  (beautiful aligned Markdown table with biological explanations only)
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

random.seed(0)

# -------------------------
# Domain tokens (expanded)
# -------------------------
LIFE_SCIENCE_TOKENS = {
    "oncology": ["tumor", "radius", "perimeter", "texture", "area", "necrosis", "concav", "malign"],
    "cardiology": ["chol", "age", "trestbps", "thal", "exang", "oldpeak", "cp", "thalach", "ecg", "heart", "troponin"],
    "hepatology": ["bilirubin", "sgot", "sgpt", "alk", "albumin", "ascites", "hepato", "cirrho", "liver", "prothrombin", "inr"],
    "diabetes": ["glucose", "insulin", "bmi", "hba1c", "preg"],
    "drug": ["smiles", "inchi", "molecular", "cid", "pic50", "logp", "molweight"],
    "genomics": ["gene", "expression", "rna", "dna", "variant", "snp"],
    "microbiology": ["colony", "culture", "pathogen", "strain", "isolate"],
    "nutrition": ["fat", "protein", "vit", "carb", "mineral", "calorie"]
}

DOMAIN_DESCRIPTIONS = {
    "oncology": "Radiomic and tumor-level descriptors linked to invasiveness, malignancy, and prognosis.",
    "cardiology": "Cardiac biomarkers and vital signs influencing ischemic disease and functional capacity.",
    "hepatology": "Liver function markers reflecting hepatocyte injury, cholestasis, and hepatic decompensation.",
    "diabetes": "Metabolic indicators reflecting glycemic control and insulin resistance.",
    "drug": "Chemical descriptors relevant to potency and ADMET behavior.",
    "genomics": "Gene expression and variant features representing biological pathway activity.",
    "microbiology": "Pathogen/strain attributes linked to virulence and antibiotic response.",
    "nutrition": "Nutrient markers affecting metabolic status and energy balance."
}

# -------------------------
# Helper Functions
# -------------------------
def detect_domain(columns):
    scores = defaultdict(int)
    for c in columns:
        lc = c.lower()
        for d, toks in LIFE_SCIENCE_TOKENS.items():
            for t in toks:
                if t in lc:
                    scores[d] += 1
    if not scores: return "general"
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

def identify_target(columns, df):
    lower = [c.lower() for c in columns]
    preferred = ["target", "diagnosis", "outcome", "label", "stage", "status", "class"]
    for t in preferred:
        if t in lower:
            return columns[lower.index(t)]
    for c in columns:
        if 2 <= df[c].nunique() <= 10:
            return c
    return columns[-1]

def encode_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.values.astype(float)
    return LabelEncoder().fit_transform(series.astype(str))

# -------------------------
# Internal scoring (MI + RF + assoc)
# -------------------------
def compute_scores(df, target, task_type):
    mi, rf, assoc = {}, {}, {}

    y = encode_series(df[target])

    # MI
    for c in df.columns:
        if c == target: continue
        try:
            x = encode_series(df[c]).reshape(-1,1)
            if task_type=="classification":
                sc = mutual_info_classif(x, y, random_state=0)
            else:
                sc = mutual_info_regression(x, y, random_state=0)
            mi[c] = float(sc[0])
        except:
            mi[c] = 0.0

    # RF
    try:
        X = df.drop(columns=[target])
        X_enc = pd.DataFrame({c: encode_series(df[c]) for c in X.columns})
        Xtr, Xts, ytr, yts = train_test_split(X_enc, y, test_size=0.25, random_state=0)
        if task_type=="classification":
            model = RandomForestClassifier(n_estimators=100, random_state=0)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(Xtr, ytr)
        for c, imp in zip(X_enc.columns, model.feature_importances_):
            rf[c] = float(imp)
    except:
        rf = {c:0.0 for c in df.columns if c!=target}

    # Assoc
    for c in df.columns:
        if c==target: continue
        try:
            x = df[c]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(df[target]):
                corr = abs(x.corr(df[target]))
                assoc[c] = 0.0 if np.isnan(corr) else corr
            else:
                xv = encode_series(df[c]).reshape(-1,1)
                if task_type=="classification":
                    sc = mutual_info_classif(xv, y, random_state=0)
                else:
                    sc = mutual_info_regression(xv, y, random_state=0)
                assoc[c] = float(sc[0])
        except:
            assoc[c] = 0.0

    return mi, rf, assoc

# -------------------------
# Biological explanation rules
# -------------------------
TEMPLATES_SEL = {
    "oncology": [
        "{} reflects tumor structural characteristics influencing malignancy behavior.",
        "{} captures radiomic heterogeneity, often associated with aggressive tumor biology.",
        "{} represents tumor size or boundary complexity relevant to staging."
    ],
    "cardiology": [
        "{} indicates cardiovascular stress or perfusion, shaping cardiac risk.",
        "{} relates to heart function or ischemic response influencing clinical outcomes."
    ],
    "hepatology": [
        "{} reflects hepatic excretory or synthetic impairment, mapping to cirrhosis severity.",
        "{} signals portal hypertension or hepatocellular injury relevant to patient prognosis.",
        "{} indicates deranged liver processing capacity, often preceding clinical deterioration."
    ],
    "diabetes": [
        "{} represents metabolic imbalance or glucose control influencing disease progression.",
        "{} reflects insulin resistance or beta-cell dysfunction linked to long-term risk."
    ],
    "general": [
        "{} shows biological variation relevant to the underlying clinical outcome."
    ]
}

TEMPLATES_REJ = {
    "oncology": [
        "{} overlaps with stronger tumor descriptors and adds limited new biological insight.",
    ],
    "cardiology": [
        "{} is lightly tied to cardiovascular physiology compared to retained markers.",
    ],
    "hepatology": [
        "{} does not directly reflect liver pathophysiology and was deprioritized.",
    ],
    "general": [
        "{} contributes minimal direct biological interpretation relative to selected markers."
    ]
}

def choose_reason(col, domain, status):
    lc = col.lower()

    if status=="Selected":
        if domain == "hepatology":
            if "bilirubin" in lc:
                return f"{col} reflects impaired bile excretion; elevation signals worsening hepatic function."
            if "ascites" in lc:
                return f"{col} indicates fluid accumulation from portal hypertension, marking liver decompensation."
            if "albumin" in lc:
                return f"{col} measures hepatic protein synthesis; low levels reflect advanced liver dysfunction."
            if "inr" in lc or "prothrombin" in lc:
                return f"{col} reflects clotting factor synthesis; prolongation suggests hepatic synthetic failure."

        if domain == "oncology":
            if "radius" in lc or "area" in lc or "perimeter" in lc:
                return f"{col} represents tumor size/shape, often correlating with stage and aggressiveness."
            if "texture" in lc or "entropy" in lc:
                return f"{col} measures radiomic heterogeneity, which is associated with aggressive tumor biology."

        templates = TEMPLATES_SEL.get(domain, TEMPLATES_SEL["general"])
        return random.choice(templates).format(col)

    else:
        if "id" in lc or lc.startswith("unnamed"):
            return f"{col} is an identifier without biological relevance."
        templates = TEMPLATES_REJ.get(domain, TEMPLATES_REJ["general"])
        return random.choice(templates).format(col)

# -------------------------
# Table builder with alignment
# -------------------------
def build_markdown_table(selected, rejected, domain):
    rows = []

    # Selected
    for c in selected:
        if c.lower() == selected[0].lower():
            status = "Selected (TARGET)"
            reason = "This is the main outcome variable for clinical interpretation."
        else:
            status = "Selected"
            reason = choose_reason(c, domain, "Selected")
        rows.append([c, status, reason])

    # Rejected
    for c in rejected:
        status = "Rejected"
        reason = choose_reason(c, domain, "Rejected")
        rows.append([c, status, reason])

    # Column widths
    col_widths = [
        max(len(str(row[i])) for row in rows + [["Column","Status","Biological Reason"]])
        for i in range(3)
    ]

    header = "| {0:<{w0}} | {1:<{w1}} | {2:<{w2}} |".format(
        "Column","Status","Biological Reason",
        w0=col_widths[0], w1=col_widths[1], w2=col_widths[2]
    )

    divider = "|-{0}-|-{1}-|-{2}-|".format(
        "-"*col_widths[0], "-"*col_widths[1], "-"*col_widths[2]
    )

    lines = [header, divider]

    for r in rows:
        lines.append(
            "| {0:<{w0}} | {1:<{w1}} | {2:<{w2}} |".format(
                r[0], r[1], r[2],
                w0=col_widths[0], w1=col_widths[1], w2=col_widths[2]
            )
        )

    return "\n".join(lines)

# -------------------------
# Feature Selection
# -------------------------
def select_features(df, target, keep_fraction=0.6):
    y = df[target]
    task_type = "classification" if not (pd.api.types.is_numeric_dtype(y) and y.nunique()>20) else "regression"

    mi, rf, assoc = compute_scores(df, target, task_type)

    # Normalize & combine
    combined = {}
    max_mi, max_rf, max_assoc = max(mi.values()), max(rf.values()), max(assoc.values())
    for c in mi.keys():
        nm = mi[c]/max_mi if max_mi>0 else 0
        nr = rf[c]/max_rf if max_rf>0 else 0
        na = assoc[c]/max_assoc if max_assoc>0 else 0
        combined[c] = 0.45*nm + 0.45*nr + 0.10*na

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    keep_k = max(3, int(len(ranked)*keep_fraction))
    selected = [target] + [c for c,_ in ranked[:keep_k] if c!=target]

    # Include high-association features
    for c,a in assoc.items():
        if a>0.3 and c not in selected:
            selected.append(c)

    return selected

# -------------------------
# Main analysis
# -------------------------
def analyze_file(path, keep_fraction=0.6):
    df = pd.read_csv(path)
    raw_cols = list(df.columns)

    domain = detect_domain(raw_cols)
    domain_desc = DOMAIN_DESCRIPTIONS.get(domain,"General life-science dataset")

    target = identify_target(raw_cols, df)

    if df[target].nunique() == len(df):
        for c in raw_cols:
            if df[c].nunique() < len(df)*0.5:
                target = c
                break

    selected = select_features(df, target, keep_fraction)
    rejected = [c for c in raw_cols if c not in selected]

    # Metadata (with TARGET stored)
    metadata = {"__target__": target}
    for c in selected:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            metadata[c] = {
                "type":"numeric",
                "min":None if s.isnull().all() else float(s.min()),
                "max":None if s.isnull().all() else float(s.max()),
                "mean":None if s.isnull().all() else float(s.mean()),
                "missing":int(s.isnull().sum())
            }
        else:
            metadata[c] = {
                "type":"categorical",
                "unique":int(s.nunique()),
                "top_values":s.astype(str).value_counts().head(5).index.tolist(),
                "missing":int(s.isnull().sum())
            }

    # Build final report
    header = (
        "======================================================================\n"
        "Life-Sciences Column Selection Report — Combined Clinical Mode\n"
        f"Dataset: {os.path.abspath(path)}\n"
        f"Detected domain: {domain} — {domain_desc}\n"
        f"Rows: {len(df)}   Columns (raw): {len(raw_cols)}\n\n"
        "Selection Table:\n\n"
    )

    table = build_markdown_table(selected, rejected, domain)

    footer = (
        "\n\nNotes:\n"
        "- Report is clinician-friendly (biology-based reasoning only).\n"
        "- Numeric stats stored separately in metadata.json.\n"
        "======================================================================"
    )

    report_text = header + table + footer

    return selected, rejected, metadata, report_text, target

# -------------------------
# CLI entry point
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep_fraction", type=float, default=0.6)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    selected, rejected, metadata, report_text, target = analyze_file(
        args.input, keep_fraction=args.keep_fraction
    )

    with open(os.path.join(args.output,"selected_columns.json"),"w") as f:
        json.dump(selected,f,indent=4)

    with open(os.path.join(args.output,"metadata.json"),"w") as f:
        json.dump(metadata,f,indent=4)

    with open(os.path.join(args.output,"report.txt"),"w",encoding="utf-8") as f:
        f.write(report_text)

    print("====================================================")
    print(f"Dataset: {args.input}")
    print(f"Target column: {target}")
    print("Selected Columns:")
    for c in selected:
        print(f"  - {c}")
    print("====================================================")
    print(f"Saved: selected_columns.json, metadata.json, report.txt → {args.output}")

if __name__ == "__main__":
    main()
