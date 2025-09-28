# --- # Predicting Final Position Based on Practice 1,2,3, Qualifying, and Race History ---


# --- ## Sources / - [Link to OpenF1 API](https://openf1.org/) ---


# --- ## Steps / | Step | Goal                                                                        | / | ---- | ----------- ---


# --- ## Setup ---

from google.colab import drive
drive.mount('/content/drive')

import os, textwrap

ROOT = "/content/drive/MyDrive/Programming/f1_project/f1_openf1_ml/src"
os.makedirs(ROOT, exist_ok=True)



# --- ## 3. Build a small multi-race dataset ---


# --- ### 3.1 assemble dataset ---

import sys, pandas as pd, numpy as np, time
sys.path.append("/content/drive/MyDrive/Programming/f1_project/f1_openf1_ml/src")

from f1ml.data.sessions import fetch_race_sessions, fetch_meeting_sessions
from f1ml.featurize.current_weekend import build_weekend_features

race_sessions = fetch_race_sessions(2023, 2024, include_sprints=False)
race_sessions = race_sessions.dropna(subset=["meeting_key"])
race_sessions["meeting_key"] = race_sessions["meeting_key"].astype(int)

pick = (
    race_sessions
    .sort_values(["year","location"])
    .groupby("year")
    .head(24)
    .sample(n=min(23, len(race_sessions)), random_state=42)
)

rows = []
for _, r in pick.iterrows():
    year = int(r["year"])
    mk = int(r["meeting_key"])
    ms = fetch_meeting_sessions(year, mk)
    if ms.empty:
        continue
    wk = build_weekend_features(ms)
    if wk.empty:
        continue
    wk["meeting_key"] = mk
    wk["year"] = year
    wk["location"] = r.get("location", "")
    rows.append(wk)
    print(f"built weekend: {year} / {r.get('location','')} (rows={len(wk)})")
    time.sleep(0.4)  # be polite to API

dataset = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
print(dataset.shape)
dataset.head()




# --- ### 3.2 quick cleaning & feature list ---

# drop rows with missing labels, and fill small gaps
dataset = dataset.dropna(subset=["final_position"]).copy()

# feature columns = everything except identifiers/labels
id_cols = ["driver_number","driver_key","race_session_key","meeting_key","year","location","final_position"]
feature_cols = [c for c in dataset.columns if c not in id_cols]

# fill NaNs in numeric features (rookies or missing FP sessions)
for c in feature_cols:
    if dataset[c].dtype.kind in "fc":
        dataset[c] = dataset[c].fillna(dataset[c].median())

print("n_rows:", len(dataset), "n_features:", len(feature_cols))
dataset[["driver_key","location","final_position"] + feature_cols[:6]].head()




# --- ## 4. Train a baseline model (GroupKFold by race) ---


# --- ### 4.1 fit & evaluate ---

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import top_k_accuracy_score, mean_absolute_error
import numpy as np

gkf = GroupKFold(n_splits=5)
top3, mae = [], []
X = dataset[feature_cols]
y = dataset["final_position"].astype(int)
groups = dataset["meeting_key"].astype(int)

for tr, te in gkf.split(X, groups=groups):
    clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    clf.fit(X.iloc[tr], y.iloc[tr])
    prob = clf.predict_proba(X.iloc[te])
    pred = clf.predict(X.iloc[te])
    top3.append(top_k_accuracy_score(y.iloc[te], prob, k=3))
    mae.append(mean_absolute_error(y.iloc[te], pred))

print(f"Top-3 accuracy: {np.mean(top3):.3f}")
print(f"MAE (finish position): {np.mean(mae):.2f}")




# --- ### 4.2 simple feature importances chart ---

import matplotlib.pyplot as plt
import numpy as np

clf_full = RandomForestClassifier(n_estimators=600, random_state=42, n_jobs=-1)
clf_full.fit(X, y)
imp = clf_full.feature_importances_
order = np.argsort(imp)[::-1][:15]

plt.figure(figsize=(6,5))
plt.barh([feature_cols[i] for i in order][::-1], imp[order][::-1])
plt.title("Top Feature Importances (RandomForest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()




# --- ## 5. Circuit similarity (polish + per-year normalization) ---

from f1ml.featurize.circuit_profile import build_circuit_profiles
from f1ml.data.sessions import fetch_race_sessions
from sklearn.metrics.pairwise import cosine_similarity

races = fetch_race_sessions(2023, 2024, include_sprints=False)
profiles = build_circuit_profiles(races)

# per-year z-normalization of numeric features
num_cols = profiles.select_dtypes(include="number").columns.tolist()
feat_cols = [c for c in num_cols if c != "session_key"]
X = profiles[feat_cols].copy()
X = X.groupby(profiles["year"]).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)).fillna(0.0)
sim = cosine_similarity(X, X)

# show top neighbors for one target
idx_map = {int(sk): i for i, sk in enumerate(profiles["session_key"].astype(int))}
target_sk = int(profiles.iloc[0]["session_key"])
i = idx_map[target_sk]
scores = np.argsort(sim[i])[::-1][1:6]
profiles.iloc[scores][["year","location","session_name","session_key"]]

def topk_similar_sessions(target_session_key: int, k: int = 8):
    key = int(target_session_key)
    if key not in idx_map:
        return []
    i = idx_map[key]
    order = np.argsort(sim[i])[::-1]
    # exclude self at [0]
    sims = []
    for j in order[1: k+1]:
        like_sk = int(profiles.iloc[j]["session_key"])
        sims.append((like_sk, float(sim[i, j])))
    return sims




label_col = "final_position"
id_cols = ["driver_key", "driver_number", "race_session_key", "meeting_key", "year", "location", label_col]

pre_cols = [c for c in dataset.columns if c.startswith(("fp1_","fp2_","fp3_","quali_"))]
X_base = dataset[pre_cols].copy()
y = dataset[label_col].astype(int)
groups = dataset["meeting_key"].astype(int)

# impute numeric gaps (rookies / missing FP sessions)
for c in pre_cols:
    if X_base[c].dtype.kind in "fc":
        X_base[c] = X_base[c].fillna(X_base[c].median())



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import top_k_accuracy_score, mean_absolute_error

def build_weighted_history_for_targets(train_df: pd.DataFrame,
                                       test_df: pd.DataFrame,
                                       k_sim: int = 8,
                                       recency_alpha: float = 0.0):
    """
    Returns test_df with two new columns:
      - hist_avg_finish_simk
      - hist_count_simk
    History is built ONLY from train_df (leakage-safe).
    """
    # training event table (what we can learn from the past)
    train_events = train_df[["driver_key", "final_position", "race_session_key", "meeting_key", "year"]].copy()
    train_events["race_session_key"] = train_events["race_session_key"].astype(int)

    # precompute a mapping from race_session_key -> (similar like_session_key -> weight)
    unique_targets = test_df[["meeting_key", "race_session_key"]].drop_duplicates()
    history_rows = []

    # optional recency: convert to a rough ordinal (year only here; you can do dates if available)
    year_map = {int(sk): int(yr) for sk, yr in zip(profiles["session_key"].astype(int), profiles["year"].astype(int))}

    for _, row in unique_targets.iterrows():
        target_sk = int(row["race_session_key"])
        target_mk = int(row["meeting_key"])

        # get top-k similar sessions (by session_key), exclude same meeting
        like = topk_similar_sessions(target_sk, k=k_sim)
        if not like:
            continue
        like_sks = [sk for sk, w in like]
        like_wts = {sk: w for sk, w in like}

        # restrict to training events in those similar sessions, excluding same meeting
        sub = train_events[train_events["race_session_key"].isin(like_sks)].copy()
        sub = sub[sub["race_session_key"].map(lambda sk: profiles.loc[profiles["session_key"].astype(int).eq(sk), "meeting_key"].values[0] != target_mk)]
        if sub.empty:
            continue

        # weight by similarity (and optional recency decay)
        def weight_for(sk):
            w = like_wts.get(int(sk), 0.0)
            if recency_alpha > 0.0 and target_sk in year_map and int(sk) in year_map:
                dt = max(0, year_map[target_sk] - year_map[int(sk)])  # years difference
                w *= np.exp(-recency_alpha * dt)
            return w

        sub["w"] = sub["race_session_key"].map(weight_for).astype(float)

        # aggregate per driver: weighted average past finish
        # aggregate per driver: weighted average past finish
        agg = (
            sub.groupby("driver_key")[["final_position", "w"]]
              .apply(lambda g: pd.Series({
                  "hist_avg_finish_simk": np.average(g["final_position"], weights=g["w"]) if g["w"].sum() > 0 else np.nan,
                  "hist_count_simk": int(len(g))
              }))
              .reset_index()
        )


        agg["race_session_key"] = target_sk  # tag so we can join to test rows of this race
        history_rows.append(agg)

    if not history_rows:
        # no history available (rookie season or small train set)
        out = test_df.copy()
        out["hist_avg_finish_simk"] = np.nan
        out["hist_count_simk"] = 0
        return out

    hist = pd.concat(history_rows, ignore_index=True)
    out = test_df.merge(hist, on=["driver_key", "race_session_key"], how="left")
    # global fallback for missing drivers
    global_fallback = train_events["final_position"].mean()
    out["hist_avg_finish_simk"] = out["hist_avg_finish_simk"].fillna(global_fallback)
    out["hist_count_simk"] = out["hist_count_simk"].fillna(0).astype(int)
    return out



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import top_k_accuracy_score, mean_absolute_error
import numpy as np
import pandas as pd

def run_cv_with_similarity_history(dataset: pd.DataFrame,
                                   pre_cols: list,
                                   k_sim: int = 8,
                                   recency_alpha: float = 0.0):
    gkf = GroupKFold(n_splits=5)
    top3_list, mae_list = [], []

    for tr_idx, te_idx in gkf.split(dataset, groups=dataset["meeting_key"].astype(int)):
        train_df = dataset.iloc[tr_idx].copy()
        test_df  = dataset.iloc[te_idx].copy()

        # Build history for TEST using only TRAIN (leakage-safe)
        test_aug = build_weighted_history_for_targets(
            train_df, test_df, k_sim=k_sim, recency_alpha=recency_alpha
        )

        # ----- Align columns: add neutral placeholders to TRAIN -----
        # Global fallback for history (mean finish in TRAIN)
        global_fallback = float(train_df["final_position"].mean())

        # Train features start with pre-race cols
        Xtr = train_df[pre_cols].copy()
        # Add history columns with neutral values so names match test
        Xtr["hist_avg_finish_simk"] = global_fallback   # neutral expectation
        Xtr["hist_count_simk"]      = 0                 # no specific history

        # Test features = pre + actual history features
        feat_cols = pre_cols + ["hist_avg_finish_simk", "hist_count_simk"]
        Xte = test_aug[feat_cols].copy()

        # Targets
        ytr = train_df["final_position"].astype(int)
        yte = test_aug["final_position"].astype(int)

        # Impute any NaNs (rookies etc.)
        for c in feat_cols:
            if Xtr[c].dtype.kind in "fc":
                # use TRAIN median to avoid peeking at test
                med = Xtr[c].median()
                Xtr[c] = Xtr[c].fillna(med)
                Xte[c] = Xte[c].fillna(med)

        # Ensure identical column order
        Xtr = Xtr[feat_cols]
        Xte = Xte[feat_cols]

        # Train & evaluate
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)
        pred = clf.predict(Xte)

        top3_list.append(top_k_accuracy_score(yte, prob, k=3))
        mae_list.append(mean_absolute_error(yte, pred))

    return float(np.mean(top3_list)), float(np.mean(mae_list))



top3_hist, mae_hist = run_cv_with_similarity_history(dataset, pre_cols, k_sim=8, recency_alpha=0.0)
print(f"[PRE-RACE + SIM-HISTORY] Top-3: {top3_hist:.3f} | MAE: {mae_hist:.2f}")
