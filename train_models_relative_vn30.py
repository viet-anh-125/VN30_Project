"""Huấn luyện V2: target vượt VN-Index, chọn model theo Top-3 2021-2024.

Bổ sung theo nhận xét GVHD (11/09/2026):
- AUC-ROC (gộp toàn bộ quan sát và trung bình theo ngày) cho từng fold,
  tổng hợp 2021-2024 và final test 2025 -> Bảng 3.2 / 3.3.
- Bảng hệ số Hồi quy Logistic đã chuẩn hoá (StandardScaler) -> Bảng 3.4,
  dùng làm căn cứ cho nhận định ở mục 4.2. Hệ số luôn được xuất, không phụ
  thuộc mô hình cuối cùng được chọn là gì, vì đây là công cụ diễn giải.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier

from relative_config_vn30 import (
    DATA_DIR, TARGET_HORIZON, TARGET_COL, MODEL_FEATURES_V2, TICKERS, SUFFIX, out_name,
)

INPUT = DATA_DIR / out_name("vn30_model_dataset_relative_2020_2025")
OUT_FOLDS = DATA_DIR / out_name("relative_walk_forward_results_2021_2024")
OUT_PRED = DATA_DIR / out_name("relative_walk_forward_predictions_2021_2024")
OUT_COMPARE = DATA_DIR / out_name("relative_model_comparison_2021_2024")
OUT_TEST = DATA_DIR / out_name("relative_test_metrics_2025")
OUT_TEST_PRED = DATA_DIR / out_name("relative_test_predictions_2025")
OUT_MODEL = DATA_DIR / f"relative_model_best{SUFFIX}.joblib"
OUT_COEF = DATA_DIR / out_name("relative_logistic_coefficients")
YEARS = [2021, 2022, 2023, 2024]


def models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
    }


def remove_last_sessions(df, n=TARGET_HORIZON):
    ordered = df.sort_values(["symbol", "time"]).copy()
    distance_from_end = ordered.groupby("symbol").cumcount(ascending=False)
    return ordered.loc[distance_from_end >= n].reset_index(drop=True)


def classification(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def auc_metrics(frame: pd.DataFrame) -> dict:
    """AUC-ROC gộp toàn bộ quan sát và AUC trung bình theo ngày.

    AUC gộp trộn lẫn các ngày với nhau nên bị ảnh hưởng bởi tỷ lệ target
    thay đổi theo thời gian; AUC trung bình theo ngày sát hơn với bài toán
    thực tế là xếp hạng CÙNG một ngày, nên đây là chỉ tiêu chính được dùng
    để bàn luận trong báo cáo. Ngày nào chỉ có một lớp (toàn 0 hoặc toàn 1)
    thì bỏ qua vì AUC không xác định.
    """
    y, p = frame["actual"], frame["probability_up"]
    pooled = float(roc_auc_score(y, p)) if y.nunique() == 2 else np.nan

    daily = (
        frame.groupby("time")
        .apply(lambda g: roc_auc_score(g["actual"], g["probability_up"])
               if g["actual"].nunique() == 2 else np.nan)
        .dropna()
    )
    return {
        "auc_pooled": pooled,
        "auc_daily_mean": float(daily.mean()) if len(daily) else np.nan,
        "auc_daily_days": int(len(daily)),
    }


def top3_metrics(pred_df):
    selected = (pred_df.sort_values(["time", "probability_up"], ascending=[True, False])
                       .groupby("time", group_keys=False).head(3))
    precision3 = selected["actual"].mean()
    prevalence = pred_df["actual"].mean()
    return {
        "precision_at_3": precision3,
        "market_prevalence": prevalence,
        "aggregate_lift_at_3": precision3 / prevalence if prevalence else np.nan,
        "mean_top3_future_excess_return": selected["future_excess_return_20d"].mean(),
    }


def prediction_frame(eval_df, model, name, fold, year):
    prob = model.predict_proba(eval_df[MODEL_FEATURES_V2])[:, 1]
    pred = (prob >= 0.5).astype(int)
    out = eval_df[["time", "symbol", "close", "future_excess_return_20d", TARGET_COL]].copy()
    out = out.rename(columns={TARGET_COL: "actual"})
    out["fold"] = fold
    out["validation_year"] = year
    out["model"] = name
    out["prediction"] = pred
    out["probability_up"] = prob
    return out


def export_logistic_coefficients(train: pd.DataFrame) -> pd.DataFrame:
    """Huấn luyện riêng một Hồi quy Logistic trên toàn bộ train (trước 2025)
    chỉ để lấy bảng hệ số diễn giải — không phụ thuộc model nào được chọn
    làm model cuối cùng.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, random_state=42)),
    ])
    pipe.fit(train[MODEL_FEATURES_V2], train[TARGET_COL])
    coef = pipe.named_steps["model"].coef_[0]
    table = pd.DataFrame({
        "dac_trung": MODEL_FEATURES_V2,
        "he_so_chuan_hoa": coef,
    })
    table["odds_ratio"] = np.exp(table["he_so_chuan_hoa"])
    table["do_lon_tuyet_doi"] = table["he_so_chuan_hoa"].abs()
    table = table.sort_values("do_lon_tuyet_doi", ascending=False).drop(columns="do_lon_tuyet_doi")
    table.to_csv(OUT_COEF, index=False, encoding="utf-8-sig")
    return table


def main():
    print("=" * 80)
    print("TRAIN V2 - TARGET VƯỢT VN-INDEX | CHỌN THEO TOP-3")
    print(f"Universe: {len(TICKERS)} mã")
    print("=" * 80)
    df = pd.read_csv(INPUT)
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    required = set(MODEL_FEATURES_V2 + [TARGET_COL, "future_excess_return_20d"])
    if not required.issubset(df.columns):
        raise ValueError(f"Thiếu cột: {sorted(required-set(df.columns))}")
    print(f"Rows={len(df):,}; Symbols={df.symbol.nunique()}; Features={len(MODEL_FEATURES_V2)}")

    fold_rows, pred_parts = [], []
    for fold, year in enumerate(YEARS, 1):
        train_before = df[df.time.dt.year < year].copy()
        train = remove_last_sessions(train_before)
        valid_full = df[df.time.dt.year == year].copy()
        valid = remove_last_sessions(valid_full)
        if train.time.max() >= valid.time.min():
            raise ValueError("Temporal ordering fail.")
        print(f"\nFold {fold} ({year}): train={len(train):,}; validation={len(valid):,}")
        for name, model in models().items():
            model.fit(train[MODEL_FEATURES_V2], train[TARGET_COL])
            p = prediction_frame(valid, model, name, fold, year)
            row = {"fold": fold, "validation_year": year, "model": name,
                   "train_rows": len(train), "validation_rows": len(valid)}
            row.update(classification(p.actual, p.prediction))
            row.update(auc_metrics(p))
            row.update(top3_metrics(p))
            fold_rows.append(row)
            pred_parts.append(p)
            print(f"  {name:20s} P@3={row['precision_at_3']:.4f} "
                  f"Lift@3={row['aggregate_lift_at_3']:.4f} "
                  f"AUC(ngày)={row['auc_daily_mean']:.4f} "
                  f"Excess20={row['mean_top3_future_excess_return']:.4f}")

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_parts, ignore_index=True)
    comparison_rows = []
    for name, group in predictions.groupby("model"):
        m = classification(group.actual, group.prediction)
        m.update(auc_metrics(group))
        m.update(top3_metrics(group))
        m["model"] = name
        comparison_rows.append(m)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["aggregate_lift_at_3", "mean_top3_future_excess_return", "f1"],
        ascending=False,
    ).reset_index(drop=True)
    best = comparison.iloc[0].model
    print("\nTỔNG HỢP 2021-2024:")
    print(comparison.to_string(index=False))
    print(f"\n=> Model V2 được chọn: {best}")

    train_before = df[df.time.dt.year < 2025].copy()
    train = remove_last_sessions(train_before)
    test = df[df.time.dt.year == 2025].copy()
    final_model = models()[best]
    final_model.fit(train[MODEL_FEATURES_V2], train[TARGET_COL])
    test_pred = prediction_frame(test, final_model, best, 5, 2025)
    test_metrics = classification(test_pred.actual, test_pred.prediction)
    test_metrics.update(auc_metrics(test_pred))
    test_metrics.update(top3_metrics(test_pred))
    test_metrics.update({"model": best, "rows": len(test_pred)})
    print("\nFINAL TEST 2025:")
    print(pd.DataFrame([test_metrics]).to_string(index=False))

    coef_table = export_logistic_coefficients(train)
    print("\nHỆ SỐ HỒI QUY LOGISTIC (10 đặc trưng |hệ số| lớn nhất):")
    print(coef_table.head(10).to_string(index=False))

    folds.to_csv(OUT_FOLDS, index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_COMPARE, index=False, encoding="utf-8-sig")
    pd.DataFrame([test_metrics]).to_csv(OUT_TEST, index=False, encoding="utf-8-sig")
    test_pred.to_csv(OUT_TEST_PRED, index=False, encoding="utf-8-sig")
    joblib.dump({
        "model": final_model, "model_name": best,
        "feature_cols": MODEL_FEATURES_V2, "target_col": TARGET_COL,
        "target_horizon": TARGET_HORIZON,
        "selection_rule": "aggregate_lift_at_3 -> mean_top3_future_excess_return -> f1",
        "comparison_2021_2024": comparison,
        "universe_tickers": TICKERS,
    }, OUT_MODEL)
    print(f"\n✓ Đã lưu toàn bộ output relative_*{SUFFIX} trong data_raw.")
    print(f"✓ {OUT_COEF}")


if __name__ == "__main__":
    main()
