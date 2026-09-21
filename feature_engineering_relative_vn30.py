"""Feature Engineering V2: dự báo cổ phiếu vượt VN-Index sau 20 phiên.

Universe của lần chạy này (30 mã cố định hoặc 23 mã giao nhau) được đọc
từ relative_config_vn30.TICKERS, quyết định bởi biến môi trường
VN30_UNIVERSE_MODE (xem research_config.py).
"""
import numpy as np
import pandas as pd

import feature_engineering_vn30 as base
from relative_config_vn30 import (
    DATA_DIR, TARGET_HORIZON, TARGET_COL, MODEL_FEATURES_V2, TICKERS, out_name,
)

RAW_STOCK = DATA_DIR / "vn30_ohlcv_2020_2025.csv"
RAW_INDEX = DATA_DIR / "vnindex_ohlcv_2020_2025.csv"
OUT_FULL = DATA_DIR / out_name("vn30_features_relative_2020_2025")
OUT_MODEL = DATA_DIR / out_name("vn30_model_dataset_relative_2020_2025")


def prepare_market(index_df):
    x = index_df.sort_values("time").copy()
    x["market_return_5d"] = x["close"].pct_change(5)
    x["market_return_20d"] = x["close"].pct_change(20)
    x["market_volatility_20d"] = x["close"].pct_change().rolling(20).std()
    x["market_close_to_ma_50"] = x["close"] / x["close"].rolling(50).mean() - 1
    x["future_market_return_20d"] = x["close"].shift(-TARGET_HORIZON) / x["close"] - 1
    return x[[
        "time", "market_return_5d", "market_return_20d",
        "market_volatility_20d", "market_close_to_ma_50",
        "future_market_return_20d",
    ]]


def main():
    print("=" * 78)
    print("FEATURE ENGINEERING V2 - TARGET VƯỢT VN-INDEX")
    print(f"Universe        : {len(TICKERS)} mã ({'fixed30' if len(TICKERS) == 30 else 'core23'})")
    print("=" * 78)
    stocks = pd.read_csv(RAW_STOCK)
    index_df = pd.read_csv(RAW_INDEX)
    for d in (stocks, index_df):
        d["time"] = pd.to_datetime(d["time"], errors="raise")
    stocks["symbol"] = stocks["symbol"].str.upper().str.strip()

    # Lọc theo universe của lần chạy hiện tại. Dữ liệu giá thô (data_raw)
    # được thu thập một lần cho toàn bộ 30 mã; universe chỉ quyết định mã
    # nào tham gia bước feature/huấn luyện/đánh giá của lần chạy này.
    missing_tickers = sorted(set(TICKERS) - set(stocks["symbol"].unique()))
    if missing_tickers:
        raise ValueError(f"Thiếu dữ liệu giá cho các mã: {missing_tickers}")
    stocks = stocks[stocks["symbol"].isin(TICKERS)].copy()

    if stocks.duplicated(["time", "symbol"]).any():
        raise ValueError("Dữ liệu cổ phiếu trùng time-symbol.")

    parts = []
    for symbol, group in stocks.groupby("symbol", sort=True):
        print(f"  -> {symbol}")
        parts.append(base.build_features_for_symbol(group.copy(), index_df.copy()))
    full = pd.concat(parts, ignore_index=True)

    market = prepare_market(index_df)
    # Loại các cột audit cũ để merge lại một nguồn thị trường duy nhất.
    full = full.drop(columns=["future_market_return_20d", "future_excess_return_20d"], errors="ignore")
    full = full.merge(market, on="time", how="left", validate="many_to_one")
    full["excess_return_5d"] = full["return_5d"] - full["market_return_5d"]
    full["excess_return_20d"] = full["return_20d"] - full["market_return_20d"]
    full["future_excess_return_20d"] = (
        full["future_return_20d"] - full["future_market_return_20d"]
    )
    valid_target = full[["future_return_20d", "future_market_return_20d"]].notna().all(axis=1)
    full[TARGET_COL] = np.nan
    full.loc[valid_target, TARGET_COL] = (
        full.loc[valid_target, "future_excess_return_20d"] > 0
    ).astype(int)

    ranks = {
        "return_20d": "rank_return_20d",
        "rsi_14": "rank_rsi_14",
        "volatility_20d": "rank_volatility_20d",
        "close_to_ma_20": "rank_close_to_ma_20",
        "volume_to_ma_20": "rank_volume_to_ma_20",
    }
    for source, target in ranks.items():
        # Xếp hạng theo phần trăm TRONG universe của lần chạy này (30 hoặc 23
        # mã) — nhất quán với cách bài toán được đặt ra: so sánh các mã đang
        # được sàng lọc với nhau, không phải với toàn thị trường.
        full[target] = full.groupby("time")[source].rank(pct=True, method="average")

    forbidden = {"future_return_20d", "future_market_return_20d", "future_excess_return_20d", TARGET_COL}
    overlap = forbidden.intersection(MODEL_FEATURES_V2)
    if overlap:
        raise ValueError(f"DATA LEAKAGE: {sorted(overlap)} nằm trong feature.")

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_FULL, index=False, encoding="utf-8-sig")
    keep = ["time", "symbol", "close", "future_excess_return_20d"] + MODEL_FEATURES_V2 + [TARGET_COL]
    model = full[keep].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model[TARGET_COL] = model[TARGET_COL].astype(int)
    if model.duplicated(["time", "symbol"]).any():
        raise ValueError("Dataset V2 trùng time-symbol.")
    if model[MODEL_FEATURES_V2].isna().any().any():
        raise ValueError("Dataset V2 còn NaN trong feature.")
    model.to_csv(OUT_MODEL, index=False, encoding="utf-8-sig")

    print(f"\nDữ liệu đầy đủ : {len(full):,} dòng")
    print(f"Dataset V2     : {len(model):,} dòng; {model['symbol'].nunique()} mã")
    print(f"Số feature     : {len(MODEL_FEATURES_V2)}")
    print(f"Giai đoạn      : {model['time'].min().date()} -> {model['time'].max().date()}")
    print("Phân phối target vượt VN-Index:")
    print(model[TARGET_COL].value_counts(normalize=True).sort_index().round(4))
    print("✓ Không có biến tương lai trong feature.")
    print(f"✓ {OUT_FULL}")
    print(f"✓ {OUT_MODEL}")


if __name__ == "__main__":
    main()
