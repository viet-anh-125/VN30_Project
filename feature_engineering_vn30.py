"""
Feature Engineering + Target cho bài toán phân loại xu hướng VN30
================================================================
Đề tài:
Xây dựng công cụ định lượng hỗ trợ sàng lọc và đánh giá xu hướng
nhóm cổ phiếu VN30 - Công ty CP Chứng khoán VPS

Input:
    data_raw/vn30_ohlcv_2020_2025.csv
    data_raw/vnindex_ohlcv_2020_2025.csv
    data_raw/vn30_universe_start_2025.csv

Output:
    data_raw/vn30_features_2020_2025.csv
        -> toàn bộ dữ liệu sau khi tính feature + target (GIỮ NGUYÊN NaN)

    data_raw/vn30_model_dataset_2020_2025.csv
        -> dataset sạch: time, symbol, close, 10 feature, target - đã loại NaN
        -> sử dụng trực tiếp cho Chương 3 (train_models_vn30.py); cột close
           được giữ lại vì Chương 4 cần giá tại thời điểm dự báo để mô phỏng
           đầu tư (tính Return/Sharpe/MDD)

Bộ 10 feature chính thức:
    1. return_1d
    2. return_5d
    3. return_20d
    4. close_to_ma_20
    5. close_to_ma_50
    6. ma_20_to_ma_50
    7. volatility_20d
    8. rsi_14
    9. volume_to_ma_20
    10. beta_60d

Target:
    Target = 1 nếu lợi suất 20 phiên tiếp theo > 0
    Target = 0 nếu lợi suất 20 phiên tiếp theo <= 0

Nguyên tắc:
    - Feature chỉ sử dụng dữ liệu tại thời điểm t và quá khứ.
    - Target sử dụng giá tương lai t+20 chỉ với vai trò nhãn.
    - Không đưa MA20, MA50 tuyệt đối trực tiếp vào mô hình.
    - Không đưa OHLCV trực tiếp vào mô hình.

QUAN TRỌNG VỀ KIỂM TRA CHẤT LƯỢNG (xem validate_feature_dataset):
    - Việc kiểm tra Missing Value PHẢI thực hiện trên vn30_features_2020_2025.csv
      (TRƯỚC khi dropna), không phải trên dataset ML đã lọc sạch. Kiểm tra
      "sau dropna" luôn luôn ra 0 Missing Value một cách hiển nhiên, không
      phát hiện được các bất thường (ví dụ beta_60d NaN toàn bộ một mã nào
      đó do lỗi merge ngày với VN-Index) - dropna() sẽ âm thầm xóa hết các
      dòng đó mà không ai biết nếu chỉ kiểm tra sau khi đã lọc.
"""

import os
import pandas as pd
import numpy as np


# ==========================================================================
# 1. CẤU HÌNH
# ==========================================================================

INPUT_STOCKS_CSV = "data_raw/vn30_ohlcv_2020_2025.csv"
INPUT_INDEX_CSV = "data_raw/vnindex_ohlcv_2020_2025.csv"
INPUT_UNIVERSE_CSV = "data_raw/vn30_universe_start_2025.csv"

OUTPUT_FEATURES_CSV = "data_raw/vn30_features_2020_2025.csv"
OUTPUT_MODEL_CSV = "data_raw/vn30_model_dataset_2020_2025.csv"


# --------------------------------------------------------------------------
# Tham số Feature
# --------------------------------------------------------------------------

RETURN_WINDOWS = [1, 5, 20]
MA_WINDOWS = [20, 50]
VOLATILITY_WINDOW = 20
RSI_WINDOW = 14
VOLUME_MA_WINDOW = 20
BETA_WINDOW = 60


# --------------------------------------------------------------------------
# Tham số Target
# --------------------------------------------------------------------------

TARGET_HORIZON = 20


# ==========================================================================
# 2. DANH SÁCH 10 FEATURE CHÍNH THỨC
# ==========================================================================

MODEL_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "close_to_ma_20",
    "close_to_ma_50",
    "ma_20_to_ma_50",
    "volatility_20d",
    "rsi_14",
    "volume_to_ma_20",
    "beta_60d",
]


# ==========================================================================
# 3. RETURN FEATURES
# ==========================================================================

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính lợi suất quá khứ trong 1, 5 và 20 phiên."""
    for window in RETURN_WINDOWS:
        df[f"return_{window}d"] = df["close"].pct_change(window, fill_method=None)
    return df


# ==========================================================================
# 4. MOVING AVERAGE FEATURES
# ==========================================================================

def add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính MA20, MA50, close_to_ma_20, close_to_ma_50, ma_20_to_ma_50.
    MA20/MA50 là biến trung gian, không dùng trực tiếp trong mô hình.
    """
    for window in MA_WINDOWS:
        df[f"ma_{window}"] = df["close"].rolling(window=window).mean()
        df[f"close_to_ma_{window}"] = df["close"] / df[f"ma_{window}"] - 1

    df["ma_20_to_ma_50"] = df["ma_20"] / df["ma_50"] - 1
    return df


# ==========================================================================
# 5. VOLATILITY
# ==========================================================================

def add_volatility_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility 20 phiên: độ lệch chuẩn của lợi suất ngày trong 20 phiên gần nhất."""
    daily_return = df["close"].pct_change(1, fill_method=None)
    df["volatility_20d"] = daily_return.rolling(window=VOLATILITY_WINDOW).std()
    return df


# ==========================================================================
# 6. RSI
# ==========================================================================

def add_rsi_feature(df: pd.DataFrame) -> pd.DataFrame:
    """RSI 14 phiên theo phương pháp Wilder."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / RSI_WINDOW, min_periods=RSI_WINDOW, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_WINDOW, min_periods=RSI_WINDOW, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


# ==========================================================================
# 7. VOLUME FEATURE
# ==========================================================================

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """So sánh khối lượng hiện tại với MA20 khối lượng."""
    volume_ma = df["volume"].rolling(window=VOLUME_MA_WINDOW).mean()
    df["volume_to_ma_20"] = df["volume"] / volume_ma - 1
    return df


# ==========================================================================
# 8. BETA SO VỚI VN-INDEX
# ==========================================================================

def add_beta_feature(df: pd.DataFrame, df_index: pd.DataFrame) -> pd.DataFrame:
    """
    Beta rolling 60 phiên: Beta = Cov(R_i, R_m) / Var(R_m).

    SỬA QUAN TRỌNG:
    - Ghép GIÁ ĐÓNG CỬA cổ phiếu và VN-Index theo cùng ngày trước.
    - Sau khi hai chuỗi giá đã căn ngày, mới tính pct_change().
    - Nhờ vậy stock_return và market_return luôn đại diện cho cùng cặp phiên.
    """
    stock = df[["time", "close"]].copy().rename(columns={"close": "stock_close"})
    market = (
        df_index[["time", "close"]]
        .copy()
        .rename(columns={"close": "market_close"})
        .sort_values("time")
        .drop_duplicates(subset=["time"])
    )

    merged = stock.merge(
        market,
        on="time",
        how="left",
        validate="one_to_one"
    )

    missing_market = int(merged["market_close"].isna().sum())
    if missing_market > 0:
        symbol = str(df["symbol"].iloc[0])
        raise ValueError(
            f"{symbol}: có {missing_market} phiên không ghép được VN-Index; "
            "không tính Beta trên hai lịch giao dịch lệch nhau."
        )

    merged["stock_return"] = merged["stock_close"].pct_change(1, fill_method=None)
    merged["market_return"] = merged["market_close"].pct_change(1, fill_method=None)

    covariance = merged["stock_return"].rolling(
        window=BETA_WINDOW,
        min_periods=BETA_WINDOW
    ).cov(merged["market_return"])

    market_variance = merged["market_return"].rolling(
        window=BETA_WINDOW,
        min_periods=BETA_WINDOW
    ).var()

    beta = covariance / market_variance.replace(0, np.nan)
    df["beta_60d"] = beta.to_numpy()
    return df


# ==========================================================================
# 9. TARGET
# ==========================================================================

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target = 1 nếu lợi suất 20 phiên tiếp theo > 0, ngược lại = 0.
    Future Return = Close(t+20) / Close(t) - 1.

    future_return_20d được GIỮ trong file feature đầy đủ để audit và phục vụ
    đánh giá Chương 4, nhưng KHÔNG được đưa vào dataset Machine Learning.
    """
    df["future_return_20d"] = (
        df["close"].shift(-TARGET_HORIZON) / df["close"] - 1
    )
    df["target"] = np.where(
        df["future_return_20d"].isna(),
        np.nan,
        (df["future_return_20d"] > 0).astype(int)
    )
    return df


def add_future_market_comparison(
    df: pd.DataFrame,
    df_index: pd.DataFrame,
) -> pd.DataFrame:
    """Thêm lợi suất tương lai VN-Index và excess return để đánh giá.

    Hai biến tương lai này chỉ tồn tại trong file feature đầy đủ. Chúng không
    được đưa vào MODEL_FEATURES hoặc dataset huấn luyện.
    """
    market = (
        df_index[["time", "close"]]
        .copy()
        .rename(columns={"close": "market_close"})
        .sort_values("time")
        .drop_duplicates(subset=["time"])
    )
    aligned = df[["time"]].merge(
        market,
        on="time",
        how="left",
        validate="one_to_one",
    )
    if aligned["market_close"].isna().any():
        symbol = str(df["symbol"].iloc[0])
        missing = int(aligned["market_close"].isna().sum())
        raise ValueError(
            f"{symbol}: thiếu {missing} giá VN-Index khi tính future market return."
        )

    df["future_market_return_20d"] = (
        aligned["market_close"].shift(-TARGET_HORIZON)
        / aligned["market_close"]
        - 1
    ).to_numpy()
    df["future_excess_return_20d"] = (
        df["future_return_20d"] - df["future_market_return_20d"]
    )
    return df


# ==========================================================================
# 10. FEATURE ENGINEERING CHO MỘT MÃ
# ==========================================================================

def build_features_for_symbol(df_symbol: pd.DataFrame, df_index: pd.DataFrame) -> pd.DataFrame:
    """Tính toàn bộ feature + target cho một mã cổ phiếu."""
    df = df_symbol.sort_values("time").reset_index(drop=True).copy()

    df = add_return_features(df)
    df = add_ma_features(df)
    df = add_volatility_feature(df)
    df = add_rsi_feature(df)
    df = add_volume_features(df)
    df = add_beta_feature(df, df_index)
    df = add_target(df)
    df = add_future_market_comparison(df, df_index)

    return df


# ==========================================================================
# 11. KIỂM TRA INPUT TỪ CODE 1
# ==========================================================================

def validate_code1_inputs(df_stocks: pd.DataFrame, df_index: pd.DataFrame) -> None:
    """Xác nhận Code 2 đang đọc đúng output đã khóa từ Code 1 mới."""
    if not os.path.exists(INPUT_UNIVERSE_CSV):
        raise FileNotFoundError(f"Không tìm thấy: {INPUT_UNIVERSE_CSV}")

    universe = pd.read_csv(INPUT_UNIVERSE_CSV)
    if "symbol" not in universe.columns:
        raise ValueError("File universe không có cột symbol.")

    expected_symbols = set(
        universe["symbol"].astype(str).str.upper().str.strip()
    )
    actual_symbols = set(
        df_stocks["symbol"].astype(str).str.upper().str.strip()
    )

    if len(expected_symbols) != 30:
        raise ValueError(
            f"Universe có {len(expected_symbols)} mã, kỳ vọng đúng 30."
        )
    if actual_symbols != expected_symbols:
        raise ValueError(
            "Danh sách mã OHLCV không khớp vn30_universe_start_2025.csv.\n"
            f"Thiếu: {sorted(expected_symbols - actual_symbols)}\n"
            f"Thừa: {sorted(actual_symbols - expected_symbols)}"
        )

    duplicate_stocks = df_stocks.duplicated(subset=["symbol", "time"]).sum()
    duplicate_index = df_index.duplicated(subset=["time"]).sum()
    if duplicate_stocks:
        raise ValueError(f"VN30 có {duplicate_stocks} duplicate symbol-time.")
    if duplicate_index:
        raise ValueError(f"VNINDEX có {duplicate_index} duplicate time.")
    if df_stocks["close"].isna().any():
        raise ValueError("VN30 còn missing close.")
    if df_index["close"].isna().any():
        raise ValueError("VNINDEX còn missing close.")

    print("✓ INPUT CODE 1 CHECK PASS")
    print(f"- Universe: {len(expected_symbols)} mã")
    print(f"- VN30 duplicate: {duplicate_stocks}")
    print(f"- VNINDEX duplicate: {duplicate_index}")


# ==========================================================================
# 12. FEATURE ENGINEERING TOÀN BỘ VN30
# ==========================================================================

def build_feature_dataset(df_stocks: pd.DataFrame, df_index: pd.DataFrame) -> pd.DataFrame:
    """Tính feature cho toàn bộ các mã VN30."""
    df_stocks = df_stocks.sort_values(["symbol", "time"]).reset_index(drop=True)
    df_index = df_index.sort_values("time").reset_index(drop=True)

    all_frames = []
    for symbol, df_symbol in df_stocks.groupby("symbol", sort=True):
        print(f"  -> Đang xử lý {symbol}...")
        df_feat = build_features_for_symbol(df_symbol, df_index)
        all_frames.append(df_feat)

    if not all_frames:
        raise RuntimeError("Không tạo được feature cho bất kỳ mã nào.")

    return pd.concat(all_frames, ignore_index=True)


# ==========================================================================
# 13. KIỂM TRA CHẤT LƯỢNG DATASET FEATURE (TRƯỚC KHI LỌC NaN)
# ==========================================================================

def validate_feature_dataset(df_features: pd.DataFrame) -> None:
    """
    Kiểm tra 5 điều kiện TRÊN DATASET FEATURE ĐẦY ĐỦ (chưa dropna):
        1) Đủ 30 mã?
        2) Đủ giai đoạn 2020-2025?
        3) Số dòng?
        4) 10 feature có NaN bất thường không? (đây là lý do PHẢI kiểm tra
           trước khi dropna - kiểm tra sau dropna sẽ luôn ra 0, vô nghĩa)
        5) Target có đủ 0/1, tỷ lệ hai lớp có quá lệch không?
    """
    print("\n" + "=" * 70)
    print("KIỂM TRA CHẤT LƯỢNG DATASET FEATURE (trước khi lọc NaN)")
    print("=" * 70)

    n_symbols = df_features["symbol"].nunique()
    print(f"1) Số mã: {n_symbols} (kỳ vọng 30)")
    if n_symbols != 30:
        print("   -> CẢNH BÁO: thiếu/thừa mã so với danh sách VN30 cố định.")

    date_min, date_max = df_features["time"].min(), df_features["time"].max()
    print(f"2) Giai đoạn: {date_min.date()} -> {date_max.date()} (kỳ vọng 2020-01 -> 2025-12)")

    print(f"3) Tổng số dòng: {len(df_features):,}")

    print("4) Số lượng NaN theo từng feature trong MODEL_FEATURES (TRƯỚC dropna):")
    nan_counts = df_features[MODEL_FEATURES].isna().sum()
    print(nan_counts)

    # Cảnh báo nếu NaN của 1 mã nào đó chiếm gần trọn chuỗi thời gian của mã
    # đó (dấu hiệu lỗi ghép ngày/tính feature cho riêng mã đó, chứ không
    # phải NaN "tự nhiên" ở đầu chuỗi do chưa đủ phiên rolling).
    rows_per_symbol = df_features.groupby("symbol").size()
    for feature in MODEL_FEATURES:
        nan_by_symbol = df_features[df_features[feature].isna()].groupby("symbol").size()
        suspicious = nan_by_symbol[nan_by_symbol > 0.9 * rows_per_symbol.reindex(nan_by_symbol.index)]
        if not suspicious.empty:
            print(f"   -> CẢNH BÁO: '{feature}' NaN gần như toàn bộ ở các mã: {list(suspicious.index)}")

    print("\n5) Phân phối Target (bỏ qua NA):")
    print(df_features["target"].value_counts(normalize=True))

    future_audit_columns = [
        "future_return_20d",
        "future_market_return_20d",
        "future_excess_return_20d",
    ]
    print("\n6) Biến tương lai chỉ dùng cho audit/đánh giá:")
    print(df_features[future_audit_columns].isna().sum())

    forbidden_in_model = set(future_audit_columns) & set(MODEL_FEATURES)
    if forbidden_in_model:
        raise ValueError(
            "Biến tương lai bị đưa nhầm vào MODEL_FEATURES: "
            f"{sorted(forbidden_in_model)}"
        )

    print("=" * 70)
    print(
        "Lưu ý: NaN ở đầu mỗi mã (chưa đủ phiên tính MA/Volatility/Beta) và "
        f"{TARGET_HORIZON} dòng cuối mỗi mã (chưa có target) là BÌNH THƯỜNG "
        "và sẽ được loại ở bước tạo dataset ML (create_model_dataset). Chỉ "
        "đáng lo nếu NaN xuất hiện RẢI RÁC giữa chuỗi hoặc CẢNH BÁO ở trên "
        "xuất hiện, hoặc nếu Target quá lệch (ví dụ > 80/20)."
    )


# ==========================================================================
# 14. TẠO DATASET DÙNG CHO MACHINE LEARNING (LỌC NaN Ở ĐÂY - MỘT LẦN DUY NHẤT)
# ==========================================================================

def create_model_dataset(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo dataset cuối cùng cho Machine Learning: giữ time, symbol, close
    (cần cho mô phỏng đầu tư ở Chương 4), 10 feature, target; loại các
    dòng thiếu feature hoặc target.
    """
    required_columns = ["time", "symbol", "close"] + MODEL_FEATURES + ["target"]

    missing_columns = [col for col in required_columns if col not in df_features.columns]
    if missing_columns:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing_columns}")

    df_model = df_features[required_columns].copy()

    before = len(df_model)
    df_model = df_model.dropna(subset=MODEL_FEATURES + ["target"]).reset_index(drop=True)
    after = len(df_model)

    print(f"\nĐã loại {before - after:,} dòng do thiếu feature/target.")

    df_model["target"] = df_model["target"].astype(int)
    df_model = df_model.sort_values(["time", "symbol"]).reset_index(drop=True)

    return df_model


# ==========================================================================
# 15. KIỂM TRA DATASET ML SAU CÙNG (bổ sung - không thay cho validate_feature_dataset)
# ==========================================================================

def validate_model_dataset(df_model: pd.DataFrame) -> None:
    """
    Kiểm tra nhanh dataset ML SAU khi đã dropna. KHÔNG dùng để kiểm tra
    Missing Value (vì chắc chắn = 0, không có giá trị cảnh báo) - chỉ dùng
    để kiểm tra kích thước, tỷ lệ Target còn lại sau lọc, và trùng lặp.
    Việc kiểm tra Missing Value THỰC SỰ nằm ở validate_feature_dataset().
    """
    print("\n" + "=" * 70)
    print("KIỂM TRA DATASET ML (sau khi đã lọc NaN)")
    print("=" * 70)

    print(f"Số dòng   : {len(df_model):,}")
    print(f"Số mã     : {df_model['symbol'].nunique()}")
    print(f"Thời gian : {df_model['time'].min().date()} -> {df_model['time'].max().date()}")

    print(f"\nSố feature ML: {len(MODEL_FEATURES)}")
    for feature in MODEL_FEATURES:
        print(f"  - {feature}")

    print("\nPhân bố Target (sau khi lọc NaN):")
    target_counts = df_model["target"].value_counts().sort_index()
    target_ratio = df_model["target"].value_counts(normalize=True).sort_index() * 100
    for value in target_counts.index:
        print(f"  Target = {value}: {target_counts[value]:,} ({target_ratio[value]:.2f}%)")

    duplicates = df_model.duplicated(subset=["time", "symbol"]).sum()
    if duplicates == 0:
        print("\n✓ Không có duplicate time-symbol.")
    else:
        raise ValueError(f"Có {duplicates:,} duplicate time-symbol trong dataset ML.")

    if df_model["symbol"].nunique() != 30:
        raise ValueError(
            f"Dataset ML chỉ còn {df_model['symbol'].nunique()} mã, kỳ vọng 30."
        )

    missing_total = int(df_model[MODEL_FEATURES + ["target"]].isna().sum().sum())
    if missing_total != 0:
        raise ValueError(f"Dataset ML còn {missing_total} giá trị NaN.")

    finite = np.isfinite(df_model[MODEL_FEATURES].to_numpy(dtype=float)).all()
    if not finite:
        raise ValueError("Dataset ML còn inf hoặc -inf.")

    print("✓ Dataset ML giữ đủ 30 mã, không NaN, không inf/-inf.")


# ==========================================================================
# 16. MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CODE 2 - FEATURE ENGINEERING + TARGET - VN30")
    print("=" * 70)

    if not os.path.exists(INPUT_STOCKS_CSV):
        raise FileNotFoundError(f"Không tìm thấy: {INPUT_STOCKS_CSV}")
    if not os.path.exists(INPUT_INDEX_CSV):
        raise FileNotFoundError(f"Không tìm thấy: {INPUT_INDEX_CSV}")
    if not os.path.exists(INPUT_UNIVERSE_CSV):
        raise FileNotFoundError(f"Không tìm thấy: {INPUT_UNIVERSE_CSV}")

    print("\n[1] Đọc dữ liệu đầu vào...")
    df_stocks = pd.read_csv(INPUT_STOCKS_CSV, parse_dates=["time"])
    df_index = pd.read_csv(INPUT_INDEX_CSV, parse_dates=["time"])

    df_stocks["symbol"] = df_stocks["symbol"].astype(str).str.upper().str.strip()
    df_index["symbol"] = df_index["symbol"].astype(str).str.upper().str.strip()
    df_stocks = df_stocks.sort_values(["symbol", "time"]).reset_index(drop=True)
    df_index = df_index.sort_values("time").reset_index(drop=True)

    validate_code1_inputs(df_stocks, df_index)
    print(f"- VN30 : {len(df_stocks):,} dòng")
    print(f"- Số mã: {df_stocks['symbol'].nunique()}")
    print(f"- VN-Index: {len(df_index):,} dòng")

    print("\n[2] Đang tính feature + target...")
    df_features = build_feature_dataset(df_stocks, df_index)
    print(f"\nSố dòng sau Feature Engineering: {len(df_features):,}")

    # QUAN TRỌNG: validate TRƯỚC khi ghi output chính thức.
    validate_feature_dataset(df_features)

    os.makedirs(os.path.dirname(OUTPUT_FEATURES_CSV), exist_ok=True)
    df_features.to_csv(OUTPUT_FEATURES_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu:\n  {OUTPUT_FEATURES_CSV}")

    print("\n[3] Tạo dataset Machine Learning...")
    df_model = create_model_dataset(df_features)

    validate_model_dataset(df_model)

    df_model.to_csv(OUTPUT_MODEL_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ Đã lưu dataset ML:\n  {OUTPUT_MODEL_CSV}")

    print("\n" + "=" * 70)
    print("HOÀN TẤT FEATURE ENGINEERING")
    print("=" * 70)
    print("\n10 feature chính thức:")
    for i, feature in enumerate(MODEL_FEATURES, start=1):
        print(f"{i:2d}. {feature}")
    print(f"\nTarget: xu hướng {TARGET_HORIZON} phiên tiếp theo")
    print(f"\nDataset DUY NHẤT Code 3 được dùng để train:\n{OUTPUT_MODEL_CSV}")
