"""
CODE 1 - THU THẬP DỮ LIỆU VN30 + VN-INDEX (BẢN SỬA)
=====================================================

Đề tài:
    Xây dựng công cụ định lượng hỗ trợ sàng lọc và đánh giá xu hướng
    nhóm cổ phiếu VN30 phục vụ hoạt động phân tích đầu tư tại
    Công ty Cổ phần Chứng khoán VPS.

Mục tiêu của Code 1:
    1. Thu thập OHLCV ngày cho một UNIVERSE VN30 CỐ ĐỊNH.
    2. Thu thập VN-Index để tính Beta và làm benchmark ở Code 4.
    3. Tạo dữ liệu 2020-2025 nhất quán cho toàn bộ pipeline.

QUY ƯỚC UNIVERSE:
    - Universe được chốt tại ĐẦU TẬP TEST 2025, ngày tham chiếu 02/01/2025.
    - Tại thời điểm này POW vẫn thuộc VN30 và LPB chưa thuộc VN30.
    - Kỳ review tháng 01/2025 của HOSE chỉ có thay đổi:
          + thêm LPB
          + loại POW
      và có hiệu lực từ 03/02/2025.
    - Vì vậy universe dùng ở đây là rổ VN30 đang có hiệu lực tại đầu 2025.
    - Đây vẫn là "fixed-universe design": dùng chính 30 mã này để lấy lịch sử
      2020-2025. Báo cáo cần nêu đây là giới hạn có survivorship bias đối với
      giai đoạn lịch sử 2020-2024, nhưng KHÔNG dùng thành phần VN30 tương lai
      sau năm 2025 để chọn universe cho Test 2025.

NGUỒN DỮ LIỆU:
    - Thư viện: vnstock bản mới.
    - Adapter: Quote.
    - Nguồn được chỉ định tường minh: VCI.
    - Không dùng stock_historical_data của Vnstock Legacy.

CÀI / NÂNG CẤP:
    python -m pip install --upgrade vnstock

Input:
    Không có file dữ liệu đầu vào.

Output:
    data_raw/vn30_ohlcv_2020_2025.csv
    data_raw/vnindex_ohlcv_2020_2025.csv
    data_raw/vn30_universe_start_2025.csv

Checkpoint mới:
    data_raw/raw_symbols_vn30_start_2025_vci/

LƯU Ý:
    - Dùng checkpoint folder MỚI để không vô tình đọc lại checkpoint cũ của
      universe 2026 / Vnstock Legacy.
    - Nếu muốn tải lại toàn bộ từ đầu, xóa riêng folder checkpoint mới trên.
"""

import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version

import pandas as pd


# =============================================================================
# 0. KIỂM TRA PHIÊN BẢN VNSTOCK
# =============================================================================

def check_vnstock_version() -> str:
    """
    Chặn Vnstock Legacy 0.2.x để tránh chạy nhầm API cũ.

    Không khóa cứng một minor version cụ thể vì Vnstock có thể được cập nhật,
    nhưng yêu cầu package hiện đại có lớp Quote.
    """
    try:
        installed_version = version("vnstock")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Chưa cài vnstock.\n"
            "Hãy chạy:\n"
            "  python -m pip install --upgrade vnstock"
        ) from exc

    if installed_version.startswith("0.2."):
        raise RuntimeError(
            f"Bạn đang dùng Vnstock Legacy {installed_version}.\n"
            "Code này không dùng API Legacy.\n"
            "Hãy nâng cấp trước:\n"
            "  python -m pip install --upgrade vnstock"
        )

    return installed_version


VNSTOCK_VERSION = check_vnstock_version()

# Vnstock hiện đại hỗ trợ import trực tiếp Quote.
# Giữ fallback để tương thích với một số bản adapter đã tái cấu trúc module.
try:
    from vnstock import Quote
except ImportError:
    try:
        from vnstock.api import Quote
    except ImportError as exc:
        raise RuntimeError(
            "Không import được lớp Quote từ vnstock.\n"
            f"Phiên bản đang cài: {VNSTOCK_VERSION}\n"
            "Hãy thử:\n"
            "  python -m pip install --upgrade vnstock"
        ) from exc


# =============================================================================
# 1. CẤU HÌNH NGHIÊN CỨU
# =============================================================================

# Universe VN30 tại đầu Test 2025 (02/01/2025).
# So với danh mục có hiệu lực 03/02/2025: dùng POW thay LPB.
VN30_SYMBOLS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

VN30_REFERENCE_DATE = "2025-01-02"
MARKET_INDEX_SYMBOL = "VNINDEX"

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
INTERVAL = "1D"
SOURCE = "VCI"

OUTPUT_DIR = "data_raw"

# Folder MỚI: không dùng lại raw_symbols/ cũ.
SYMBOL_DIR = os.path.join(
    OUTPUT_DIR,
    "raw_symbols_vn30_start_2025_vci"
)

OUTPUT_VN30 = os.path.join(
    OUTPUT_DIR,
    "vn30_ohlcv_2020_2025.csv"
)

OUTPUT_VNINDEX = os.path.join(
    OUTPUT_DIR,
    "vnindex_ohlcv_2020_2025.csv"
)

OUTPUT_UNIVERSE = os.path.join(
    OUTPUT_DIR,
    "vn30_universe_start_2025.csv"
)


# =============================================================================
# 2. CẤU HÌNH RATE LIMIT / RETRY
# =============================================================================

# Giữ tốc độ thận trọng để giảm rủi ro bị chặn ở gói miễn phí.
MAX_REQUESTS_PER_WINDOW = 12
WINDOW_SECONDS = 60
SLEEP_BETWEEN_REQUESTS = 2.0

MAX_RETRIES = 5
RATE_LIMIT_WAIT_SECONDS = 65
GENERIC_ERROR_WAIT_SECONDS = 8

request_times = []


# =============================================================================
# 3. HÀM TIỆN ÍCH
# =============================================================================

def wait_for_rate_limit() -> None:
    """Bộ đếm sliding-window để tránh gửi quá nhiều request trong 60 giây."""
    global request_times

    now = time.time()
    request_times = [
        t for t in request_times
        if now - t < WINDOW_SECONDS
    ]

    if len(request_times) >= MAX_REQUESTS_PER_WINDOW:
        oldest = min(request_times)
        wait_seconds = WINDOW_SECONDS - (now - oldest) + 2

        if wait_seconds > 0:
            print(
                f"\n⏳ Đã đạt {MAX_REQUESTS_PER_WINDOW} request/"
                f"{WINDOW_SECONDS}s. Chờ {wait_seconds:.0f}s...\n"
            )
            time.sleep(wait_seconds)

        now = time.time()
        request_times = [
            t for t in request_times
            if now - t < WINDOW_SECONDS
        ]

    request_times.append(time.time())
    time.sleep(SLEEP_BETWEEN_REQUESTS)


def normalize_history(
    df: pd.DataFrame,
    symbol: str
) -> pd.DataFrame:
    """
    Chuẩn hóa output của Quote.history về schema cố định:
        time, symbol, open, high, low, close, volume
    """
    if df is None or df.empty:
        raise ValueError(f"{symbol}: API trả về dữ liệu rỗng.")

    out = df.copy()

    # Chuẩn hóa tên cột.
    out.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in out.columns
    ]

    # Một số phiên bản có thể dùng date thay time.
    if "time" not in out.columns and "date" in out.columns:
        out = out.rename(columns={"date": "time"})

    # Một số output có ticker/symbol sẵn; ta gán lại symbol nghiên cứu để
    # đảm bảo thống nhất tuyệt đối.
    required_market_cols = {
        "time", "open", "high", "low", "close", "volume"
    }

    missing = required_market_cols - set(out.columns)

    if missing:
        raise ValueError(
            f"{symbol}: thiếu cột bắt buộc {sorted(missing)}.\n"
            f"Các cột nhận được: {out.columns.tolist()}"
        )

    out["time"] = pd.to_datetime(
        out["time"],
        errors="coerce"
    )

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce"
        )

    out["symbol"] = symbol

    out = out[
        ["time", "symbol", "open", "high", "low", "close", "volume"]
    ].copy()

    out = out.dropna(
        subset=["time", "close"]
    )

    # Chỉ giữ đúng khoảng nghiên cứu.
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE)

    out = out[
        (out["time"] >= start_ts)
        & (out["time"] <= end_ts)
    ].copy()

    out = (
        out
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError(
            f"{symbol}: không còn dữ liệu trong khoảng "
            f"{START_DATE} -> {END_DATE} sau chuẩn hóa."
        )

    if (out["close"] <= 0).any():
        raise ValueError(
            f"{symbol}: phát hiện close <= 0."
        )

    if out["volume"].notna().any():
        negative_volume = (
            out.loc[out["volume"].notna(), "volume"] < 0
        ).any()

        if negative_volume:
            raise ValueError(
                f"{symbol}: phát hiện volume < 0."
            )

    return out


def is_rate_limit_error(error: BaseException) -> bool:
    """Nhận diện tương đối các thông báo rate-limit thường gặp."""
    text = str(error).lower()

    keywords = [
        "rate limit",
        "ratelimit",
        "too many requests",
        "429",
        "giới hạn",
    ]

    return any(keyword in text for keyword in keywords)


# =============================================================================
# 4. LẤY DỮ LIỆU TỪ VCI
# =============================================================================

def fetch_history(
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE
) -> pd.DataFrame:
    """
    Lấy OHLCV bằng Quote(source='VCI', symbol=...).

    Cùng một cơ chế được dùng cho cổ phiếu và VNINDEX; Vnstock hiện đại
    nhận diện VNINDEX là chỉ số.
    """
    wait_for_rate_limit()

    quote = Quote(
        source=SOURCE,
        symbol=symbol
    )

    df = quote.history(
        start=start,
        end=end,
        interval=INTERVAL
    )

    return normalize_history(
        df=df,
        symbol=symbol
    )


def fetch_history_with_retry(
    symbol: str,
    start: str = START_DATE,
    end: str = END_DATE
) -> pd.DataFrame:
    """Retry với backoff; không bắt KeyboardInterrupt."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_history(
                symbol=symbol,
                start=start,
                end=end
            )

        except (Exception, SystemExit) as exc:
            last_error = exc

            if is_rate_limit_error(exc):
                wait_seconds = RATE_LIMIT_WAIT_SECONDS
                reason = "Rate limit"
            else:
                wait_seconds = GENERIC_ERROR_WAIT_SECONDS * attempt
                reason = type(exc).__name__

            print(
                f"   ✗ {symbol}: lỗi lần "
                f"{attempt}/{MAX_RETRIES} ({reason})"
            )
            print(f"     Chi tiết: {exc}")

            if attempt < MAX_RETRIES:
                print(
                    f"     Chờ {wait_seconds:.0f}s rồi thử lại..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Không lấy được dữ liệu cho {symbol} sau "
        f"{MAX_RETRIES} lần thử. Lỗi cuối: {last_error}"
    )


# =============================================================================
# 5. CHECKPOINT
# =============================================================================

def symbol_checkpoint_path(symbol: str) -> str:
    return os.path.join(
        SYMBOL_DIR,
        f"{symbol}.csv"
    )


def save_symbol_checkpoint(
    symbol: str,
    df: pd.DataFrame
) -> None:
    os.makedirs(
        SYMBOL_DIR,
        exist_ok=True
    )

    df.to_csv(
        symbol_checkpoint_path(symbol),
        index=False,
        encoding="utf-8-sig"
    )


def load_symbol_checkpoint(
    symbol: str
):
    """
    Đọc checkpoint và validate lại.
    Nếu checkpoint hỏng -> trả None để tải lại.
    """
    path = symbol_checkpoint_path(symbol)

    if not os.path.exists(path):
        return None

    try:
        cached = pd.read_csv(
            path,
            parse_dates=["time"]
        )

        cached = normalize_history(
            cached,
            symbol=symbol
        )

        return cached

    except Exception as exc:
        print(
            f"   ⚠ Checkpoint {symbol} không hợp lệ: {exc}"
        )
        print("     -> Sẽ tải lại mã này.")
        return None


# =============================================================================
# 6. THU THẬP 30 MÃ VN30
# =============================================================================

def fetch_vn30():
    os.makedirs(
        SYMBOL_DIR,
        exist_ok=True
    )

    all_frames = []
    failed = []

    for i, symbol in enumerate(
        VN30_SYMBOLS,
        start=1
    ):
        cached = load_symbol_checkpoint(
            symbol
        )

        if cached is not None:
            print(
                f"[{i:02d}/{len(VN30_SYMBOLS)}] "
                f"{symbol}: dùng checkpoint "
                f"({len(cached):,} dòng, "
                f"{cached['time'].min().date()} -> "
                f"{cached['time'].max().date()})"
            )

            all_frames.append(cached)
            continue

        print(
            f"[{i:02d}/{len(VN30_SYMBOLS)}] "
            f"Đang tải {symbol} từ {SOURCE}..."
        )

        try:
            df_symbol = fetch_history_with_retry(
                symbol
            )

            save_symbol_checkpoint(
                symbol,
                df_symbol
            )

            all_frames.append(
                df_symbol
            )

            print(
                f"   ✓ {symbol}: {len(df_symbol):,} dòng | "
                f"{df_symbol['time'].min().date()} -> "
                f"{df_symbol['time'].max().date()}"
            )

        except Exception as exc:
            print(
                f"   ✗ THẤT BẠI {symbol}: {exc}"
            )
            failed.append(symbol)

    if failed:
        raise RuntimeError(
            "Chưa lấy đủ 30 mã. Các mã thất bại: "
            f"{failed}\n"
            "Hãy chạy lại script; các mã đã có checkpoint "
            "sẽ được bỏ qua."
        )

    if len(all_frames) != len(VN30_SYMBOLS):
        raise RuntimeError(
            "Số DataFrame thu được không bằng 30."
        )

    df_vn30 = pd.concat(
        all_frames,
        ignore_index=True
    )

    df_vn30 = (
        df_vn30
        .sort_values(["symbol", "time"])
        .reset_index(drop=True)
    )

    return df_vn30


# =============================================================================
# 7. THU THẬP VN-INDEX
# =============================================================================

def fetch_vnindex() -> pd.DataFrame:
    print("\nĐang xử lý VNINDEX...")

    cached = load_symbol_checkpoint(
        MARKET_INDEX_SYMBOL
    )

    if cached is not None:
        print(
            f"   {MARKET_INDEX_SYMBOL}: dùng checkpoint "
            f"({len(cached):,} dòng, "
            f"{cached['time'].min().date()} -> "
            f"{cached['time'].max().date()})"
        )
        return cached

    print(
        f"   Đang tải {MARKET_INDEX_SYMBOL} "
        f"từ {SOURCE}..."
    )

    df_index = fetch_history_with_retry(
        MARKET_INDEX_SYMBOL
    )

    save_symbol_checkpoint(
        MARKET_INDEX_SYMBOL,
        df_index
    )

    print(
        f"   ✓ {MARKET_INDEX_SYMBOL}: "
        f"{len(df_index):,} dòng | "
        f"{df_index['time'].min().date()} -> "
        f"{df_index['time'].max().date()}"
    )

    return df_index


# =============================================================================
# 8. VALIDATION
# =============================================================================

def validate_vn30_dataset(
    df: pd.DataFrame
) -> None:
    print("\n" + "=" * 80)
    print("KIỂM TRA DATASET VN30")
    print("=" * 80)

    actual_symbols = set(
        df["symbol"].unique()
    )

    expected_symbols = set(
        VN30_SYMBOLS
    )

    missing_symbols = sorted(
        expected_symbols - actual_symbols
    )

    unexpected_symbols = sorted(
        actual_symbols - expected_symbols
    )

    print(
        f"1) Số mã: {df['symbol'].nunique()} "
        f"(kỳ vọng {len(VN30_SYMBOLS)})"
    )

    if missing_symbols:
        raise ValueError(
            f"Thiếu mã: {missing_symbols}"
        )

    if unexpected_symbols:
        raise ValueError(
            f"Có mã ngoài universe: {unexpected_symbols}"
        )

    duplicates = df.duplicated(
        subset=["symbol", "time"]
    ).sum()

    print(
        f"2) Duplicate symbol-time: {duplicates:,}"
    )

    if duplicates != 0:
        raise ValueError(
            "Dataset VN30 có duplicate symbol-time."
        )

    print(
        f"3) Giai đoạn toàn bộ: "
        f"{df['time'].min().date()} -> "
        f"{df['time'].max().date()}"
    )

    print(
        f"4) Tổng số dòng: {len(df):,}"
    )

    missing_close = int(
        df["close"].isna().sum()
    )

    print(
        f"5) Missing close: {missing_close:,}"
    )

    if missing_close > 0:
        raise ValueError(
            "Có missing close trong dataset."
        )

    # Bảng số dòng và thời gian từng mã.
    summary = (
        df.groupby("symbol")
        .agg(
            rows=("time", "size"),
            first_date=("time", "min"),
            last_date=("time", "max")
        )
        .sort_index()
    )

    print("\nChi tiết từng mã:")
    print(summary.to_string())

    # Cảnh báo nếu mã không có dữ liệu gần cuối 2025.
    last_dates = summary["last_date"]
    stale = summary[
        last_dates < pd.Timestamp("2025-12-01")
    ]

    if not stale.empty:
        print(
            "\n⚠ CẢNH BÁO: một số mã không có dữ liệu "
            "tới ít nhất 01/12/2025:"
        )
        print(
            stale[["rows", "last_date"]].to_string()
        )

    print(
        "\n✓ VALIDATION VN30 HOÀN TẤT."
    )


def validate_vnindex_dataset(
    df: pd.DataFrame
) -> None:
    print("\n" + "=" * 80)
    print("KIỂM TRA VN-INDEX")
    print("=" * 80)

    if df["symbol"].nunique() != 1:
        raise ValueError(
            "VNINDEX dataset có nhiều hơn 1 symbol."
        )

    if (
        df["symbol"].iloc[0]
        != MARKET_INDEX_SYMBOL
    ):
        raise ValueError(
            "Symbol trong file index không phải VNINDEX."
        )

    duplicates = df.duplicated(
        subset=["time"]
    ).sum()

    if duplicates != 0:
        raise ValueError(
            f"VNINDEX có {duplicates} duplicate date."
        )

    if df["close"].isna().any():
        raise ValueError(
            "VNINDEX có missing close."
        )

    print(
        f"Số dòng : {len(df):,}"
    )

    print(
        f"Thời gian: "
        f"{df['time'].min().date()} -> "
        f"{df['time'].max().date()}"
    )

    print(
        "✓ VALIDATION VNINDEX HOÀN TẤT."
    )


# =============================================================================
# 9. LƯU OUTPUT
# =============================================================================

def save_outputs(
    df_vn30: pd.DataFrame,
    df_index: pd.DataFrame
) -> None:
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df_vn30.to_csv(
        OUTPUT_VN30,
        index=False,
        encoding="utf-8-sig"
    )

    df_index.to_csv(
        OUTPUT_VNINDEX,
        index=False,
        encoding="utf-8-sig"
    )

    universe_df = pd.DataFrame({
        "symbol": VN30_SYMBOLS,
        "universe_reference_date": VN30_REFERENCE_DATE,
        "design": "fixed_universe_at_start_of_test_2025",
        "data_source": SOURCE,
    })

    universe_df.to_csv(
        OUTPUT_UNIVERSE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("ĐÃ LƯU OUTPUT")
    print("=" * 80)

    print(
        f"✓ {OUTPUT_VN30} "
        f"({len(df_vn30):,} dòng)"
    )

    print(
        f"✓ {OUTPUT_VNINDEX} "
        f"({len(df_index):,} dòng)"
    )

    print(
        f"✓ {OUTPUT_UNIVERSE} "
        f"({len(universe_df)} mã)"
    )


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("CODE 1 - THU THẬP DỮ LIỆU VN30 + VN-INDEX")
    print("=" * 80)

    print(
        f"Vnstock version : {VNSTOCK_VERSION}"
    )

    print(
        f"Data source     : {SOURCE}"
    )

    print(
        f"Universe date   : {VN30_REFERENCE_DATE}"
    )

    print(
        f"Giai đoạn       : {START_DATE} -> {END_DATE}"
    )

    print(
        f"Tần suất        : {INTERVAL}"
    )

    print(
        f"Số mã VN30      : {len(VN30_SYMBOLS)}"
    )

    print(
        f"Checkpoint      : {SYMBOL_DIR}"
    )

    print("\nDanh sách universe:")
    print(
        ", ".join(VN30_SYMBOLS)
    )

    if len(VN30_SYMBOLS) != 30:
        raise ValueError(
            "VN30_SYMBOLS phải có đúng 30 mã."
        )

    if len(set(VN30_SYMBOLS)) != 30:
        raise ValueError(
            "VN30_SYMBOLS có mã bị lặp."
        )

    print("\n" + "=" * 80)
    print("BƯỚC 1 - THU THẬP 30 MÃ")
    print("=" * 80)

    df_vn30 = fetch_vn30()

    print("\n" + "=" * 80)
    print("BƯỚC 2 - THU THẬP VNINDEX")
    print("=" * 80)

    df_index = fetch_vnindex()

    validate_vn30_dataset(
        df_vn30
    )

    validate_vnindex_dataset(
        df_index
    )

    save_outputs(
        df_vn30,
        df_index
    )

    print("\n" + "=" * 80)
    print("HOÀN TẤT CODE 1")
    print("=" * 80)

    print(
        f"VN30 rows   : {len(df_vn30):,}"
    )

    print(
        f"VN30 symbols: {df_vn30['symbol'].nunique()}/30"
    )

    print(
        f"VNINDEX rows: {len(df_index):,}"
    )

    print("\nBƯỚC TIẾP THEO:")
    print(
        "Gửi lại toàn bộ output terminal của Code 1. "
        "Chưa chạy Code 2 cũ."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nĐã dừng bởi người dùng. "
            "Checkpoint các mã tải xong vẫn được giữ lại."
        )
        sys.exit(130)
