from pathlib import Path

from research_config import UNIVERSE_MODE, UNIVERSE_SUFFIX, get_universe_symbols

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_raw"
TARGET_HORIZON = 20
TARGET_COL = "target_outperform_20d"

BASE_FEATURES = [
    "return_1d", "return_5d", "return_20d", "close_to_ma_20",
    "close_to_ma_50", "ma_20_to_ma_50", "volatility_20d", "rsi_14",
    "volume_to_ma_20", "beta_60d",
]

RELATIVE_FEATURES = [
    "excess_return_5d", "excess_return_20d", "market_return_20d",
    "market_volatility_20d", "market_close_to_ma_50",
    "rank_return_20d", "rank_rsi_14", "rank_volatility_20d",
    "rank_close_to_ma_20", "rank_volume_to_ma_20",
]

MODEL_FEATURES_V2 = BASE_FEATURES + RELATIVE_FEATURES

# Universe dùng cho lần chạy hiện tại (30 mã cố định, hoặc 23 mã giao nhau
# khi VN30_UNIVERSE_MODE=core23). Xem research_config.py.
TICKERS = get_universe_symbols()
SUFFIX = UNIVERSE_SUFFIX


def out_name(stem: str) -> str:
    """Tên file output có gắn hậu tố universe, ví dụ:
    out_name('relative_test_metrics_2025') ->
        'relative_test_metrics_2025.csv'          (fixed30)
        'relative_test_metrics_2025_core23.csv'   (core23)
    """
    return f"{stem}{SUFFIX}.csv"
