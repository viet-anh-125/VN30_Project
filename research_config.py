"""Cấu hình nghiên cứu dùng chung cho pipeline VN30 nâng cấp."""

import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data_raw"
REFERENCE_DIR = PROJECT_ROOT / "data_reference"
CHART_DIR = PROJECT_ROOT / "output_charts"

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
UNIVERSE_REFERENCE_DATE = "2025-01-02"
TARGET_HORIZON = 20
RANDOM_STATE = 42

# Tập nghiên cứu cố định tại đầu tập kiểm định 2025 (universe chính của đề tài).
VN30_SYMBOLS_START_2025 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

# Kịch bản chi phí hai chiều dùng cho phân tích độ nhạy, không đại diện biểu phí VPS.
TRANSACTION_COST_SCENARIOS = [0.0, 0.002, 0.004, 0.006]

# ---------------------------------------------------------------------------
# Kiểm tra độ nhạy universe (Nhận xét GVHD mục 2 / Chương 4.5.1)
#
# fixed30 : universe chính của đề tài, cố định theo VN30 tại 02/01/2025.
# core23  : chỉ 23 mã có mặt ở CẢ hai mốc 03/02/2020 và 02/01/2025, dùng để
#           định lượng mức ảnh hưởng của look-ahead/selection bias khi cố
#           định danh sách theo thành phần tương lai.
#
# Chọn chế độ bằng biến môi trường VN30_UNIVERSE_MODE (không sửa code khi
# chạy chay_pipeline_v2.py hai lần):
#   PowerShell:  $env:VN30_UNIVERSE_MODE = "core23"
# ---------------------------------------------------------------------------
UNIVERSE_MODE = os.environ.get("VN30_UNIVERSE_MODE", "fixed30").strip().lower()
if UNIVERSE_MODE not in {"fixed30", "core23"}:
    raise ValueError(
        f"VN30_UNIVERSE_MODE='{UNIVERSE_MODE}' không hợp lệ. "
        "Chỉ nhận 'fixed30' hoặc 'core23'."
    )

# Hậu tố gắn vào tên file output để hai lần chạy không ghi đè lên nhau.
# fixed30 giữ nguyên tên file cũ (không hậu tố) để dashboard hiện tại không
# phải sửa gì; core23 có hậu tố "_core23".
UNIVERSE_SUFFIX = "" if UNIVERSE_MODE == "fixed30" else "_core23"

OVERLAP_FILE = DATA_DIR / "vn30_universe_overlap_2020_2025.csv"


def get_core23_symbols() -> list[str]:
    """Đọc 23 mã có mặt ở cả mốc 2020 và 2025 từ output của
    phan_tich_thanh_phan_vn30.py. Script đó PHẢI được chạy trước.
    """
    if not OVERLAP_FILE.exists():
        raise FileNotFoundError(
            f"Chưa có {OVERLAP_FILE}. Hãy chạy phan_tich_thanh_phan_vn30.py trước "
            "khi dùng UNIVERSE_MODE='core23'."
        )
    overlap = pd.read_csv(OVERLAP_FILE)
    shared = overlap[overlap["membership_group"] == "Có mặt ở cả hai mốc"]
    symbols = sorted(shared["symbol"].astype(str).str.upper().str.strip().tolist())
    if len(symbols) != 23:
        raise ValueError(
            f"Kỳ vọng 23 mã giao nhau nhưng đọc được {len(symbols)} mã: {symbols}. "
            "Kiểm tra lại data_reference/vn30_review_changes_2020_2025.csv."
        )
    return symbols


def get_universe_symbols() -> list[str]:
    """Danh sách mã dùng cho lần chạy hiện tại, theo UNIVERSE_MODE."""
    if UNIVERSE_MODE == "fixed30":
        return list(VN30_SYMBOLS_START_2025)
    return get_core23_symbols()


def ensure_project_directories() -> None:
    for path in (DATA_DIR, REFERENCE_DIR, CHART_DIR):
        path.mkdir(parents=True, exist_ok=True)
