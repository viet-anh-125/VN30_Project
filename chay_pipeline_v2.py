"""Chạy pipeline V2.

Mặc định chỉ chạy universe chính (30 mã cố định theo VN30 đầu 2025), đúng
như trước đây.

Thêm cờ --with-core23 để chạy NỐI TIẾP lần thứ hai trên 23 mã có mặt ở cả
hai mốc 2020 và 2025 (Nhận xét GVHD mục 2 / mục 3 phần 2), rồi tự động
chạy so_sanh_universe.py để tạo bảng so sánh cho mục 4.5.1 của báo cáo.

    python chay_pipeline_v2.py --with-core23
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

# Bước 1 chạy trước và dùng chung cho cả hai universe: nó tạo ra
# vn30_universe_overlap_2020_2025.csv, là nguồn xác định 23 mã giao nhau.
STEP_UNIVERSE_ANALYSIS = "phan_tich_thanh_phan_vn30.py"
STEPS = [
    "feature_engineering_relative_vn30.py",
    "train_models_relative_vn30.py",
    "danh_gia_sang_loc_relative_vn30.py",
]


def run_step(step: str, env: dict) -> None:
    print("\n" + "=" * 76 + f"\nCHẠY {step}  (VN30_UNIVERSE_MODE={env.get('VN30_UNIVERSE_MODE', 'fixed30')})\n" + "=" * 76)
    subprocess.run([sys.executable, str(BASE / step)], check=True, cwd=BASE, env=env)


def run_universe(mode: str) -> None:
    env = os.environ.copy()
    env["VN30_UNIVERSE_MODE"] = mode
    for step in STEPS:
        run_step(step, env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-core23", action="store_true",
        help="Chạy thêm lần thứ hai trên 23 mã giao nhau để kiểm tra độ nhạy bias.",
    )
    args = parser.parse_args()

    # Bước dùng chung, luôn chạy đúng một lần với biến môi trường mặc định.
    env = os.environ.copy()
    env.pop("VN30_UNIVERSE_MODE", None)
    run_step(STEP_UNIVERSE_ANALYSIS, env)

    print("\n" + "#" * 76)
    print("# UNIVERSE CHÍNH: fixed30 (30 mã cố định theo VN30 đầu 2025)")
    print("#" * 76)
    run_universe("fixed30")

    if args.with_core23:
        print("\n" + "#" * 76)
        print("# KIỂM TRA ĐỘ NHẠY: core23 (23 mã có mặt ở cả hai mốc 2020 và 2025)")
        print("#" * 76)
        run_universe("core23")

        print("\n" + "=" * 76 + "\nCHẠY so_sanh_universe.py\n" + "=" * 76)
        subprocess.run([sys.executable, str(BASE / "so_sanh_universe.py")], check=True, cwd=BASE)

    print("\nHOÀN TẤT PIPELINE V2. Chạy Dashboard bằng:")
    print(f'"{sys.executable}" -m streamlit run dashboard_vn30_v2.py')


if __name__ == "__main__":
    main()
