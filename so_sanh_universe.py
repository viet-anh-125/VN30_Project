"""So sánh universe fixed30 (30 mã cố định) với core23 (23 mã giao nhau).

Chạy SAU KHI đã có đủ 2 lần chạy (chay_pipeline_v2.py --with-core23, hoặc
chạy tay từng bước với VN30_UNIVERSE_MODE=fixed30 rồi VN30_UNIVERSE_MODE=
core23). Script chỉ đọc lại các file *_2025 / *_core23 đã có, không huấn
luyện lại.

Xuất: data_raw/bang_4_5_1_so_sanh_universe.csv — dán trực tiếp vào Chương 4
(mục 4.5.1) hoặc mục 2.2/3.1.1 của báo cáo, kèm ghi chú tỷ lệ rổ để không so
sánh Top-N khác tỷ lệ với nhau như trong bản đầu tiên.
"""
import numpy as np
import pandas as pd

from research_config import DATA_DIR

OUT_TABLE = DATA_DIR / "bang_4_5_1_so_sanh_universe.csv"


def read_pair(stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = pd.read_csv(DATA_DIR / f"{stem}.csv")
    core = pd.read_csv(DATA_DIR / f"{stem}_core23.csv")
    return fixed, core


def main() -> None:
    test_fixed, test_core = read_pair("relative_test_metrics_2025")
    topn_fixed, topn_core = read_pair("relative_topn_summary_2021_2025")
    bt_fixed, bt_core = read_pair("relative_backtest_summary_2021_2025")

    rows = []

    tf, tc = test_fixed.iloc[0], test_core.iloc[0]
    rows.append({
        "chi_tieu": "Tỷ lệ vượt VN-Index chung (2025)",
        "fixed30_30_ma": f"{tf['market_prevalence']:.2%}",
        "core23_23_ma": f"{tc['market_prevalence']:.2%}",
        "ghi_chu": "Không phụ thuộc N; so sánh trực tiếp được.",
    })
    rows.append({
        "chi_tieu": "AUC-ROC trung bình theo ngày (2025)",
        "fixed30_30_ma": f"{tf['auc_daily_mean']:.4f}",
        "core23_23_ma": f"{tc['auc_daily_mean']:.4f}",
        "ghi_chu": "Không phụ thuộc N; so sánh trực tiếp được.",
    })
    rows.append({
        "chi_tieu": "Precision@3 (2025)",
        "fixed30_30_ma": f"{tf['precision_at_3']:.2%}  (10,0% rổ)",
        "core23_23_ma": f"{tc['precision_at_3']:.2%}  (13,0% rổ)",
        "ghi_chu": "CÙNG N=3 nhưng KHÁC tỷ lệ rổ — chỉ tham khảo, không kết luận trực tiếp.",
    })
    rows.append({
        "chi_tieu": "Lift@3 (2025)",
        "fixed30_30_ma": f"{tf['aggregate_lift_at_3']:.3f}",
        "core23_23_ma": f"{tc['aggregate_lift_at_3']:.3f}",
        "ghi_chu": "Xem cùng dòng ở trên.",
    })

    def topn_row(stage: str, top_n_fixed: int, top_n_core: int, label: str, note: str):
        f_row = topn_fixed[(topn_fixed.evaluation_stage == stage) & (topn_fixed.top_n == top_n_fixed)]
        c_row = topn_core[(topn_core.evaluation_stage == stage) & (topn_core.top_n == top_n_core)]
        if f_row.empty or c_row.empty:
            return
        f_row, c_row = f_row.iloc[0], c_row.iloc[0]
        rows.append({
            "chi_tieu": f"Precision@N — {label} ({stage})",
            "fixed30_30_ma": f"{f_row['mean_precision_at_n']:.2%}  (N={top_n_fixed}, {top_n_fixed/30:.1%} rổ)",
            "core23_23_ma": f"{c_row['mean_precision_at_n']:.2%}  (N={top_n_core}, {top_n_core/23:.1%} rổ)",
            "ghi_chu": note,
        })
        rows.append({
            "chi_tieu": f"Lift@N — {label} ({stage})",
            "fixed30_30_ma": f"{f_row['aggregate_lift_at_n']:.3f}",
            "core23_23_ma": f"{c_row['aggregate_lift_at_n']:.3f}",
            "ghi_chu": note,
        })

    # So sánh đúng tỷ lệ rổ: Top-3/30 (10,0%) với Top-2/23 (8,7%);
    # Top-10/30 (33,3%) với Top-8/23 (34,8%).
    topn_row("final_test", 3, 2, "tỷ lệ rổ tương đương ~10%",
             "Đã quy đổi N để tỷ lệ rổ gần bằng nhau — so sánh được.")
    topn_row("final_test", 10, 8, "tỷ lệ rổ tương đương ~33-35%",
             "Đã quy đổi N để tỷ lệ rổ gần bằng nhau — so sánh được.")
    topn_row("combined_oos_descriptive", 3, 2, "tỷ lệ rổ tương đương ~10%, gộp 2021-2025",
             "Đã quy đổi N để tỷ lệ rổ gần bằng nhau — so sánh được. "
             "Giai đoạn 2021-2024 đã tham gia chọn mô hình, không phải kiểm định độc lập.")

    def bt_row(stage: str, top_n_fixed: int, top_n_core: int, label: str, cost: float = 0.004):
        f_row = bt_fixed[(bt_fixed.evaluation_stage == stage) & (bt_fixed.top_n == top_n_fixed)
                          & np.isclose(bt_fixed.round_trip_cost_rate, cost)]
        c_row = bt_core[(bt_core.evaluation_stage == stage) & (bt_core.top_n == top_n_core)
                         & np.isclose(bt_core.round_trip_cost_rate, cost)]
        if f_row.empty or c_row.empty:
            return
        f_row, c_row = f_row.iloc[0], c_row.iloc[0]
        rows.append({
            "chi_tieu": f"Lợi suất sau phí 0,4% — {label} ({stage})",
            "fixed30_30_ma": f"{f_row['net_final_return']:.2%}",
            "core23_23_ma": f"{c_row['net_final_return']:.2%}",
            "ghi_chu": "Đã quy đổi N để tỷ lệ rổ gần bằng nhau — so sánh được.",
        })
        rows.append({
            "chi_tieu": f"MDD — {label} ({stage})",
            "fixed30_30_ma": f"{f_row['max_drawdown']:.2%}",
            "core23_23_ma": f"{c_row['max_drawdown']:.2%}",
            "ghi_chu": "",
        })
        rows.append({
            "chi_tieu": f"Sharpe (quy đổi năm) — {label} ({stage})",
            "fixed30_30_ma": f"{f_row['sharpe_annualized']:.3f}",
            "core23_23_ma": f"{c_row['sharpe_annualized']:.3f}",
            "ghi_chu": "",
        })

    bt_row("final_test", 3, 2, "tỷ lệ rổ tương đương ~10%")

    table = pd.DataFrame(rows)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    print("=" * 90)
    print("BẢNG 4.5.1 — SO SÁNH ĐỘ NHẠY: FIXED30 (30 mã) vs CORE23 (23 mã giao nhau)")
    print("=" * 90)
    print(table.to_string(index=False))
    print(f"\n✓ Đã lưu: {OUT_TABLE}")
    print(
        "\nLưu ý khi viết mục 4.5.1: đọc trước tiên các dòng đã ghi 'so sánh được' "
        "(N đã quy đổi theo tỷ lệ rổ). Các dòng Precision@3/Lift@3 thô (N=3 cả hai "
        "bên) chỉ nên nêu như tham khảo, có ghi chú KHÁC tỷ lệ rổ."
    )


if __name__ == "__main__":
    main()
