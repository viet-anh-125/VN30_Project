"""Lượng hóa thay đổi thành phần VN30 ở 12 kỳ review bán niên 2020-2025.

Script này chỉ phục vụ mô tả và lượng hóa giới hạn của universe cố định.
Nó không lọc lại dữ liệu mô hình và không tạo rổ động theo ngày.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from research_config import (
    CHART_DIR,
    DATA_DIR,
    REFERENCE_DIR,
    VN30_SYMBOLS_START_2025,
    ensure_project_directories,
)


INPUT_FILE = REFERENCE_DIR / "vn30_review_changes_2020_2025.csv"
OUTPUT_REVIEW_SUMMARY = DATA_DIR / "vn30_review_summary_2020_2025.csv"
OUTPUT_OVERLAP = DATA_DIR / "vn30_universe_overlap_2020_2025.csv"
OUTPUT_CHART = CHART_DIR / "vn30_review_changes_2020_2025.png"


def split_symbols(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item.strip().upper() for item in str(value).split("|") if item.strip()}


def load_and_validate_reviews() -> pd.DataFrame:
    reviews = pd.read_csv(INPUT_FILE, dtype=str).fillna("")
    required = {
        "review_period", "effective_date", "added", "removed",
        "change_count", "source_url", "source_note",
    }
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Thiếu cột trong file review: {sorted(missing)}")
    if len(reviews) != 12:
        raise ValueError(f"Cần đúng 12 kỳ review, hiện có {len(reviews)}.")
    if reviews["review_period"].duplicated().any():
        raise ValueError("Có kỳ review bị trùng.")

    reviews["effective_date"] = pd.to_datetime(reviews["effective_date"])
    reviews["added_set"] = reviews["added"].map(split_symbols)
    reviews["removed_set"] = reviews["removed"].map(split_symbols)
    reviews["observed_change_count"] = reviews["added_set"].map(len)

    unequal = reviews[
        reviews["added_set"].map(len) != reviews["removed_set"].map(len)
    ]
    if not unequal.empty:
        raise ValueError(
            "Số mã thêm và loại không cân bằng ở các kỳ: "
            f"{unequal['review_period'].tolist()}"
        )
    return reviews.sort_values("effective_date").reset_index(drop=True)


def reconstruct_early_2020_universe(reviews: pd.DataFrame) -> set[str]:
    """Đảo ngược các review sau 03/02/2020 từ universe đầu 2025."""
    universe = set(VN30_SYMBOLS_START_2025)
    relevant = reviews[
        (reviews["effective_date"] > pd.Timestamp("2020-02-03"))
        & (reviews["effective_date"] <= pd.Timestamp("2025-01-02"))
    ].sort_values("effective_date", ascending=False)

    for row in relevant.itertuples(index=False):
        universe.difference_update(row.added_set)
        universe.update(row.removed_set)

    if len(universe) != 30:
        raise ValueError(
            f"Universe tái dựng tại 03/02/2020 có {len(universe)} mã, kỳ vọng 30."
        )
    return universe


def build_outputs(reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_universe = reconstruct_early_2020_universe(reviews)
    end_universe = set(VN30_SYMBOLS_START_2025)
    shared = start_universe & end_universe

    summary_rows = []
    for row in reviews.itertuples(index=False):
        summary_rows.append(
            {
                "review_period": row.review_period,
                "effective_date": row.effective_date.date(),
                "added": "|".join(sorted(row.added_set)),
                "removed": "|".join(sorted(row.removed_set)),
                "number_added": len(row.added_set),
                "number_removed": len(row.removed_set),
                "main_basket_changed": int(bool(row.added_set or row.removed_set)),
                "source_url": row.source_url,
            }
        )
    summary = pd.DataFrame(summary_rows)

    symbols = sorted(start_universe | end_universe)
    overlap = pd.DataFrame(
        {
            "symbol": symbols,
            "in_vn30_effective_2020_02_03": [s in start_universe for s in symbols],
            "in_fixed_universe_2025_01_02": [s in end_universe for s in symbols],
        }
    )
    overlap["membership_group"] = overlap.apply(
        lambda r: (
            "Có mặt ở cả hai mốc"
            if r["in_vn30_effective_2020_02_03"] and r["in_fixed_universe_2025_01_02"]
            else "Chỉ ở mốc 2020"
            if r["in_vn30_effective_2020_02_03"]
            else "Chỉ ở mốc 2025"
        ),
        axis=1,
    )

    print("=" * 72)
    print("PHÂN TÍCH THAY ĐỔI THÀNH PHẦN VN30")
    print("=" * 72)
    print(f"Số kỳ review                         : {len(reviews)}")
    print(f"Số kỳ có thay đổi danh mục chính     : {summary['main_basket_changed'].sum()}")
    print(f"Tổng lượt thêm/loại                   : {summary['number_added'].sum()}")
    print(f"Số mã giao nhau giữa hai mốc          : {len(shared)}/30")
    print(f"Chỉ thuộc mốc 2020                    : {sorted(start_universe - end_universe)}")
    print(f"Chỉ thuộc universe đầu 2025           : {sorted(end_universe - start_universe)}")
    return summary, overlap


def save_chart(summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    labels = plot_df["review_period"].str.replace("-", "/", regex=False)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, plot_df["number_added"], label="Số mã thêm", color="#2E86AB")
    ax.bar(
        labels,
        -plot_df["number_removed"],
        label="Số mã loại",
        color="#D1495B",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Số mã thay đổi")
    ax.set_title("Thay đổi thành phần VN30 tại các kỳ review bán niên 2020-2025")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_project_directories()
    reviews = load_and_validate_reviews()
    summary, overlap = build_outputs(reviews)
    summary.to_csv(OUTPUT_REVIEW_SUMMARY, index=False, encoding="utf-8-sig")
    overlap.to_csv(OUTPUT_OVERLAP, index=False, encoding="utf-8-sig")
    save_chart(summary)
    print(f"\nĐã lưu: {OUTPUT_REVIEW_SUMMARY}")
    print(f"Đã lưu: {OUTPUT_OVERLAP}")
    print(f"Đã lưu: {OUTPUT_CHART}")


if __name__ == "__main__":
    main()
