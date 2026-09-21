"""Đánh giá nâng cấp cho công cụ sàng lọc VN30.

Bao gồm:
1. Precision@Top-N và Lift@Top-N theo ngày.
2. Moving-block bootstrap cho chênh lệch Precision 2025.
3. Backtest tín hiệu ngày t, mua Open phiên t+1, giữ đúng 20 phiên
   (từ t+1 đến t+20) và bán theo Close phiên cuối.
4. So sánh đồng nhất với VN-Index và danh mục ngang trọng số các mã có dữ liệu.
5. Phân tích độ nhạy theo chi phí hai chiều.

Kết quả 2021-2024 là dự báo ngoài mẫu trong quá trình walk-forward validation.
Năm 2025 vẫn được báo cáo riêng là final test.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research_config import (
    DATA_DIR,
    TARGET_HORIZON,
    TRANSACTION_COST_SCENARIOS,
    UNIVERSE_MODE,
    ensure_project_directories,
)
from relative_config_vn30 import SUFFIX, out_name


FINAL_PRED_FILE = DATA_DIR / out_name("relative_test_predictions_2025")
WF_PRED_FILE = DATA_DIR / out_name("relative_walk_forward_predictions_2021_2024")
FEATURE_FILE = DATA_DIR / "vn30_ohlcv_2020_2025.csv"
INDEX_FILE = DATA_DIR / "vnindex_ohlcv_2020_2025.csv"
MODEL_FILE = DATA_DIR / f"relative_model_best{SUFFIX}.joblib"

OUT_TOPN_DAILY = DATA_DIR / out_name("relative_topn_daily_2021_2025")
OUT_TOPN_SUMMARY = DATA_DIR / out_name("relative_topn_summary_2021_2025")
OUT_BOOTSTRAP = DATA_DIR / out_name("relative_top3_block_bootstrap_2025")
OUT_BACKTEST_PERIODS = DATA_DIR / out_name("relative_backtest_periods_2021_2025")
OUT_BACKTEST_SUMMARY = DATA_DIR / out_name("relative_backtest_summary_2021_2025")
OUT_BACKTEST_BOOTSTRAP = DATA_DIR / out_name("relative_backtest_excess_bootstrap_2021_2025")
OUT_RUN_METADATA = DATA_DIR / f"relative_evaluation_metadata{SUFFIX}.json"

# fixed30: Top-3/5/10 trên 30 mã (10,0% / 16,7% / 33,3% rổ).
# core23 : giữ Top-3/5/10 để so trực tiếp CÙNG N với fixed30, đồng thời thêm
#          Top-2 và Top-8 xấp xỉ CÙNG TỶ LỆ rổ (8,7% và 34,8%) — xem Bảng
#          4.5.1 (nhận xét GVHD mục 3).
TOP_N_LIST = [3, 5, 10] if UNIVERSE_MODE == "fixed30" else [2, 3, 5, 8, 10]
BLOCK_LENGTH = 20
N_BOOTSTRAP = 2000
RANDOM_STATE = 42
TRADING_DAYS_PER_YEAR = 252


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Thiếu file đầu vào:\n- " + "\n- ".join(missing))


def load_predictions() -> tuple[pd.DataFrame, str]:
    require_files([FINAL_PRED_FILE, WF_PRED_FILE, MODEL_FILE])
    package = joblib.load(MODEL_FILE)
    best_model = str(package["model_name"])

    final_df = pd.read_csv(FINAL_PRED_FILE)
    final_df = final_df.rename(columns={"date": "time"})
    final_df["time"] = pd.to_datetime(final_df["time"])
    final_df["model"] = best_model
    final_df["evaluation_stage"] = "final_test"
    final_df["evaluation_year"] = 2025

    wf_df = pd.read_csv(WF_PRED_FILE)
    wf_df["time"] = pd.to_datetime(wf_df["time"])
    wf_df = wf_df[wf_df["model"] == best_model].copy()
    wf_df["evaluation_stage"] = "walk_forward_validation"
    wf_df["evaluation_year"] = wf_df["validation_year"].astype(int)

    required = {"time", "symbol", "actual", "prediction", "probability_up"}
    for label, frame in (("final", final_df), ("walk-forward", wf_df)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} predictions thiếu cột: {sorted(missing)}")

    predictions = pd.concat([wf_df, final_df], ignore_index=True, sort=False)
    predictions["symbol"] = predictions["symbol"].astype(str).str.upper().str.strip()
    predictions["actual"] = pd.to_numeric(predictions["actual"], errors="raise").astype(int)
    predictions["probability_up"] = pd.to_numeric(
        predictions["probability_up"], errors="raise"
    )
    predictions = predictions.sort_values(["time", "symbol"]).reset_index(drop=True)

    if predictions.duplicated(["time", "symbol"]).any():
        raise ValueError("Dự báo gộp bị trùng time-symbol.")
    if not predictions["probability_up"].between(0, 1).all():
        raise ValueError("Xác suất nằm ngoài [0,1].")
    return predictions, best_model


def calculate_topn_metrics(
    predictions: pd.DataFrame,
    top_n_list: list[int] = TOP_N_LIST,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    for date, day in predictions.groupby("time", sort=True):
        day = day.sort_values(["probability_up", "symbol"], ascending=[False, True])
        prevalence = float(day["actual"].mean())
        for top_n in top_n_list:
            if len(day) < top_n:
                continue
            selected = day.head(top_n)
            precision_at_n = float(selected["actual"].mean())
            lift_at_n = precision_at_n / prevalence if prevalence > 0 else np.nan
            records.append(
                {
                    "time": date,
                    "year": int(date.year),
                    "evaluation_stage": day["evaluation_stage"].iloc[0],
                    "top_n": top_n,
                    "available_symbols": len(day),
                    "selected_symbols": "|".join(selected["symbol"].tolist()),
                    "precision_at_n": precision_at_n,
                    "market_prevalence": prevalence,
                    "lift_at_n": lift_at_n,
                }
            )
    daily = pd.DataFrame(records)
    if daily.empty:
        raise ValueError("Không tính được KPI Top-N.")

    summary = (
        daily.groupby(["evaluation_stage", "year", "top_n"], as_index=False)
        .agg(
            observations=("time", "nunique"),
            mean_precision_at_n=("precision_at_n", "mean"),
            median_precision_at_n=("precision_at_n", "median"),
            mean_market_prevalence=("market_prevalence", "mean"),
            mean_lift_at_n=("lift_at_n", "mean"),
            share_days_lift_above_1=("lift_at_n", lambda x: float((x > 1).mean())),
        )
    )
    summary["aggregate_lift_at_n"] = (
        summary["mean_precision_at_n"] / summary["mean_market_prevalence"]
    )
    overall = (
        daily.groupby("top_n", as_index=False)
        .agg(
            observations=("time", "nunique"),
            mean_precision_at_n=("precision_at_n", "mean"),
            median_precision_at_n=("precision_at_n", "median"),
            mean_market_prevalence=("market_prevalence", "mean"),
            mean_lift_at_n=("lift_at_n", "mean"),
            share_days_lift_above_1=("lift_at_n", lambda x: float((x > 1).mean())),
        )
    )
    overall["aggregate_lift_at_n"] = (
        overall["mean_precision_at_n"] / overall["mean_market_prevalence"]
    )
    overall.insert(0, "year", "2021-2025_combined")
    overall.insert(0, "evaluation_stage", "combined_oos_descriptive")
    summary = pd.concat([summary, overall], ignore_index=True, sort=False)
    return daily, summary


def moving_block_bootstrap_precision(
    predictions_2025: pd.DataFrame,
    block_length: int = BLOCK_LENGTH,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap theo khối ngày cho Precision@3 trừ tỷ lệ outperform chung."""
    daily_rows = []
    for date, day in predictions_2025.groupby("time", sort=True):
        top3 = day.sort_values("probability_up", ascending=False).head(3)
        daily_rows.append({
            "time": date,
            "top3_precision": float(top3["actual"].mean()),
            "prevalence": float(day["actual"].mean()),
        })
    daily = pd.DataFrame(daily_rows).sort_values("time").reset_index(drop=True)
    dates = np.arange(len(daily))
    if len(dates) < block_length * 2:
        raise ValueError("Không đủ ngày cho moving-block bootstrap.")

    blocks = [dates[i : i + block_length] for i in range(len(dates) - block_length + 1)]
    n_blocks = int(np.ceil(len(dates) / block_length))
    rng = np.random.default_rng(random_state)
    rows = []

    for iteration in range(n_bootstrap):
        chosen = rng.integers(0, len(blocks), size=n_blocks)
        sampled_dates = np.concatenate([blocks[i] for i in chosen])[: len(dates)]
        sample = daily.iloc[sampled_dates]
        model_precision = float(sample["top3_precision"].mean())
        baseline_precision = float(sample["prevalence"].mean())
        rows.append(
            {
                "iteration": iteration,
                "top3_precision": model_precision,
                "market_prevalence": baseline_precision,
                "top3_precision_difference": model_precision - baseline_precision,
            }
        )

    boot = pd.DataFrame(rows).dropna()
    output = []
    for metric in ["top3_precision", "market_prevalence", "top3_precision_difference"]:
        output.append(
            {
                "metric": metric,
                "estimate_mean": float(boot[metric].mean()),
                "ci_95_lower": float(boot[metric].quantile(0.025)),
                "ci_95_upper": float(boot[metric].quantile(0.975)),
                "bootstrap_iterations": len(boot),
                "block_length_sessions": block_length,
            }
        )
    return pd.DataFrame(output)


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_files([FEATURE_FILE, INDEX_FILE])
    stock = pd.read_csv(FEATURE_FILE)
    index = pd.read_csv(INDEX_FILE)
    stock["time"] = pd.to_datetime(stock["time"])
    index["time"] = pd.to_datetime(index["time"])
    required_stock = {"time", "symbol", "open", "close"}
    if missing := required_stock - set(stock.columns):
        raise ValueError(f"File feature thiếu cột giá: {sorted(missing)}")
    required_index = {"time", "open", "close"}
    if missing := required_index - set(index.columns):
        raise ValueError(f"File VN-Index thiếu cột: {sorted(missing)}")
    stock["symbol"] = stock["symbol"].astype(str).str.upper().str.strip()
    for frame, columns, label in (
        (stock, ["open", "close"], "cổ phiếu"),
        (index, ["open", "close"], "VN-Index"),
    ):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[columns].isna().any().any():
            raise ValueError(f"Dữ liệu giá {label} có giá trị thiếu/không hợp lệ.")
        if (frame[columns] <= 0).any().any():
            raise ValueError(f"Dữ liệu giá {label} phải lớn hơn 0.")
    if stock.duplicated(["time", "symbol"]).any():
        raise ValueError("Dữ liệu cổ phiếu bị trùng time-symbol.")
    if index.duplicated(["time"]).any():
        raise ValueError("Dữ liệu VN-Index bị trùng ngày.")
    return stock.sort_values(["time", "symbol"]), index.sort_values("time")


def build_yearly_rebalance_calendar(
    prediction_year: pd.DataFrame,
    market_dates: list[pd.Timestamp],
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    signal_dates = sorted(prediction_year["time"].unique())
    available = set(signal_dates)
    year = int(prediction_year["evaluation_year"].iloc[0])
    year_market = [d for d in market_dates if d.year == year]
    position = {d: i for i, d in enumerate(year_market)}
    periods = []
    next_allowed_index = 0

    for signal in signal_dates:
        signal = pd.Timestamp(signal)
        if signal not in position or signal not in available:
            continue
        signal_index = position[signal]
        if signal_index < next_allowed_index:
            continue
        entry_index = signal_index + 1
        # entry_index là phiên t+1. Cộng horizon - 1 để dải
        # [entry_index, exit_index] gồm đúng TARGET_HORIZON phiên.
        exit_index = entry_index + TARGET_HORIZON - 1
        if exit_index >= len(year_market):
            break
        entry_date = year_market[entry_index]
        exit_date = year_market[exit_index]
        periods.append((signal, entry_date, exit_date))
        # Chỉ nhận tín hiệu mới sau khi kỳ hiện tại đã kết thúc. Quy tắc này
        # giữ các kỳ độc lập, không dùng chính ngày thoát làm ngày tín hiệu mới.
        next_allowed_index = exit_index + 1
    return periods


def run_enhanced_backtest(
    predictions: pd.DataFrame,
    stock_prices: pd.DataFrame,
    index_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_open = stock_prices.set_index(["time", "symbol"])["open"]
    stock_close = stock_prices.set_index(["time", "symbol"])["close"]
    index_by_date = index_prices.drop_duplicates("time").set_index("time")
    index_open = index_by_date["open"]
    index_close = index_by_date["close"]
    market_dates = list(index_close.index.sort_values())
    period_rows = []

    for year, pred_year in predictions.groupby("evaluation_year", sort=True):
        periods = build_yearly_rebalance_calendar(pred_year, market_dates)
        for period_id, (signal_date, entry_date, exit_date) in enumerate(periods, start=1):
            day = pred_year[pred_year["time"] == signal_date].sort_values(
                ["probability_up", "symbol"], ascending=[False, True]
            )
            if day.empty:
                continue
            all_returns = {}
            for symbol in day["symbol"]:
                entry_key = (entry_date, symbol)
                exit_key = (exit_date, symbol)
                if entry_key not in stock_open.index or exit_key not in stock_close.index:
                    continue
                entry_price = float(stock_open.loc[entry_key])
                exit_price = float(stock_close.loc[exit_key])
                if entry_price > 0:
                    all_returns[symbol] = exit_price / entry_price - 1
            if len(all_returns) < max(TOP_N_LIST):
                raise ValueError(
                    f"{signal_date.date()}: chỉ có {len(all_returns)} mã đủ giá backtest."
                )
            if entry_date not in index_open.index or exit_date not in index_close.index:
                raise ValueError(
                    f"Thiếu giá VN-Index tại kỳ {entry_date.date()} -> {exit_date.date()}."
                )
            # Cùng quy ước với cổ phiếu: mua Open t+1, bán Close phiên thoát.
            benchmark_return = float(
                index_close.loc[exit_date] / index_open.loc[entry_date] - 1
            )
            equal_weight_gross_return = float(np.mean(list(all_returns.values())))
            available_universe_size = len(all_returns)

            for top_n in TOP_N_LIST:
                selected = [s for s in day["symbol"] if s in all_returns][:top_n]
                gross_return = float(np.mean([all_returns[s] for s in selected]))
                for cost_rate in TRANSACTION_COST_SCENARIOS:
                    period_rows.append(
                        {
                            "evaluation_stage": pred_year["evaluation_stage"].iloc[0],
                            "year": int(year),
                            "period_id": period_id,
                            "signal_date": signal_date,
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                            "top_n": top_n,
                            "selected_symbols": "|".join(selected),
                            "gross_return": gross_return,
                            "round_trip_cost_rate": cost_rate,
                            "net_return": gross_return - cost_rate,
                            # Giữ tên cột cũ để dashboard hiện tại vẫn đọc được,
                            # nhưng giá trị đã là lợi suất SAU PHÍ giống Top-N.
                            "equal_weight_30_return": equal_weight_gross_return - cost_rate,
                            "equal_weight_gross_return": equal_weight_gross_return,
                            "available_universe_size": available_universe_size,
                            "vnindex_return": benchmark_return,
                        }
                    )

    periods_df = pd.DataFrame(period_rows)
    if periods_df.empty:
        raise ValueError("Không tạo được kỳ backtest nâng cấp.")

    summary_rows = []
    group_cols = ["evaluation_stage", "year", "top_n", "round_trip_cost_rate"]
    for keys, group in periods_df.groupby(group_cols, sort=True):
        stage, year, top_n, cost_rate = keys
        rp = group["net_return"].astype(float)
        rm = group["vnindex_return"].astype(float)
        rew = group["equal_weight_30_return"].astype(float)
        wealth = (1 + rp).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        std = rp.std(ddof=1)
        sharpe = rp.mean() / std * np.sqrt(252 / TARGET_HORIZON) if len(rp) > 1 and std > 0 else np.nan
        summary_rows.append(
            {
                "evaluation_stage": stage,
                "year": year,
                "top_n": top_n,
                "round_trip_cost_rate": cost_rate,
                "observations": len(group),
                "gross_final_return": float(np.prod(1 + group["gross_return"]) - 1),
                "net_final_return": float(np.prod(1 + rp) - 1),
                "vnindex_final_return": float(np.prod(1 + rm) - 1),
                "equal_weight_30_final_return": float(np.prod(1 + rew) - 1),
                "excess_vs_vnindex": float(np.prod(1 + rp) - np.prod(1 + rm)),
                "excess_vs_equal_weight_30": float(np.prod(1 + rp) - np.prod(1 + rew)),
                "sharpe_annualized": sharpe,
                # MDD từ chuỗi tài sản tại cuối mỗi kỳ tái cân bằng; không phải
                # MDD nội kỳ theo dữ liệu hằng ngày.
                "max_drawdown": float(drawdown.min()),
                "max_drawdown_method": "period_end_equity_curve",
                "positive_period_rate": float((rp > 0).mean()),
            }
        )
    # Kết quả gộp toàn bộ 55 kỳ ngoài mẫu, không chỉ từng năm riêng lẻ.
    for (top_n, cost_rate), group in periods_df.groupby(
        ["top_n", "round_trip_cost_rate"], sort=True
    ):
        group = group.sort_values(["year", "period_id"])
        rp = group["net_return"].astype(float)
        rm = group["vnindex_return"].astype(float)
        rew = group["equal_weight_30_return"].astype(float)
        wealth = (1 + rp).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        std = rp.std(ddof=1)
        sharpe = rp.mean() / std * np.sqrt(252 / TARGET_HORIZON) if std > 0 else np.nan
        summary_rows.append({
            "evaluation_stage": "combined_oos_descriptive",
            "year": "2021-2025",
            "top_n": top_n,
            "round_trip_cost_rate": cost_rate,
            "observations": len(group),
            "gross_final_return": float(np.prod(1 + group["gross_return"]) - 1),
            "net_final_return": float(np.prod(1 + rp) - 1),
            "vnindex_final_return": float(np.prod(1 + rm) - 1),
            "equal_weight_30_final_return": float(np.prod(1 + rew) - 1),
            "excess_vs_vnindex": float(np.prod(1 + rp) - np.prod(1 + rm)),
            "excess_vs_equal_weight_30": float(np.prod(1 + rp) - np.prod(1 + rew)),
            "sharpe_annualized": sharpe,
            "max_drawdown": float(drawdown.min()),
            "max_drawdown_method": "period_end_equity_curve",
            "positive_period_rate": float((rp > 0).mean()),
        })
    return periods_df, pd.DataFrame(summary_rows)


def bootstrap_backtest_excess(periods_df, cost_rate=0.004, iterations=5000):
    """Bootstrap ghép cặp 55 kỳ không chồng lấn: Top-N trừ VN-Index."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    selected_cost = periods_df[np.isclose(periods_df["round_trip_cost_rate"], cost_rate)]
    for top_n, group in selected_cost.groupby("top_n"):
        excess = (group["net_return"] - group["vnindex_return"]).to_numpy(float)
        estimates = np.empty(iterations)
        for i in range(iterations):
            estimates[i] = rng.choice(excess, size=len(excess), replace=True).mean()
        rows.append({
            "top_n": int(top_n), "round_trip_cost_rate": cost_rate,
            "observations": len(excess),
            "mean_period_excess_estimate": float(excess.mean()),
            "ci_95_lower": float(np.quantile(estimates, .025)),
            "ci_95_upper": float(np.quantile(estimates, .975)),
            "bootstrap_iterations": iterations,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_project_directories()
    predictions, best_model = load_predictions()
    topn_daily, topn_summary = calculate_topn_metrics(predictions)
    final_2025 = predictions[predictions["evaluation_stage"] == "final_test"].copy()
    bootstrap = moving_block_bootstrap_precision(final_2025)
    stock_prices, index_prices = load_prices()
    backtest_periods, backtest_summary = run_enhanced_backtest(
        predictions, stock_prices, index_prices
    )
    backtest_bootstrap = bootstrap_backtest_excess(backtest_periods)

    topn_daily.to_csv(OUT_TOPN_DAILY, index=False, encoding="utf-8-sig")
    topn_summary.to_csv(OUT_TOPN_SUMMARY, index=False, encoding="utf-8-sig")
    bootstrap.to_csv(OUT_BOOTSTRAP, index=False, encoding="utf-8-sig")
    backtest_periods.to_csv(OUT_BACKTEST_PERIODS, index=False, encoding="utf-8-sig")
    backtest_summary.to_csv(OUT_BACKTEST_SUMMARY, index=False, encoding="utf-8-sig")
    backtest_bootstrap.to_csv(OUT_BACKTEST_BOOTSTRAP, index=False, encoding="utf-8-sig")

    metadata = {
        "universe_mode": UNIVERSE_MODE,
        "best_model": best_model,
        "target_horizon": TARGET_HORIZON,
        "top_n_list": TOP_N_LIST,
        "bootstrap_block_length": BLOCK_LENGTH,
        "bootstrap_iterations": N_BOOTSTRAP,
        "execution_rule": "signal_close_t_entry_open_t_plus_1_hold_exactly_20_sessions_exit_close_t_plus_20",
        "benchmark_rule": "vnindex_entry_open_t_plus_1_exit_close_same_exit_date",
        "equal_weight_rule": "available_symbols_only_and_same_round_trip_cost_as_top_n",
        "max_drawdown_method": "period_end_equity_curve_not_intraperiod_daily_mdd",
        "transaction_cost_scenarios": TRANSACTION_COST_SCENARIOS,
        "walk_forward_years": [2021, 2022, 2023, 2024],
        "final_test_year": 2025,
    }
    OUT_RUN_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("HOÀN TẤT ĐÁNH GIÁ RELATIVE/OUTPERFORMANCE V2")
    print("=" * 80)
    print(f"Model được đánh giá : {best_model}")
    print(f"Số ngày KPI Top-N   : {topn_daily['time'].nunique()}")
    print(f"Số kỳ backtest      : {backtest_periods[['year','period_id']].drop_duplicates().shape[0]}")
    print("\nKết quả bootstrap 2025:")
    print(bootstrap.to_string(index=False))
    print("\nBootstrap chênh lệch lợi suất mỗi kỳ so với VN-Index (phí 0,4%):")
    print(backtest_bootstrap.to_string(index=False))
    print("\nĐã lưu các file relative_* trong data_raw.")


if __name__ == "__main__":
    main()
