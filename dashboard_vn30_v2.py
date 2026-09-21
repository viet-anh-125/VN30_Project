"""Dashboard V2 – sàng lọc khả năng vượt VN-Index sau 20 phiên."""
from pathlib import Path
import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_raw"

FILES = {
    "stock": DATA / "vn30_ohlcv_2020_2025.csv",
    "index": DATA / "vnindex_ohlcv_2020_2025.csv",
    "features": DATA / "vn30_features_relative_2020_2025.csv",
    "pred": DATA / "relative_test_predictions_2025.csv",
    "comparison": DATA / "relative_model_comparison_2021_2024.csv",
    "test": DATA / "relative_test_metrics_2025.csv",
    "topn": DATA / "relative_topn_summary_2021_2025.csv",
    "bootstrap": DATA / "relative_top3_block_bootstrap_2025.csv",
    "backtest": DATA / "relative_backtest_summary_2021_2025.csv",
    "backtest_boot": DATA / "relative_backtest_excess_bootstrap_2021_2025.csv",
    "model": DATA / "relative_model_best.joblib",
    "coefficients": DATA / "relative_logistic_coefficients.csv",
}

st.set_page_config(page_title="VN30 Quantitative Screening V2", layout="wide")


OPTIONAL_KEYS = {"coefficients"}  # file mới, không chặn dashboard nếu chưa chạy lại pipeline


def require_files():
    missing = [str(p) for k, p in FILES.items() if k not in OPTIONAL_KEYS and not p.exists()]
    if missing:
        st.error("Thiếu file đầu vào:\n" + "\n".join(missing))
        st.stop()


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


require_files()
stock = load_csv(FILES["stock"])
index_df = load_csv(FILES["index"])
features = load_csv(FILES["features"])
pred = load_csv(FILES["pred"])
comparison = load_csv(FILES["comparison"])
test_metrics = load_csv(FILES["test"])
topn = load_csv(FILES["topn"])
bootstrap = load_csv(FILES["bootstrap"])
backtest = load_csv(FILES["backtest"])
backtest_boot = load_csv(FILES["backtest_boot"])
model_package = load_model(FILES["model"])
coefficients = load_csv(FILES["coefficients"]) if FILES["coefficients"].exists() else None

for df in (stock, index_df, features):
    df["time"] = pd.to_datetime(df["time"])
pred["time"] = pd.to_datetime(pred["time"])
for df in (stock, features, pred):
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

st.title("Công cụ định lượng sàng lọc cổ phiếu VN30 – V2")
st.caption(
    "Mục tiêu: ước lượng khả năng cổ phiếu có lợi suất 20 phiên cao hơn VN-Index. "
    "Kết quả phục vụ hỗ trợ phân tích, không phải khuyến nghị đầu tư."
)

evaluation_mode = st.sidebar.checkbox(
    "Chế độ đánh giá lịch sử", value=False,
    help="Bật để xem nhãn thực tế; mặc định ẩn nhằm mô phỏng sử dụng công cụ."
)
available_dates = sorted(pred["time"].dropna().unique(), reverse=True)
selected_date = pd.Timestamp(st.sidebar.selectbox(
    "Ngày sàng lọc", options=available_dates, index=0,
    format_func=lambda x: pd.Timestamp(x).strftime("%d/%m/%Y")
))
symbols = sorted(stock["symbol"].unique())
selected_symbol = st.sidebar.selectbox("Mã cổ phiếu", symbols)

tabs = st.tabs(["Tổng quan thị trường", "Sàng lọc cổ phiếu", "Dự báo và đánh giá", "Chi tiết cổ phiếu"])

with tabs[0]:
    idx_2025 = index_df[index_df.time.dt.year == 2025].set_index("time")["close"]
    latest_feature = features[features.time <= selected_date].sort_values("time").groupby("symbol").tail(1)
    market_r20 = float(latest_feature["market_return_20d"].median())
    outperform_count = int((latest_feature["excess_return_20d"] > 0).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ngày phân tích", selected_date.strftime("%d/%m/%Y"))
    c2.metric("Số mã", len(latest_feature))
    c3.metric("Mã vượt VN-Index 20 phiên", f"{outperform_count}/30")
    c4.metric("Return 20 phiên VN-Index", f"{market_r20:.2%}")
    st.subheader("VN-Index năm 2025")
    st.line_chart(idx_2025)
    breadth = latest_feature[["symbol", "excess_return_20d"]].sort_values("excess_return_20d", ascending=False)
    st.subheader("Lợi suất tương đối 20 phiên của các cổ phiếu")
    breadth_chart = alt.Chart(breadth).mark_bar().encode(
        x=alt.X("symbol:N", title="Mã cổ phiếu", sort="-y", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("excess_return_20d:Q", title="Lợi suất tương đối", axis=alt.Axis(format=".0%")),
        color=alt.condition(
            alt.datum.excess_return_20d >= 0,
            alt.value("#1677c8"), alt.value("#d9534f")
        ),
        tooltip=[alt.Tooltip("symbol:N", title="Mã"), alt.Tooltip("excess_return_20d:Q", title="Lợi suất tương đối", format=".2%")],
    ).properties(height=360)
    st.altair_chart(breadth_chart, use_container_width=True)

with tabs[1]:
    day = pred[pred.time == selected_date].sort_values("probability_up", ascending=False).copy()
    current = features[features.time == selected_date][[
        "symbol", "return_20d", "excess_return_20d", "rsi_14",
        "volatility_20d", "beta_60d", "rank_return_20d"
    ]]
    day = day.merge(current, on="symbol", how="left", validate="one_to_one")
    day["xep_hang"] = np.arange(1, len(day) + 1)
    show_n = st.radio("Phạm vi hiển thị", [3, 5, 10, 30], horizontal=True, index=2)
    cols = ["xep_hang", "symbol", "probability_up", "return_20d", "excess_return_20d", "rsi_14", "volatility_20d", "beta_60d"]
    if evaluation_mode:
        cols += ["actual", "future_excess_return_20d"]
    shown = day[cols].head(show_n).rename(columns={
        "xep_hang": "Hạng", "symbol": "Mã", "probability_up": "Xác suất vượt VN-Index",
        "return_20d": "Return 20 phiên", "excess_return_20d": "Excess return hiện tại",
        "rsi_14": "RSI14", "volatility_20d": "Volatility20", "beta_60d": "Beta60",
        "actual": "Nhãn thực tế", "future_excess_return_20d": "Excess return tương lai",
    })
    for col in ["Return 20 phiên", "Excess return hiện tại", "Excess return tương lai"]:
        if col in shown.columns:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    for col in ["RSI14", "Volatility20", "Beta60"]:
        if col in shown.columns:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    st.dataframe(shown, use_container_width=True, hide_index=True, column_config={
        "Xác suất vượt VN-Index": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
    })
    st.download_button("Tải bảng sàng lọc CSV", day.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"screening_relative_{selected_date.date()}.csv", mime="text/csv")

with tabs[2]:
    st.subheader("So sánh mô hình – walk-forward 2021–2024")
    comparison_show = comparison.rename(columns={
        "model": "Mô hình", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1-score", "precision_at_3": "Precision@3",
        "market_prevalence": "Tỷ lệ vượt chỉ số chung",
        "aggregate_lift_at_3": "Lift@3",
        "mean_top3_future_excess_return": "Excess return Top-3",
    })
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "Precision@3", "Tỷ lệ vượt chỉ số chung", "Excess return Top-3"]:
        comparison_show[col] = comparison_show[col].map(lambda x: f"{x:.2%}")
    comparison_show["Lift@3"] = comparison_show["Lift@3"].map(lambda x: f"{x:.3f}")
    ordered_cols = ["Mô hình", "Accuracy", "Precision", "Recall", "F1-score", "Precision@3", "Tỷ lệ vượt chỉ số chung", "Lift@3", "Excess return Top-3"]
    st.dataframe(comparison_show[ordered_cols], use_container_width=True, hide_index=True)
    selected_model = model_package["model_name"]
    st.info(f"Mô hình được chọn: {selected_model}. Tiêu chí: Lift@3 → excess return Top-3 → F1.")

    row = test_metrics.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Accuracy 2025", f"{row['accuracy']:.2%}")
    b.metric("Precision 2025", f"{row['precision']:.2%}")
    c.metric("Precision@3", f"{row['precision_at_3']:.2%}")
    d.metric("Aggregate Lift@3", f"{row['aggregate_lift_at_3']:.3f}")

    if "auc_daily_mean" in test_metrics.columns:
        e, f = st.columns(2)
        e.metric("AUC-ROC (trung bình theo ngày)", f"{row['auc_daily_mean']:.4f}")
        f.metric("AUC-ROC (gộp toàn bộ quan sát)", f"{row['auc_pooled']:.4f}")
        st.caption(
            "AUC 0,50 = không phân biệt được; AUC trung bình theo ngày sát với "
            "bài toán xếp hạng trong cùng một ngày hơn AUC gộp."
        )
    else:
        st.info(
            "Chưa có cột AUC-ROC trong relative_test_metrics_2025.csv — chạy lại "
            "train_models_relative_vn30.py (bản đã cập nhật) để bổ sung."
        )

    if coefficients is not None:
        with st.expander("Hệ số Hồi quy Logistic (đặc trưng đã chuẩn hoá)"):
            st.dataframe(
                coefficients.rename(columns={
                    "dac_trung": "Đặc trưng", "he_so_chuan_hoa": "Hệ số (chuẩn hoá)",
                    "odds_ratio": "Odds ratio",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Hệ số dương: đặc trưng càng lớn thì xác suất vượt VN-Index càng cao. "
                "Vì đặc trưng đã chuẩn hoá, độ lớn |hệ số| so sánh được giữa các đặc trưng."
            )

    st.subheader("KPI sàng lọc Top-N")
    selected_topn = topn[(topn.evaluation_stage == "final_test") | (topn.evaluation_stage == "combined_oos_descriptive")].copy()
    selected_topn["Phạm vi"] = selected_topn["evaluation_stage"].map({"final_test": "Kiểm định 2025", "combined_oos_descriptive": "Ngoài mẫu 2021–2025"})
    selected_topn = selected_topn.rename(columns={"top_n": "Top-N", "observations": "Số ngày", "mean_precision_at_n": "Precision@N", "mean_market_prevalence": "Tỷ lệ chung", "aggregate_lift_at_n": "Lift@N"})
    for col in ["Precision@N", "Tỷ lệ chung"]:
        selected_topn[col] = selected_topn[col].map(lambda x: f"{x:.2%}")
    selected_topn["Lift@N"] = selected_topn["Lift@N"].map(lambda x: f"{x:.3f}")
    st.dataframe(selected_topn[["Phạm vi", "Top-N", "Số ngày", "Precision@N", "Tỷ lệ chung", "Lift@N"]], use_container_width=True, hide_index=True)

    st.subheader("Backtest sau chi phí 0,4%")
    bt = backtest[np.isclose(backtest.round_trip_cost_rate, .004)].copy()
    bt_focus = bt[(bt.evaluation_stage == "final_test") | (bt.evaluation_stage == "combined_oos_descriptive")].copy()
    bt_focus["Phạm vi"] = bt_focus["evaluation_stage"].map({"final_test": "Kiểm định 2025", "combined_oos_descriptive": "Ngoài mẫu 2021–2025"})
    bt_focus = bt_focus.rename(columns={"top_n": "Top-N", "observations": "Số kỳ", "net_final_return": "Lợi suất Top-N", "vnindex_final_return": "VN-Index", "equal_weight_30_final_return": "VN30 đồng trọng số", "sharpe_annualized": "Sharpe", "max_drawdown": "MDD"})
    for col in ["Lợi suất Top-N", "VN-Index", "VN30 đồng trọng số", "MDD"]:
        bt_focus[col] = bt_focus[col].map(lambda x: f"{x:.2%}")
    bt_focus["Sharpe"] = bt_focus["Sharpe"].map(lambda x: f"{x:.3f}")
    st.dataframe(bt_focus[["Phạm vi", "Top-N", "Số kỳ", "Lợi suất Top-N", "VN-Index", "VN30 đồng trọng số", "Sharpe", "MDD"]], use_container_width=True, hide_index=True)
    final_bt = bt[bt.evaluation_stage == "final_test"][[
        "top_n", "net_final_return", "vnindex_final_return", "equal_weight_30_final_return"
    ]].copy()
    chart = final_bt.melt(
        id_vars="top_n", var_name="Chuỗi", value_name="Lợi suất"
    )
    chart["Chuỗi"] = chart["Chuỗi"].map({
        "net_final_return": "Danh mục Top-N",
        "vnindex_final_return": "VN-Index",
        "equal_weight_30_final_return": "VN30 đồng trọng số",
    })
    grouped_chart = alt.Chart(chart).mark_bar().encode(
        x=alt.X("top_n:N", title="Danh mục"),
        xOffset="Chuỗi:N",
        y=alt.Y("Lợi suất:Q", axis=alt.Axis(format=".0%")),
        color=alt.Color("Chuỗi:N", title="So sánh"),
        tooltip=[alt.Tooltip("top_n:N", title="Top-N"), "Chuỗi:N", alt.Tooltip("Lợi suất:Q", format=".2%")],
    ).properties(height=350)
    st.altair_chart(grouped_chart, use_container_width=True)

    precision_diff = bootstrap[bootstrap.metric == "top3_precision_difference"].iloc[0]
    st.warning(
        f"CI 95% của chênh lệch Precision@3: {precision_diff.ci_95_lower:.2%} đến "
        f"{precision_diff.ci_95_upper:.2%}. Khoảng tin cậy chứa 0; chưa đủ bằng chứng "
        "khẳng định lợi thế thống kê ổn định."
    )
    st.dataframe(backtest_boot, use_container_width=True, hide_index=True)

with tabs[3]:
    history = stock[stock.symbol == selected_symbol].sort_values("time").set_index("time")
    feature_history = features[features.symbol == selected_symbol].sort_values("time")
    probability_history = pred[pred.symbol == selected_symbol].sort_values("time").set_index("time")
    st.subheader(f"Diễn biến giá {selected_symbol}")
    st.line_chart(history[["close"]])
    latest = feature_history[feature_history.time <= selected_date].tail(1)
    if not latest.empty:
        latest = latest.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Return 20 phiên", f"{latest['return_20d']:.2%}")
        m2.metric("Excess return 20 phiên", f"{latest['excess_return_20d']:.2%}")
        m3.metric("RSI14", f"{latest['rsi_14']:.2f}")
        m4.metric("Beta60", f"{latest['beta_60d']:.2f}")
    st.subheader("Xác suất vượt VN-Index trong 20 phiên tiếp theo")
    st.line_chart(probability_history[["probability_up"]])
    selected_prob = probability_history.loc[probability_history.index <= selected_date].tail(1)
    if not selected_prob.empty:
        st.metric("Xác suất tại ngày đã chọn", f"{selected_prob.iloc[0]['probability_up']:.2%}")

st.divider()
st.caption(
    "Giới hạn: universe cố định tại 02/01/2025; dữ liệu công khai; chưa mô hình hóa đầy đủ "
    "thanh khoản và trượt giá; kết quả bootstrap chưa chứng minh lợi thế có ý nghĩa thống kê."
)
