# VN30 Project V2 — Relative Outperformance

Công cụ định lượng hỗ trợ sàng lọc cổ phiếu theo xác suất vượt VN-Index trong 20 phiên tiếp theo. Dự án dùng dữ liệu 2020–2025, walk-forward 2021–2024 và final test 2025.

Chạy nhanh trên PowerShell:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
python chay_pipeline_v2.py
python -m streamlit run dashboard_vn30_v2.py
```

Xem hướng dẫn đầy đủ tại `HUONG_DAN_CHAY.md`.

