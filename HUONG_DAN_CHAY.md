# HƯỚNG DẪN CHẠY DỰ ÁN VN30 V2

## 1. Yêu cầu

- Windows 10/11, Python 3.12 64-bit.
- Chạy PowerShell tại thư mục `VN30_Project_V2_Final`.
- Bộ dữ liệu đầu vào 2020–2025 đã có sẵn trong `data_raw` nên không cần tải lại.

## 2. Tạo môi trường và cài thư viện

```powershell
py -3.12 -m venv .venv312
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra:

```powershell
python --version
```

Kết quả cần là Python 3.12.x.

## 3. Chạy toàn bộ pipeline từ đầu

```powershell
python chay_pipeline_v2.py
```

Pipeline tự chạy bốn bước trên universe chính (30 mã cố định theo VN30 đầu 2025):

1. Phân tích thay đổi thành phần VN30 (cũng tạo ra file giao nhau 23 mã dùng ở bước 3.3 bên dưới).
2. Tạo 20 feature và target vượt VN-Index sau 20 phiên.
3. Walk-forward Logistic Regression, Random Forest, XGBoost, kiểm định 2025.
   Từ bản cập nhật này, bước huấn luyện tự động xuất thêm **AUC-ROC** (Bảng 3.2/3.3)
   và **bảng hệ số Hồi quy Logistic** (`relative_logistic_coefficients.csv`, dùng
   cho Bảng 3.4 và mục 4.2).
4. KPI Top-N, bootstrap và backtest 55 kỳ có chi phí giao dịch.

Các file kết quả được tự động tạo trong `data_raw`; biểu đồ thay đổi thành phần được tạo trong `output_charts`.

### 3.1. Kiểm tra độ nhạy với 23 mã giao nhau (Nhận xét GVHD mục 2)

Chạy thêm cờ `--with-core23` để pipeline tự chạy NỐI TIẾP lần thứ hai chỉ
trên 23 mã có mặt ở cả mốc 03/02/2020 và 02/01/2025, rồi tự tạo bảng so sánh:

```powershell
python chay_pipeline_v2.py --with-core23
```

Kết quả:

- Toàn bộ file của lần chạy 23 mã có hậu tố `_core23` (ví dụ
  `relative_test_metrics_2025_core23.csv`), không ghi đè lên kết quả 30 mã.
- File `data_raw/bang_4_5_1_so_sanh_universe.csv` — dán trực tiếp vào mục
  4.5.1 của báo cáo, đã tự quy đổi Top-N theo đúng tỷ lệ rổ (ví dụ Top-3/30
  mã so với Top-2/23 mã) để không so sánh lệch chuẩn như bản đầu tiên.

Cũng có thể chạy tay từng lần bằng biến môi trường, nếu chỉ muốn chạy lại
một bước:

```powershell
$env:VN30_UNIVERSE_MODE = "core23"
python feature_engineering_relative_vn30.py
python train_models_relative_vn30.py
python danh_gia_sang_loc_relative_vn30.py
Remove-Item Env:\VN30_UNIVERSE_MODE   # quay lại universe chính (fixed30)
```

## 4. Mở dashboard

Sau khi pipeline hoàn tất:

```powershell
python -m streamlit run dashboard_vn30_v2.py
```

Trình duyệt thường mở tại `http://localhost:8501`.

## 5. Chạy từng bước khi cần kiểm tra lỗi

```powershell
python phan_tich_thanh_phan_vn30.py
python feature_engineering_relative_vn30.py
python train_models_relative_vn30.py
python danh_gia_sang_loc_relative_vn30.py
```

## 6. Kết quả kiểm tra chính

- Dữ liệu giá cổ phiếu: 44.638 dòng, 30 mã.
- Dataset mô hình: 42.238 dòng, 20 feature.
- Walk-forward: 2021–2024; final test độc lập: 2025.
- Mô hình được chọn: Logistic Regression.
- Final test 2025: Accuracy khoảng 51,50%; Precision@3 khoảng 41,92%; Lift@3 khoảng 1,048.
- Backtest: 11 kỳ/năm, tổng 55 kỳ; chi phí chính 0,4% hai chiều.

## 7. Ghi chú về file

- `feature_engineering_vn30.py` là mô-đun hàm nền được V2 import; không xóa.
- `relative_config_vn30.py` và `research_config.py` là file cấu hình; không xóa.
  `research_config.py` chứa `UNIVERSE_MODE` (đọc từ biến môi trường
  `VN30_UNIVERSE_MODE`, mặc định `fixed30`) và hàm `get_core23_symbols()`
  dùng cho phần kiểm tra độ nhạy bias.
- `so_sanh_universe.py` chỉ chạy được SAU KHI đã có đủ hai lần chạy
  (fixed30 và core23); nó chỉ đọc lại file CSV đã có, không huấn luyện lại.
- `thu_thap_du_lieu_vn30.py` chỉ dùng khi muốn thu thập lại dữ liệu. Pipeline mặc định dùng ba file dữ liệu đã kiểm tra đi kèm.
- Không đưa `.venv312`, `__pycache__` hoặc output V1 vào bài nộp.

