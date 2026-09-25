# Công cụ định lượng sàng lọc và xếp hạng cổ phiếu VN30

Công cụ Python hỗ trợ **sàng lọc và xếp hạng nhóm cổ phiếu VN30** theo xác suất vượt VN-Index trong 20 phiên giao dịch tiếp theo, được xây dựng trong quá trình thực tập tại **Công ty Cổ phần Chứng khoán VPS** (Phòng Tư vấn Đầu tư).

> Báo cáo thực tập tốt nghiệp — Trường Đại học Ngân hàng TP. Hồ Chí Minh, Khoa Khoa học Dữ liệu trong Kinh doanh.
> Sinh viên thực hiện: Đào Việt Anh (ĐH39KH01) — GVHD: TS. Nguyễn Khắc Cường.

---

## Giới thiệu

Việc theo dõi thủ công nhiều mã cổ phiếu (giá, khối lượng, xu hướng, biến động) trong quá trình phân tích đầu tư tốn thời gian, dễ sai sót và khó nhất quán giữa các lần phân tích. Dự án xây dựng một quy trình định lượng có thể tái sử dụng bằng Python để tự động hóa bước **sàng lọc ban đầu** trong rổ VN30, giúp chuyên viên rút gọn danh sách theo dõi trước khi đi vào phân tích chuyên sâu (định giá, đánh giá rủi ro, thông tin doanh nghiệp...).

**Bài toán:** mỗi ngày, dùng dữ liệu có đến cuối phiên để ước lượng xác suất từng cổ phiếu trong VN30 sẽ **vượt VN-Index trong 20 phiên tiếp theo**, từ đó xếp hạng và tạo nhóm Top-N ưu tiên phân tích.

## Dữ liệu

- Nguồn: thư viện [`vnstock`](https://github.com/thinh-vu/vnstock)
- Phạm vi: giá OHLCV theo ngày của 30 cổ phiếu VN30 (cố định theo thành phần tại 02/01/2025) và VN-Index, giai đoạn **02/01/2020 – 31/12/2025**
- Quy mô: ~44.638 quan sát cổ phiếu, 1.499 quan sát VN-Index
- Giai đoạn 2021–2024 dùng để xác thực theo thời gian và lựa chọn mô hình; năm 2025 là **tập kiểm định độc lập duy nhất**

## Phương pháp

- **20 đặc trưng** chia thành 3 nhóm: (1) diễn biến của từng cổ phiếu (lợi suất, xu hướng, biến động, động lượng, thanh khoản, Beta 60 phiên); (2) trạng thái VN-Index và chênh lệch cổ phiếu–thị trường; (3) vị trí tương đối của cổ phiếu so với các mã khác trong cùng ngày. Mọi đặc trưng tại thời điểm *t* chỉ dùng dữ liệu đã có đến *t*.
- **Biến mục tiêu:** nhị phân — cổ phiếu có vượt lợi suất VN-Index trong 20 phiên tiếp theo hay không.
- **Mô hình:** Hồi quy Logistic (baseline), Random Forest, XGBoost — cùng huấn luyện trên bộ 20 đặc trưng.
- **Xác thực:** chia theo trình tự thời gian (không chia ngẫu nhiên), đánh giá mở rộng qua các năm 2021–2024, có áp dụng cơ chế **purge/embargo** (loại 20 quan sát cuối tại ranh giới) để kiểm soát rò rỉ dữ liệu.
- **Chỉ tiêu đánh giá:** Accuracy/Precision/Recall/F1 so với baseline; Precision@N, Lift@N tại Top-3/Top-5/Top-10; mô phỏng đầu tư (giả định chi phí khứ hồi 0,4%/chu kỳ); MDD; Sharpe ratio quy đổi năm.

## Kết quả chính (kiểm định độc lập năm 2025)

| Chỉ tiêu | Kết quả |
|---|---|
| Mô hình được chọn | Hồi quy Logistic (theo tiêu chí Aggregate Lift@3, 2021–2024) |
| Accuracy | 51,50% (baseline dự báo toàn 0: 60,01%) |
| Precision@3 | 41,92% (tỷ lệ vượt chung: 39,99%) |
| Lift@3 | 1,048 |
| AUC-ROC (trung bình theo ngày) | 0,4966 |
| Lợi suất mô phỏng Top-3 (2025, sau phí 0,4%) | 44,75% |

**Lưu ý quan trọng:** AUC-ROC trên tập kiểm định 2025 xấp xỉ 0,50, tức trên toàn bộ 30 mã mô hình **không** xếp hạng tốt hơn ngẫu nhiên; mức cải thiện Lift@N chỉ thể hiện ở nhóm nhỏ đầu bảng xếp hạng. Khoảng tin cậy của chênh lệch Precision@N và lợi suất mô phỏng đều chứa 0, nên **chưa đủ bằng chứng thống kê** để khẳng định một lợi thế sàng lọc ổn định. Đóng góp chính của dự án là **một quy trình sàng lọc có thể tái lập và kiểm soát rò rỉ dữ liệu**, không phải một chiến lược đầu tư đã được chứng minh sinh lời. Kết quả nên được dùng như thông tin tham khảo cho bước sàng lọc sơ bộ, kết hợp với phân tích chuyên môn.

## Dashboard

Xây dựng bằng **Streamlit**, đọc trực tiếp kết quả đã tạo từ pipeline xử lý/huấn luyện. Gồm 4 tab:

- **Tổng quan thị trường** — diễn biến VN-Index, số mã vượt chỉ số, lợi suất tương đối 20 phiên của 30 mã
- **Sàng lọc cổ phiếu** — xếp hạng theo xác suất vượt VN-Index tại ngày lựa chọn, xuất được CSV
- **Dự báo và đánh giá** — so sánh 3 mô hình, chỉ tiêu Top-N, kết quả mô phỏng và khoảng tin cậy
- **Chi tiết cổ phiếu** — giá, lợi suất 20 phiên, lợi suất vượt VN-Index, RSI 14, Beta 60, lịch sử xác suất dự báo

## Cấu trúc dự án

```
.
├── thu_thap_du_lieu_vn30.py               # Thu thập và kiểm tra dữ liệu 30 mã, VN-Index
├── phan_tich_thanh_phan_vn30.py           # Đối chiếu thay đổi thành phần VN30 2020–2025
├── feature_engineering_relative_vn30.py   # Tạo 20 đặc trưng và biến mục tiêu vượt VN-Index
├── train_models_relative_vn30.py          # Purged walk-forward, chọn mô hình và kiểm định 2025
├── danh_gia_sang_loc_relative_vn30.py     # Top-N, lấy mẫu lặp lại (bootstrap) và mô phỏng có chi phí
├── dashboard_vn30_v2.py                   # Dashboard tương tác 4 nhóm chức năng (Streamlit)
├── chay_pipeline_v2.py                    # Chạy tuần tự toàn bộ pipeline
├── relative_config_vn30.py                # Danh sách đặc trưng, target, horizon, đường dẫn output
├── research_config.py                     # Cấu hình universe, khoảng nghiên cứu, chi phí, thư mục
└── requirements.txt                       # Phiên bản thư viện (Python 3.12)
```

## Cài đặt & chạy

Yêu cầu **Python 3.12**.

```bash
# 1. Tạo môi trường ảo và cài thư viện
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Chạy toàn bộ pipeline (thu thập dữ liệu → đặc trưng → huấn luyện → đánh giá)
python chay_pipeline_v2.py

# 3. Mở dashboard
python -m streamlit run dashboard_vn30_v2.py
```

Dashboard đọc các tệp `relative_*` được tạo ra bởi `feature_engineering_relative_vn30.py`, `train_models_relative_vn30.py` và `danh_gia_sang_loc_relative_vn30.py`, nên cần chạy pipeline trước khi mở dashboard. Chế độ "đánh giá lịch sử" trong dashboard mặc định tắt, chỉ bật khi cần xem nhãn/lợi suất tương lai để kiểm tra hậu nghiệm.

## Hạn chế & hướng phát triển

- Rổ 30 mã được **cố định** theo thành phần VN30 tại 02/01/2025 cho toàn bộ giai đoạn 2020–2025, tạo look-ahead/selection bias (chỉ 23/30 mã trùng nhau giữa đầu 2020 và đầu 2025). Hướng xử lý: xây dựng universe động theo thành phần VN30 có hiệu lực tại từng thời điểm.
- Mô hình mới chỉ dùng dữ liệu giá, khối lượng và đặc trưng thị trường; chưa có dữ liệu tài chính doanh nghiệp, định giá, ngành, tin tức, thanh khoản chi tiết.
- MDD hiện chỉ tính tại cuối kỳ; nên xây dựng đường giá trị danh mục theo từng ngày để đánh giá rủi ro đầy đủ hơn.
- VN-Index là chỉ số giá không bao gồm cổ tức trong khi giá cổ phiếu đã điều chỉnh cộng dồn cổ tức, tạo một chênh lệch hệ thống ước tính khoảng 0,1–0,15 điểm phần trăm mỗi kỳ 20 phiên khi so sánh vượt VN-Index.
- Cần kiểm định thêm trên dữ liệu mới, mô hình hóa rõ thuế/phí/trượt giá, và hiệu chỉnh xác suất dự báo.

## Công nghệ sử dụng

Python 3.12 · pandas / numpy · scikit-learn · XGBoost · vnstock · Streamlit

## Tài liệu tham khảo chính

- Jegadeesh, N. & Titman, S. (1993)
- Breiman, L. (2001), *Random Forests*
- Chen, T. & Guestrin, C. (2016), *XGBoost*
- Gu, S., Kelly, B. & Xiu, D. (2020)
- López de Prado, M. (2018), *Advances in Financial Machine Learning*
- Pedregosa, F. et al. (2011), *Scikit-learn*

## Tuyên bố miễn trừ trách nhiệm

Công cụ này là sản phẩm nghiên cứu/học thuật thực hiện trong khuôn khổ báo cáo thực tập tốt nghiệp, **không phải khuyến nghị đầu tư**. Kết quả sàng lọc chỉ mang tính tham khảo cho bước lọc sơ bộ và cần được kết hợp với phân tích chuyên môn, thông tin doanh nghiệp và đánh giá rủi ro trước khi ra quyết định đầu tư.

## License

Chưa xác định — thêm file `LICENSE` phù hợp trước khi công khai repository (đặc biệt nếu dữ liệu hoặc mã nguồn có ràng buộc từ VPS).
