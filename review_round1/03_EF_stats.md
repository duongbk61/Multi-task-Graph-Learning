# (E-F) Kiểm định thống kê — phần đã tính được

Protocol: fixed split (SPLIT_SEED=14), **5 restart cùng seed** {14,21,42,77,100} cho cả 2 model
(paired). Chỉ số so sánh: **Macro-F1 (%)**. CI 95% dùng t-phân phối df=4 (t*=2.776);
p-value: paired t-test + Wilcoxon signed-rank (scipy).

### Khung 3.5 — điền số
```
So sánh                 | Data  | mean diff | 95% CI          | test        | p-value        | significant?
UnifiedHMSL vs MT-HGNN  | Ponzi | +0.79     | [-4.16, +5.74]  | paired t    | t=0.68 W=1.00  | KHÔNG
UnifiedHMSL vs MT-HGNN  | Phish | +0.07     | [-0.94, +1.08]  | paired t    | t=0.86 W=1.00  | KHÔNG
UnifiedHMSL vs Meta-IFD | Ponzi |  (chưa)   |                 |             |                | [CẦN CHẠY]
UnifiedHMSL vs Meta-IFD | Phish |  (chưa)   |                 |             |                | [CẦN CHẠY]
```
(MT-HGNN = UnifiedHMSL tắt cả 3 novelty CPA/TCG/Expert — baseline nội bộ gần Meta-IFD nhất hiện có.)

## Đọc kết quả — TRUNG THỰC, quan trọng cho bài
- Chênh lệch **UnifiedHMSL − MT-HGNN**: Ponzi **+0.79**, Phish **+0.07** — nhưng **KHÔNG đạt ý nghĩa
  thống kê** (p ≫ 0.05, CI 95% **chứa 0**). Tức là với ngân sách hiện tại (batch 64, 5 restart, nhiễu
  GPU cao), **chưa thể khẳng định 3 novelty cải thiện có ý nghĩa**.
- CI của Ponzi rất rộng (±5) vì phương sai giữa các restart lớn (nhiễu atomic-add GPU + test set nhỏ
  ~38 fraud). Đây đúng là mối lo thầy nêu ("phishing chỉ +0.19, có ý nghĩa không").
- **Hệ quả cho cách viết**: KHÔNG nên tuyên bố "novelty cải thiện đáng kể". Hai hướng xử lý chính danh:
  1. **Tăng số restart** (vd 15–20) + cố gắng giảm nhiễu (batch lớn hơn qua gradient-accumulation, hoặc
     bật deterministic) để CI hẹp lại — nếu hiệu ứng thật, p sẽ giảm.
  2. Nếu vẫn không đạt significance, phát biểu trung thực: novelty giúp **giảm phương sai / ổn định** và
     có xu hướng dương, nhưng chưa đủ bằng chứng thống kê ở quy mô dữ liệu này.

## Còn thiếu
- `UnifiedHMSL vs Meta-IFD`: cần **định nghĩa "Meta-IFD" trong code của ta** rồi chạy 5 restart cùng seed.
  (Ứng viên: model legacy `unified_model_legacy.py`, hoặc chính MT-HGNN. Cần chốt.)
