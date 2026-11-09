# (E-A) Ablation Generative — Macro-F1 (%), fixed-split + 5 restarts

Cùng protocol với các bảng khác: fixed split (SPLIT_SEED=14), 5 restart {14,21,42,77,100},
batch 64. "Full" và "ICVAE (interaction-condition)" tái dùng từ `fixed_table` (cùng cấu hình).

### Khung 3.1 — điền số
```
Config                         | Ponzi Macro-F1 | Phish Macro-F1
Full                           | 79.66±1.76     | 97.24±0.64
- ICVAE (raw only)             | 77.38±1.69     | 97.14±0.54
- triplet                      | 78.90±2.04     | 96.77±0.52
Gaussian-noise thay ICVAE      | 78.96±0.53     | 96.52±0.57
VAE (no interaction-condition) | 79.09±1.79     | 96.99±0.69
ICVAE (interaction-condition)  | 79.66±1.76     | 97.24±0.64   (= Full, mốc so với dòng trên)
- generative path (cont only)  | 77.03±3.20     | 97.32±0.55
- contrastive path (gen only)  | 81.45±1.22     | 97.06±0.31
```

## Phân tích — TRUNG THỰC

### Điểm CỦNG CỐ luận điểm chính (title "Generative Feature Augmentation")
- **Bỏ ICVAE (chỉ raw) là TỆ NHẤT nhóm augmentation**: Ponzi 79.66 → **77.38 (−2.28)**. Tức là
  generative augmentation **đóng góp rõ** trên Ponzi.
- **Thứ tự: ICVAE (79.66) ≥ VAE-no-cond (79.09) > Gaussian-noise (78.96) > raw (77.38)**. →
  **Sinh feature (VAE/ICVAE) thắng nhiễu Gaussian, và tất cả đều hơn không augment.** Bằng chứng
  trực tiếp cho việc augmentation có cấu trúc là cần thiết (đúng nhấn mạnh của title). ✅
- **Interaction-conditioning (ICVAE vs VAE-no-cond)**: +0.57 Ponzi / +0.25 Phish — **dương nhưng NHỎ,
  nằm trong nhiễu** (std ~1.8/0.7). Tức là điều-kiện-theo-loại-quan-hệ (chữ "I" trong ICVAE) có xu
  hướng giúp nhưng **chưa đủ bằng chứng** ở quy mô này. Nên trình bày dè dặt: "conditioning giúp nhẹ".
- **Triplet giúp nhẹ**: −triplet → 78.90 (−0.76 Ponzi).
- **Nhánh generative quan trọng**: bỏ generative (chỉ contrastive) → 77.03 (−2.63 Ponzi).

### ⚠️ Điểm NGHỊCH cần trình bày trung thực
- **Nhánh contrastive KHÔNG có lợi trên Ponzi**: "chỉ generative" (bỏ contrastive) = **81.45**, còn
  CAO HƠN Full (79.66) ~ +1.79. Nghĩa là nhánh contrastive + CPA/TCG, ít nhất trên Ponzi, **không giúp
  mà còn hơi hại**. Khớp với ablation trước (CPA/TCG marginal). → Không nên tuyên bố nhánh contrastive
  cải thiện Ponzi.
- **Phish bão hòa ~97**: mọi cấu hình 96.5–97.3, chênh lệch **trong nhiễu** — E-A không phân biệt được
  trên Phish.

### Lưu ý về độ tin cậy
- std 1.2–3.2 (Ponzi) nên nhiều chênh lệch **giao nhau**. Tín hiệu **vững** nhất: (1) raw tệ nhất →
  ICVAE có ích; (2) ICVAE > Gaussian > raw; (3) gen-only cao. Các so sánh sát nhau (vd −triplet vs
  Gaussian) nằm trong nhiễu — nếu cần khẳng định, tăng restart (10–15) + paired test.
- Gaussian-noise dùng σ=0.1 thêm vào feature raw đã chuẩn hóa L1 (xem `run_ea.py`).

### Định nghĩa cấu hình (để bên viết mô tả đúng)
- **− ICVAE (raw only)**: 3 view = feature raw chuẩn hóa L1 (không sinh mới).
- **Gaussian-noise**: 3 view = raw chuẩn hóa + N(0, 0.1).
- **VAE (no interaction-cond)**: ICVAE nhưng edge-type = vector 0 (pretrain + inference) → chỉ điều
  kiện trên feature bản thân node, không theo loại quan hệ. (Pretrain trên **5M cạnh subsample** để
  tractable — đủ cho VAE 12K params; ICVAE gốc pretrain trên toàn bộ cạnh.)
- **− generative (cont only)** / **− contrastive (gen only)**: giữ đúng một trong hai nhánh biểu diễn.
