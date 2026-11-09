# Meta-IFD / UnifiedHMSL — Tổng hợp toàn bộ công việc

_Cập nhật: 2026-07 (máy train GTX 1660S 6GB). Mọi số Macro-F1 (%), fixed-split (SPLIT_SEED=14),
mean±std qua 5 restart {14,21,42,77,100}, trừ khi ghi khác._

## 1. Môi trường (xong)
- Stack CUDA 11.6: torch 1.13.1+cu116, PyG 2.4.0, torch-scatter/sparse pt113cu116 (khớp driver 512.95).
- `data → HEIG` junction; dùng `icvae_{Ponzi,Phish}.pkl` (no underscore, module `icvae`, latent 50).
- **batch_size=64** (batch 512 của paper OOM trên 6GB) — khác biệt then chốt so với paper.

## 2. Kết quả chính — MỘT BẢNG

| Model / cấu hình | Ponzi Macro | Phish Macro | Ghi chú |
|---|---|---|---|
| Meta-IFD (paper) | 82.54±1.20 | 97.94±0.24 | số từ paper (batch 512) |
| MT-HGNN (bỏ cả 3 novelty) | 78.87±2.67 | 97.17±0.56 | baseline nội bộ |
| **UnifiedHMSL (full, chưa fix)** | 79.66±1.76 | 97.24±0.64 | tái hiện |
| **UnifiedHMSL (full, + FIX scaling)** ⭐ | **81.28±0.98** | 96.96±0.47 | **cải thiện thật, std chặt hơn** |
| UnifiedHMSL — 5-fold CV | 80.95–81.9 | 96.4–96.8 | protocol chặt nhất |
| best single run (seed 77) | 82.73 | 97.06 | chỉ báo kèm, KHÔNG headline |

## 3. Ablation & thí nghiệm review vòng 1 (folder `review_round1/`)
- **E-A Generative** (8 cấu hình): raw(77.38) tệ nhất → **ICVAE(79.66) ≥ VAE-nocond(79.09) > Gaussian(78.96) > raw**.
  ⇒ **Generative augmentation được chứng minh** (khớp title). Interaction-cond +0.57 (trong nhiễu).
- **E-C leave-one-out** (Micro+Macro): TCG là novelty rõ nhất; CPA/Expert marginal.
- **E-F thống kê**: UnifiedHMSL vs MT-HGNN **KHÔNG significant** (Ponzi p=0.68, Phish p=0.86, CI chứa 0).
- **E-G compute**: 538K params (+12.5K/ICVAE), ~15s/epoch, peak infer 577MB.
- **Facts Q1/Q2/F-***: 2 graph riêng, GNN chung + ICVAE riêng, loss cộng 1 bước; **TRANSDUCTIVE**;
  feature RAW ETH không log/chuẩn hóa.

## 4. ⭐ Cải thiện MỚI (cái bạn nhớ) — FIX chuẩn hóa đầu vào
- **Phát hiện lỗi thật**: feature raw (tới 8.4×10⁷) đi thẳng vào nhánh contrastive/TaskGate → tanh bão hòa,
  nhánh contrastive gần như không học. Sửa: **signed-log** cho đầu vào contrastive (giữ ExpertRules trên ETH thô).
- **Kết quả**: Full **79.66 → 81.28 (+1.62)**, std **1.76 → 0.98**. Đây là cải thiện **chính danh, báo cáo được**.

### NHƯNG — sự thật cần giữ trung thực
Ablation +fix cho thấy cải thiện đến từ FIX, **không phải từ CPA/TCG**:

| +fix | Ponzi | Phish |
|---|---|---|
| Full | 81.28±0.98 | **96.96±0.47** |
| w/o CPA | 81.90±1.81 | 96.49 |
| w/o TCG | 81.07±2.08 | 96.45 |
| gen-only (bỏ contrastive) | 81.94±2.53 | 96.37 |

→ Trên **Ponzi**, bỏ novelty vẫn ngang/ cao hơn → novelty chưa chứng minh được là nguồn cải thiện.
→ Trên **Phish**, Full > mọi ablation (novelty giúp nhẹ, nhất quán nhưng trong nhiễu).

## 5. Cái GÌ báo cáo được (chắc thắng) vs KHÔNG
**✅ Chứng minh được:** (a) Generative augmentation có cấu trúc (ICVAE > noise > raw); (b) FIX scaling
kéo Ponzi 79.66→81.28; (c) unified một-model cho 2 task; (d) trên Phish full model tốt nhất.
**❌ CHƯA chứng minh:** CPA/TCG cải thiện có ý nghĩa trên Ponzi; novelty vs Meta-IFD significant.

## 6. Deliverables đã gom
- `results_export/` (+.zip): bảng so sánh + CV + ablation + split-sensitivity.
- `review_round1/` (+.zip): facts + E-A + E-C + E-F + E-G cho bên viết paper.
- `comparison_table.*`, `image_table.*`, `ablation_table.*`: các bảng theo mẫu.

## 7. Hướng chính danh để chốt (chờ quyết định)
1. **Đưa FIX vào là kết quả chính** + chạy lại 5-fold CV / bảng so sánh với fix (số headline cao hơn, thật).
2. Tăng restart 10–15 để thử làm tín hiệu Phish-novelty significant.
3. Gradient-accumulation batch 512 (như paper) đẩy Ponzi tuyệt đối lên gần/qua 82.54.
4. Viết lại framing quanh generative + fix + unified (cái chắc thắng), nói dè dặt về CPA/TCG.
