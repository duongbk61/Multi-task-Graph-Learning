# Review vòng 1 — phần TRAIN: trạng thái & bàn giao

Trả lời các mục trong `VIEC_CHO_NGUOI_TRAIN.md`. File trong folder này:

| File | Nội dung |
|---|---|
| `01_FACTS_Q1_Q2_F.md` | **Q1, Q2, F-Encoder/ICVAE/Triplet/MTL-opt/Dataset/Preprocess/Baseline** + đơn vị threshold |
| `02_EG_compute.md` | **(E-G)** cấu hình máy, #params, thời gian, peak mem, throughput |
| `03_EF_stats.md` | **(E-F)** kiểm định thống kê (phần UnifiedHMSL vs MT-HGNN đã tính) |
| `04_EC_ablation_leaveoneout_PARTIAL.md` | **(E-C, một phần)** ablation trừ-dần CPA/TCG/Expert (Micro+Macro) |
| `05_main_comparison_grouped.md` | Bảng so sánh chính (baseline paper + MT-HGNN + UnifiedHMSL) |
| `06_EA_generative_ablation.md` | **(E-A) XONG** — 8 cấu hình ablation generative (ICVAE/triplet/Gaussian/VAE-no-cond/gen/cont) |
| `raw/` | JSON gốc: `ea_results.json`, `ea_vaenc.json` (+ các bảng khác) |

## Trạng thái từng yêu cầu

| Mục | Trạng thái | Ghi chú |
|---|---|---|
| **Q1** setup multi-task | ✅ XONG | 2 graph riêng; GNN chung, ICVAE riêng; loss cộng 1 bước |
| **Q2** leakage | ✅ XONG | Kết luận: **TRANSDUCTIVE** |
| **F-Encoder** | ✅ XONG | sum-agg, 2 lớp, không relation-weight, tanh, không norm |
| **F-ICVAE** | ✅ XONG | condition = feature bản thân + one-hot quan hệ; 3 view = 3 quan hệ |
| **F-Triplet** | ✅ XONG | 2 quan hệ trans = pos, call = neg; margin 0.3 cố định |
| **F-MTL-opt** | ✅ XONG | CE mean, 1:1, không class-weight |
| **F-Dataset** (số node, split, build-before-split) | ✅ XONG | vài mục nguồn dữ liệu **[CẦN KIỂM TRA]** (không có trong repo) |
| **Đơn vị threshold** | ✅ XONG | RAW ETH (không log/Wei); 269.0 là count |
| **F-Preprocess** | ✅ XONG | KHÔNG log/scale/clip; chỉ L1-row cho feature ICVAE |
| **F-Baseline-config** | ⚠️ MỘT PHẦN | baseline **lấy số từ Meta-IFD** (không có code) → cần footnote |
| **E-G** compute | ✅ XONG | vài mục (pretrain time, peak host mem) là ước lượng, đo lại nếu cần |
| **E-F** stats | ⚠️ MỘT PHẦN | UnifiedHMSL vs MT-HGNN: **KHÔNG significant**; vs Meta-IFD **[CẦN CHẠY]** |
| **E-C** ablation trừ-dần | ⚠️ MỘT PHẦN | đã có leave-one-out CPA/TCG/Expert; **cộng-dần + TCG-variants CHƯA** |
| **E-A** ablation generative | ✅ XONG | 8 cấu hình, 5 restart → `06_EA_...md`. ICVAE>Gaussian>raw; conditioning giúp nhẹ |
| **E-B** rule baselines | ❌ CHƯA | cần code baseline mới (DT/GBDT/MLP standalone, random-feature…) |
| **E-E** metric mở rộng | ❌ CHƯA | cần thêm PR-AUC/ROC-AUC/Bal-Acc/FPR vào eval rồi chạy lại model chính |
| **E-D** temporal/inductive split | ❌ CHƯA | bonus; cần mốc thời gian / định nghĩa inductive |

## ⚠️ Hai điểm TRUNG THỰC cần lưu ý khi viết
1. **Novelty chưa significant**: UnifiedHMSL vs MT-HGNN (bỏ 3 novelty) chênh +0.79 Ponzi / +0.07 Phish,
   **p ≫ 0.05** ở n=5. Cần thêm restart / giảm nhiễu, hoặc phát biểu dè dặt (xem `03_EF_stats.md`).
2. **Tái hiện dưới paper**: batch 64 (GPU 6 GB) thay vì 512 → Ponzi của ta thấp hơn Meta-IFD ~3 điểm.

## Việc train còn lại (theo thứ tự ưu tiên của thầy) — cần chốt scope
- ~~**E-A**~~ ✅ ĐÃ XONG (`06_EA_...md`).
- **E-E + E-F đầy đủ** (ưu tiên #3): thêm 4 metric mới (PR-AUC/ROC-AUC/Bal-Acc/FPR) + so sánh vs Meta-IFD.
- **E-C cộng-dần + TCG variants**: cần toggle ICVAE/triplet (đã có gen/cont) + 6 biến thể fusion.
- **E-B** rule baselines: nặng nhất, cần viết baseline mới.
- **E-D** temporal/inductive: bonus.

## Tóm tắt phát hiện E-A (mới)
- **Generative augmentation được xác nhận**: raw (77.38) tệ nhất; **ICVAE ≥ VAE-no-cond > Gaussian > raw**.
- **Nhánh contrastive không giúp Ponzi**: gen-only (81.45) > Full (79.66) — nêu trung thực.
- Interaction-conditioning giúp **nhẹ, trong nhiễu** (+0.57 Ponzi).

→ Xem tin nhắn kèm để chốt chạy mục nào trước (mỗi mục ~vài chục phút–vài giờ GPU).
