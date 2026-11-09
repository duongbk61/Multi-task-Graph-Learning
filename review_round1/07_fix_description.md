# [6] Mô tả FIX (để viết Method) — cấu hình chuẩn MỚI: rule-free + fix

## Công thức chính xác
`x' = sign(x) · log1p(|x|) = sign(x) · ln(1 + |x|)` — dùng `torch.log1p` (giữ dấu, nén biên độ lớn).
Code: `unified_model.py` forward, `rt = {k: torch.sign(v)*torch.log1p(v.abs()) for k,v in raw_x_dict.items()}`.

## Áp cho INPUT nào
- Áp cho **feature RAW** (`raw_x_dict`) TRƯỚC khi vào:
  - **nhánh contrastive** `lin_orig` (tính `h_cont`), và
  - **TaskGate** (đầu vào `raw_x` để tính gate α).
- **KHÔNG** áp cho feature do ICVAE sinh (nhánh generative) — nhánh đó đi qua `lin_dict`, không dùng `rt`.
- Lý do: raw feature un-normalized tới ~8.4×10⁷ làm `tanh(lin_orig(x))` bão hòa tức thì → nhánh
  contrastive không học được. signed-log nén về ~[−18, 18].

## CA và EOA
- **Cả hai** (dict comprehension chạy trên mọi node-type: 'CA' và 'EOA').

## Feature ICVAE có còn L1-normalize?
- **CÓ, không đổi.** `feature_tensor_normalize` (chia tổng hàng) vẫn áp cho output ICVAE như cũ
  (`run_unified.py:32` / `run_ea.py`). Fix chỉ thêm ở nhánh contrastive, không đụng nhánh generative.

## ExpertRules — rule-free hoàn toàn?
- **Rule-free hoàn toàn về mặt tính toán.** Bật bằng cờ `expert_mode='none'`:
  - `ExpertRules.compute_ponzi/phish_score` **KHÔNG được gọi** (guard `if ... expert_mode != 'none'`).
  - Head input = `hidden` (không cộng +1 expert score); không có nhánh MSE expert.
  - Class `ExpertRules` vẫn nằm trong file nhưng là **dead code** (không tham gia forward/loss).
  → Nếu muốn "bỏ hẳn" về mặt mã nguồn, có thể xoá class, nhưng kết quả số **không đổi**.

## #params sau fix
- Fix signed-log **thêm 0 tham số** (chỉ là phép biến đổi, không có trọng số học).
- Rule-free (`expert_mode='none'`): **538,122** params (so với 538,126 khi có rule — ít hơn 4 do bỏ
  chiều +1 ở 2 head). ICVAE vẫn **12,5K/task** (không đổi).

## Cấu hình chuẩn dùng cho MỌI dòng (để so sánh công bằng)
`expert_mode=none`, `raw_transform=signed_log`, batch 64, neighbors [100,100]×2, hidden 128,
lr 0.001, λ(triplet)=0.1, ICVAE latent 50 (pretrain 3 epoch), concat=3 view, fixed split SEED=14,
5 restart {14,21,42,77,100}. Áp cho Full, mọi ablation, và MT-HGNN.
