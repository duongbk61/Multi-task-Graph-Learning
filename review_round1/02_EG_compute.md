# (E-G) Compute cost — số thật đo trên máy train

Điền theo khung mục 3.6 của file thầy. Số đo trên chính máy đang chạy thí nghiệm.

```
GPU: NVIDIA GeForce GTX 1660 SUPER   VRAM: 6.44 GB (6 GB usable)
CPU: Intel Core (Family 6 Model 165 = Comet Lake, 4 cores / 8 threads)   RAM: 17 GB (16 GiB)
OS: Windows 10 Pro (10.0.19044)
PyTorch: 1.13.1+cu116   PyG: 2.4.0   CUDA: 11.6   Python: 3.10.0
torch-scatter 2.1.1+pt113cu116   torch-sparse 0.6.17+pt113cu116

#params (total):
  - UnifiedHMSL (GNN)         : 538,126  (~538.1K) trainable
  - ICVAE (mỗi task)          : 12,466   (~12.5K)  → 2 task = ~25K
  - Tổng (GNN + 2 ICVAE)      : ~563K

ICVAE pretrain: pretrain_epochs = 3 (utils.py), latent=50, Adam lr 0.01.
  → Thời gian pretrain THỰC: [CHƯA ĐO LẠI] — dùng checkpoint có sẵn (icvae_*.pkl).
     Có thể đo lại nếu cần, nhưng cần CSV thô trong ./dataset/ (hiện chỉ có data.pt đã xử lý).

GNN train time (batch_size=64, num_neighbors=[100,100]):
  - ~15 s/epoch (đo: 12 epoch = 201 s, gồm ~10 s khởi tạo)
  - Early-stop thực tế: best-val ~epoch 8–46, thường dừng ~30–50 epoch → ~8–13 phút / 1 lần train
  - 1 cấu hình × 5 restart (fixed split): ~20–25 phút (đo thực: mỗi restart ~200–500 s)

#epochs (thực): thay đổi theo restart, best-val epoch quan sát được: 8, 10, 31, 39, 46, ...
  (patience=20, epochs tối đa=60)

peak GPU mem:
  - Inference (batch 64): 577 MB
  - Training (batch 64): vừa khít trong 6 GB
  - batch 512 (như paper): OOM trên 6 GB → đây là lý do phải hạ batch = 64

peak host mem (RAM): xây NeighborSampler cho graph Phish (22M node EOA) là phần tốn RAM nhất;
  ~vài GB; nếu dựng nhiều loader cùng lúc từng gây thiếu RAM (đã fix bằng chia sẻ 1 sampler/graph).

inference time / throughput:
  - Ponzi test (269 node): 27.7 s → ~10 node/s
  - Nút cổ chai KHÔNG phải model mà là AUGMENTATION: mỗi batch chạy ICVAE inference
    ×concat(3) ×6 quan hệ. Nếu tắt augmentation, throughput cao hơn nhiều bậc.

1 unified model vs 2 single-task model (thời gian train):
  - Unified (multi-task, cả 2 task 1 lần train): ~10–13 phút/restart
  - 2 single-task tách rời: Ponzi-only ~5 phút + Phish-only ~7 phút = ~12 phút (đo từ ablation harness)
  → Unified ~ tương đương hoặc nhỉnh hơn về thời gian, nhưng cho RA MỘT model dùng chung
    (tiết kiệm tham số/triển khai) thay vì 2 model. [CON SỐ GẦN ĐÚNG — có thể đo lại chính xác nếu cần]
```

## Ghi chú cho bên viết
- Con số **batch 64 (không phải 512)** và **GPU 6 GB** cần nêu trung thực — đây là khác biệt then chốt so
  với paper Meta-IFD (batch 512) và là lý do bản tái hiện của ta thấp hơn Meta-IFD ~3 điểm Ponzi.
- Nếu cần **peak host mem / pretrain time / unified-vs-2single** đo chính xác (không phải ước lượng),
  báo mình chạy một buổi đo riêng.
