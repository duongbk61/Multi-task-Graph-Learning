# Meta-IFD Project

Master thesis: Ethereum fraud detection (Ponzi + Phishing) using unified multi-task GNN.

## Architecture
- `UnifiedHMSL` in [unified_model.py](unified_model.py): multi-task heterogeneous GNN
- Two node types: CA (Contract Accounts) → Ponzi detection, EOA (Externally Owned Accounts) → Phishing detection
- Augmentation: CVAE pretrained models in `./pretrain_model/`
- Key components: `CrossPathAttention` + `TaskGate` + `ExpertRules` + `TripletLoss`
- Custom heterogeneous conv in [attention_conv.py](attention_conv.py): `my_conv` (sum aggregation with skip connections)

### Điểm khác biệt so với repo gốc (Task Unification)
Repo gốc train **hai model riêng biệt** — một cho Ponzi, một cho Phishing — không có shared representation hay joint optimization.

Meta-IFD hợp nhất thành **một model duy nhất** với:
- **Shared GNN backbone**: cùng `my_conv` layers xử lý cả CA và EOA nodes
- **Dual classification heads**: `head_ponzi` (CA) + `head_phish` (EOA) trên cùng embedding space
- **Joint training loop** ([run_unified.py:80](run_unified.py)): `zip(ponzi_loader, phish_loader)` — mỗi batch co-train cả hai task
- **Combined loss** ([run_unified.py:99](run_unified.py)): `loss = loss_ponzi + loss_phish + λ·loss_contrastive`
- **Joint model selection**: best model theo `f1_ponzi + f1_phish` thay vì chọn riêng từng task
- **Cross-task contrastive**: `TripletLoss` áp dụng đồng thời trên cả CA và EOA embeddings (`contrast_module`)

## Train command
```
python run_unified.py --expert_mode feature --aug_method cvae --hidden 128 --epochs 1000 --lr 0.001 --batch_size 512 --gpu 0
```

---

## Research Iterations

### Phase 0 — Repo gốc (Baseline, trước Dec 2024)
- Hai model hoàn toàn độc lập: một cho Ponzi (CA nodes), một cho Phishing (EOA nodes)
- Train riêng, evaluate riêng, không có shared representation
- Đây là baseline để so sánh hiệu quả của task unification

### Phase 1 — Base Unified Model (Dec 2024)
- **Hợp nhất hai task**: thay thế 2 model độc lập bằng `UnifiedHMSL` duy nhất với dual heads
- Joint training loop: co-train cả Ponzi và Phishing trong cùng một vòng lặp
- CVAE làm augmentation: sinh ra nhiều view ngẫu nhiên từ phân phối latent
- Multi-view pooling: mean over stochastic views, hai đầu phân loại riêng biệt (Ponzi / Phishing)
- TripletLoss để học embedding phân biệt class, áp dụng đồng thời cho cả CA và EOA

### Phase 2 — Expert Knowledge Integration (Mar 2026)
- **Knowledge-Guided Loss**: inject luật chuyên gia vào hàm loss (mode `loss`) thông qua MSE giữa output và expert score
- **Feature Fusion** (mode `feature`): nối expert score trực tiếp vào embedding trước khi phân loại — linh hoạt hơn hard MSE
- **ExpertRules** được cập nhật với tên feature ngữ nghĩa dựa trên bảng đặc trưng thực tế:
  - Ponzi: `call_total_sent`, `call_balance`, `trans_total_sent`, `trans_balance` + Decision Tree thresholds
  - Phishing: `trans_total_recv`, `trans_total_sent`, `call_total_sent` + DT-extracted thresholds
- Thêm arg `--expert_mode [loss | feature | none]` để dễ bật/tắt khi thực nghiệm
- Ngưỡng DT-extracted cụ thể từ Decision Tree huấn luyện trên tập train

### Phase 3 — Diffusion Augmentation Experiments (Apr 2–6, 2026)
- Thử nghiệm **Conditional DDPM** ([diffusion.py](diffusion.py)) thay thế CVAE làm augmentation:
  - `ConditionalNoisePredictor`: MLP dự đoán noise với time embedding + edge-type conditioning
  - `ConditionalDDPM`: forward/reverse diffusion process (100 steps, linear beta schedule)
- Tối ưu pipeline diffusion (beta schedule, sampling)
- Kết quả: CVAE vẫn ổn định hơn → giữ CVAE làm augmentation chính, diffusion không được merge vào luồng chính

### Phase 4 — CrossPathAttention + TaskGate (Apr 17–28, 2026)
- **CrossPathAttention** ([unified_model.py:50](unified_model.py)): mutual attention giữa hai luồng
  - *Generative path* (`h_gen`): self-attention over CVAE stochastic views
  - *Contrastive path* (`h_cont`): linear transform của raw features gốc
  - Hai luồng học từ nhau qua gated residual: `h' = h + sigmoid(q·k/√d) * v_other`
- **TaskGate** ([unified_model.py:96](unified_model.py)): task-conditioned gating riêng cho CA và EOA
  - `alpha = sigmoid(MLP(raw_x))` → blend giữa `h_gen'` và `h_cont'`
  - CA và EOA học chiến lược blend khác nhau từ cùng không gian đặc trưng
- Thêm `lin_orig` encoder riêng cho contrastive path
- Thử seed mới, cập nhật môi trường ([requirement.txt](requirement.txt))

---

## Key Files
| File | Vai trò |
|------|---------|
| [unified_model.py](unified_model.py) | Model chính: UnifiedHMSL, ExpertRules, CrossPathAttention, TaskGate |
| [run_unified.py](run_unified.py) | Training loop, args, evaluation |
| [icvae.py](icvae.py) | ICVAE augmentation model |
| [diffusion.py](diffusion.py) | Thực nghiệm Conditional DDPM (không dùng chính) |
| [attention_conv.py](attention_conv.py) | Custom HeteroConv layer |
| [dataset.py](dataset.py) | Ponzi / Phish dataset loaders |
| [utils.py](utils.py) | Helper: parser, normalization, seed |
