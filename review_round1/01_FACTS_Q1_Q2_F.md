# FACTS từ code — trả lời review vòng 1 (phần TRAIN)

_Trả lời dựa trên đọc code trực tiếp. Mọi tham chiếu ghi `file:line`. Chỗ nào **không xác định
được từ code** (nằm ở pipeline dựng dữ liệu upstream, không có trong repo) đều ghi rõ **[CẦN KIỂM TRA]**._

Môi trường số liệu dưới đây: torch 1.13.1+cu116, PyG 2.4.0, CUDA 11.6, GPU GTX 1660 Super 6 GB
(batch_size thực tế = 64 vì VRAM; paper Meta-IFD dùng 512).

---

## 0. NÚT THẮT

### (Q1) Multi-task setup

- **MỘT hay HAI graph?** → **HAI graph riêng biệt.** Có 2 đối tượng `HeteroData`: `data_ponzi`
  (`./data/Ponzi/`) và `data_phish` (`./data/Phish/`), load từ 2 file `processed/data.pt` khác nhau
  (`run_unified.py:172-179`). **Mỗi graph chứa CẢ hai loại node** CA + EOA và **cả 6 quan hệ**, nhưng
  là 2 đồ thị khác nhau: graph Ponzi tâm điểm các CA (3.104.188 node CA tổng, gồm cả unlabeled),
  graph Phish tâm điểm các EOA (22.429.276 node EOA tổng). Node có nhãn: Ponzi 1342 CA (191 fraud /
  1151 normal), Phish 2763 EOA (1206 phish / 1557 normal).
- **Tham số chung / riêng?** → **Toàn bộ GNN dùng CHUNG một model `UnifiedHMSL`** (`run_unified.py:204`).
  Chung: encoder `my_conv` (2 lớp), `lin_dict`, `lin_orig`, `CPA`, `TCG`, các `k/q/v_lin` — tất cả
  keyed theo **node-type** (CA/EOA có projection riêng, nhưng dùng chung giữa 2 task).
  **Riêng theo task** thực chất chỉ là 2 head: `head_ponzi` (áp lên node CA) và `head_phish` (áp lên
  node EOA) (`unified_model.py:186-188`). **ICVAE thì RIÊNG hoàn toàn từng task** — 2 checkpoint
  `icvae_Ponzi.pkl` và `icvae_Phish.pkl` pretrain độc lập (`run_unified.py:200-201`).
- **Batch 2 task lấy thế nào?** → `zip(train_loader_ponzi, train_loader_phish)` (`run_unified.py:80`):
  mỗi vòng lặp lấy **đồng thời** 1 batch Ponzi (seed = CA từ graph Ponzi) + 1 batch Phish (seed = EOA
  từ graph Phish). Hai batch từ hai graph khác nhau, **không phải luân phiên**.
- **Hai loss cộng cùng bước hay luân phiên?** → **Cộng cùng MỘT bước optimizer.**
  `loss = loss_ponzi + loss_phish + 0.1·(triplet_ponzi + triplet_phish)` rồi **một** `backward()` +
  `optimizer.step()` (`run_unified.py:99-111`). Không có task-weighting (tỉ lệ 1:1), không class-weighting.
  Triplet weight = `args.loss_train` = 0.1; nếu `expert_mode='loss'` thì cộng thêm `0.5·MSE(expert)`.
- **Address xuất hiện ở cả 2 dataset?** → **[CẦN KIỂM TRA]** code KHÔNG dedupe address chéo giữa 2
  dataset; 2 graph dựng từ 2 nguồn CSV riêng. `processed/data.pt` không lưu address gốc nên không tính
  được % overlap từ dữ liệu đã xử lý — cần danh sách address gốc (`./dataset/...csv`).
- **ICVAE chung hay riêng?** → **RIÊNG** từng task (2 checkpoint, cùng kiến trúc, khác trọng số).

### (Q2) Data leakage — kết luận: **TRANSDUCTIVE**

- **Train node có lấy val/test làm hàng xóm?** → **CÓ.** `NeighborLoader` thấy toàn bộ cấu trúc đồ thị;
  chỉ **nhãn** của val/test bị mask, còn node/feature/edge của val/test vẫn được sample làm neighbor của
  train node (`run_unified.py:183-197`). → transductive.
- **Feature dùng toàn bộ lịch sử hay chỉ trước thời điểm dự đoán?** → Feature là thống kê tổng hợp
  **precomputed** (tổng/đếm/trung bình), **không có mốc thời gian** trong code; nhiều khả năng dùng
  **toàn bộ lịch sử**. Không loại trừ được giao dịch xảy ra *sau* khi bị gán nhãn. **[CẦN KIỂM TRA]**
  ở pipeline dựng feature (không có trong repo).
- **Scaler / threshold DT fit chỉ trên train?** → **Không có scaler fit** (chỉ chuẩn hóa L1 theo hàng,
  xem F-Preprocess) nên không có rò rỉ scaler. **Threshold DT** trong `ExpertRules` là **hằng số
  hard-code**, comment ghi trích từ DT huấn luyện trên train split (`unified_model.py:58-92`). Đã "đóng
  băng" → với fixed-split (train seed-14) là hợp lệ; với K-fold CV thì KHÔNG re-fit từng fold (đã ghi
  caveat trong kết quả CV).
- **Address ở train task này nhưng test task kia?** → **[CẦN KIỂM TRA]** (phụ thuộc overlap ở Q1).
- **ICVAE dùng thống kê neighbourhood liên quan val/test?** → ICVAE pretrain trên **toàn bộ cạnh** của
  graph, gồm cả feature node val/test (`icvae_pretrain.py:25-73`), nhưng **unsupervised** (không thấy
  nhãn). Condition `c` = **feature của chính node đó** (không phải neighbourhood) (`run_unified.py:31`).
- → **KẾT LUẬN: TRANSDUCTIVE** (không phải inductive). Bên viết nên ghi rõ.

---

## 1. FACTS TỪ CODE

### (F-Encoder) — `attention_conv.py` (`my_conv`) + `unified_model.py`
- **AGG**: **SUM** (`aggr='add'`, `attention_conv.py:36`); gộp 6 quan hệ cũng bằng **sum** (`group='sum'`,
  dòng 88). Message = giá trị node nguồn đã chiếu (`v_j`), không có attention trong conv (`:95-98`).
- **Relation-specific weight?** → **KHÔNG.** `v_lin`/`a_lin` keyed theo **node-type**, **không** theo
  quan hệ; mọi quan hệ dùng chung `v_lin` của node-type nguồn (`:48-51, 71, 83`).
- **Self-loop?** → Không cộng self-loop; thay vào đó **nối (concat) feature của chính node** với tổng
  neighbor rồi qua `a_lin(tanh(·))` (`:89-90`). (Có tham số `skip` khai báo nhưng **không dùng** trong
  forward — nên ghi chú.)
- **Normalization?** → **Không** (không BatchNorm/LayerNorm). **Activation** = `tanh`. **Dropout** =
  **không** trong conv. (Dropout 0.3 chỉ có trong CPA/TCG — `unified_model.py:118,154`.)
- **Số layer / hidden**: **2 lớp** `my_conv` (`unified_model.py:199-200`), **hidden = 128**.
- **CA/EOA có projection riêng?** → **CÓ** (mọi ModuleDict keyed theo node-type: `lin_dict`,
  `lin_dict_mean`, `lin_orig`, `k/q/v_lin`, `cross_attn`, `task_gate` — `unified_model.py:178-197`).
- **6 quan hệ concat hay sum?** → **SUM** (rồi mới concat với feature bản thân node) — không concat 6
  quan hệ.

### (F-ICVAE) — `run_unified.py:18-37`, `icvae.py`, `icvae_pretrain.py`
- **Condition `c`**: = **feature RAW của chính node** (`data[src].x`), **không** trung bình neighbourhood
  (`run_unified.py:31`), cộng với **one-hot loại quan hệ** (1 trong 6).
- **Node không có cạnh của quan hệ đó?** → Vẫn sinh feature: hàm sinh cho **tất cả** node của node-type
  nguồn, one-hot lặp cho mọi node (`run_unified.py:25-37`) — không lọc theo bậc.
- **3 view ↔ 3 quan hệ nào?** → View = **3 quan hệ mà node-type đó là NGUỒN**:
  - **CA**: `CA-call-CA`, `CA-trans-CA`, `CA-trans-EOA`.
  - **EOA**: `EOA-call-CA`, `EOA-trans-CA`, `EOA-trans-EOA`.
  3 view này + **feature gốc** = 4 tensor, stack rồi **mean-pool** trong model; toàn bộ lặp lại
  `concat=3` lần (3 lần rút mẫu ngẫu nhiên) (`run_unified.py:48-70`, `unified_model.py:226-236`).
- **ICVAE Ponzi vs Phish**: **cùng kiến trúc** (latent=50, enc `[14,64]`, dec `[64,14]`), **khác trọng
  số** (`icvae_pretrain.py:56-60`).
- **Pretrain mấy epoch?** → `args.pretrain_epochs` mặc định **3** (`utils.py:15`); latent bị hard-code
  = **50** khi pretrain (`icvae_pretrain.py:57`, ghi đè default 10). Optimizer Adam lr 0.01.
- **Recon vs KL**: `loss = (BCE + KLD)/batch` — **tỉ lệ 1:1** (`icvae_pretrain.py:18-22`), BCE trên
  `sigmoid(x)`.
- **Fit chỉ trên train?** → **KHÔNG** — pretrain trên toàn bộ cạnh của graph (gồm node val/test),
  nhưng unsupervised (`icvae_pretrain.py:25-73`).

### (F-Triplet) — `unified_model.py:12-36, 284-306`
- **2 transfer relation cụ thể**: triplet lấy trên **các VIEW của cùng node** (không phải node khác).
  Thứ tự view CA = `[call-CA (v0), trans-CA (v1), trans-EOA (v2)]`. Code đặt **anchor=v1, positive=v2,
  negative=v0** (`:286-289`) → **kéo gần 2 quan hệ TRANS** (`trans-CA`, `trans-EOA`), **đẩy xa quan hệ
  CALL**. Đúng nghĩa "2 transfer relation" thầy hỏi.
- **Negative là node khác hay cùng node quan hệ khác?** → **Cùng node, quan hệ (view) khác.**
- **Distance trước hay sau graph-conv?** → **TRƯỚC** conv (tính trên embedding view qua `self.attention`,
  `:290-293`).
- **Normalize trước khi tính loss?** → **Không** normalize embedding.
- **Margin 0.3**: **cố định** (khởi tạo `TripletLoss(margin=0.3)`, `:170`), không tune từ validation.
  (Dùng `nn.TripletMarginLoss` p=2 khi có margin.)

### (F-MTL-opt) — `run_unified.py:74-116`
- **Cross-entropy**: reduction **mean** (mặc định) (`:91,96`).
- **Batch size mỗi task**: cùng `args.batch_size` (mặc định 512; chạy thực 64).
- **Mỗi batch có cùng #CA và #EOA target?** → Có cùng **số lượng** (đều `batch_size` seed node), nhưng
  khác node-type/graph: batch Ponzi = `batch_size` seed CA, batch Phish = `batch_size` seed EOA.
- **Task/class weighting?** → **Không** (1:1, không class weight). Triplet weight 0.1; expert MSE 0.5
  (chỉ ở `expert_mode='loss'`).

### (F-Dataset) — `dataset.py`
- **Nguồn**: Ponzi từ `./dataset/Ponzi_unbalance/` (`CA_labeled.csv`, `CA_unlabeled.csv`, `EOA.csv`,
  `edge/*.csv`); Phish node từ `./dataset/node_hete_EOA.csv` (`dataset.py:39-58`).
- **Số node train/val/test (chính xác)**: split stratified, `test_size=0.2` rồi `val=0.25` phần train
  (`dataset.py:122-131`) → **Ponzi**: train 804 / val 269 / test 269 (trên 1342 labeled);
  **Phish**: train 1657 / val 553 / test 553 (trên 2763). #fraud/#normal: Ponzi 191/1151, Phish 1206/1557.
- **Loại duplicate/empty**: bỏ CA có cả 4 count = 0 (`dataset.py:42-49`). Nhãn Ponzi gán cứng
  `[1]*191 + [0]*1151` (`:115`).
- **Graph build TRƯỚC hay SAU split?** → **TRƯỚC.** Đồ thị dựng một lần; split chỉ chia **chỉ số node có
  nhãn** (stratified, `SPLIT_SEED=14`) → **transductive** (`:116-160`).
- **Block range / khoảng thời gian / nguồn nhãn / internal txn / ERC-20 vs native ETH / contract-creation
  / cách expand 1st-2nd order**: **[CẦN KIỂM TRA]** — không có trong repo (dữ liệu là CSV precomputed từ
  pipeline upstream, nhiều khả năng kế thừa từ bộ dữ liệu Meta-IFD/HEIG).
- **% overlap giữa 2 dataset**: **[CẦN KIỂM TRA]** — cần address gốc (không lưu trong `data.pt`).

**→ Đơn vị feature & threshold** (đo trực tiếp từ `data['CA'].x` / `data['EOA'].x`):
- Giá trị RAW, khoảng **[−1,2×10⁷ … +8,4×10⁷]**, mean ≈ 75 (CA). → **KHÔNG log-transform** (log sẽ nén về
  1 chữ số), **KHÔNG Wei** (Wei ~1e18/ETH), **KHÔNG GWei**.
- Kết luận: cột **giá trị** ở đơn vị **ETH thô** (tổng luỹ kế), cột **đếm** là **số nguyên thô**.
  Do đó threshold: **−10, −12.75, 102.04 = ETH**; **269.0 = số đếm** (`trans_count_recv`).
- **Lưu ý quan trọng**: feature RAW **KHÔNG được chuẩn hóa** trước khi vào nhánh contrastive (`lin_orig`)
  và `ExpertRules` (chỉ feature do ICVAE sinh mới bị chuẩn hóa L1 theo hàng). Giá trị tới ~8,4×10⁷ đi
  thẳng vào Linear — nên nêu rõ trong phần dữ liệu.

### (F-Preprocess) — `utils.py:95-101`, `run_unified.py:32`
- **log1p? clipping? standardization? robust scaling?** → **Không có cái nào.**
- Biến đổi duy nhất: `feature_tensor_normalize` = **chuẩn hóa L1 theo HÀNG** (chia tổng hàng), và **chỉ áp
  cho feature ICVAE sinh ra** (`run_unified.py:32`), **không** cho feature raw.
- **Missing value**: bỏ CA all-zero-count; còn lại không xử lý riêng. **Signed balance**: giữ nguyên dấu
  (giá trị âm tồn tại, vd −1,2×10⁷).
- **Scaler fit chỉ trên train?** → Không có scaler cần fit (chuẩn hóa per-row, không phụ thuộc tập).

### (F-Baseline-config)
- **Repo KHÔNG chứa code baseline** (GCN/GAT/GraphSAGE/RGCN/HGT/LightGBM). → Số baseline trong bảng so
  sánh **lấy từ paper Meta-IFD** (arXiv:2408.00641). **Đề nghị bên viết ghi footnote**: "baseline results
  quoted from Meta-IFD". Ta **chỉ tự chạy các model của mình**: UnifiedHMSL (full), MT-HGNN (= tắt 3
  novelty), và bản tái hiện theo protocol Meta-IFD.
- **#params (của ta)**: UnifiedHMSL = **538,1K** trainable; ICVAE = **12,5K/task** (×2 = 25K).
- Baseline layer/hidden/lr/dropout/sampling/search-space: **[CẦN KIỂM TRA từ Meta-IFD]** (không có ở đây).
