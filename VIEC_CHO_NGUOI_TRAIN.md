# Việc cần làm phía TRAIN/THỰC NGHIỆM — UnifiedHMSL (paper SoICT)

_Gửi: người đang cầm máy GPU + code. Cập nhật: 2026-07-22._

## Bối cảnh ngắn
Paper: **UnifiedHMSL** — multi-task heterogeneous GNN phát hiện **Ponzi (CA)** và **phishing (EOA)**
trên Ethereum, phát triển từ baseline **Meta-IFD** (Jin et al., arXiv:2408.00641). Thầy review vòng 1
yêu cầu **nói rõ setup thật + chạy thí nghiệm nghiêm túc hơn + thêm metric**. Dưới đây là phần **kỹ thuật**
cần bạn làm ở máy train và gửi kết quả về (bên viết paper sẽ ráp vào bài).

**Cách gửi về:** trả lời facts theo gạch đầu dòng; số điền vào các **khung bảng** ở cuối file (paste thẳng, kèm `mean±std` qua **5 lần chạy**, cùng seed giữa các model).

---

## 0. NÚT THẮT — trả lời TRƯỚC TIÊN (chỉ cần đọc code)

**(Q1) Multi-task setup — quan trọng nhất, quyết định cả bài viết:**
- Model train trên **MỘT graph chung** (CA+EOA cùng đồ thị) hay **HAI graph riêng** (graph Ponzi và graph phishing tách biệt)?
- Nếu 2 graph: **tham số nào dùng chung** (encoder? ICVAE?) và **tham số nào riêng** (head? CPA? TCG?)
- Trong 1 vòng lặp: batch của 2 task lấy thế nào (cùng batch / luân phiên)?
- Hai loss (Ponzi + phishing) **cộng cùng một bước** optimizer hay **cập nhật luân phiên**?
- Có **address xuất hiện ở cả hai** dataset không? Nếu có, xử lý & chia train/test ra sao?
- ICVAE: train **chung 1 cái** cho cả 2 task hay **riêng từng task**?

**(Q2) Data leakage — kiểm tra và trả lời có/không:**
- Train node có lấy **val/test node làm hàng xóm** trong NeighborLoader không?
- Feature node tính từ **toàn bộ lịch sử** hay chỉ dữ liệu **trước thời điểm** dự đoán? Có dùng giao dịch xảy ra **sau** khi account bị gán fraud không?
- Scaler/normalization và **threshold của decision tree** có được fit **chỉ trên train split** không?
- Có address nằm trong **train của task này** nhưng **test của task kia** không?
- ICVAE có dùng neighbourhood statistics liên quan val/test không?
- → Kết luận: đây là **transductive** hay **inductive**? (bên viết sẽ ghi rõ trong bài)

---

## 1. FACTS TỪ CODE (đọc code, không cần chạy — trả lời gạch đầu dòng)

**(F-Encoder)** AGG là mean/sum/attention? có relation-specific weight? self-loop? normalization? residual? activation & dropout sau mỗi layer? số layer & hidden dim? CA/EOA có projection layer riêng? 6 relation aggregate rồi concat hay sum?

**(F-ICVAE)** condition vector `c` tạo thế nào (1 neighbour / trung bình neighbourhood / relation-aggregate)? node không có edge của relation đó thì xử lý sao? 3 view ↔ chính xác 3 relation nào (cho CA và cho EOA)? ICVAE Ponzi vs phishing chung architecture/weight? pre-train mấy epoch? trọng số reconstruction vs KL? fit chỉ trên train?

**(F-Triplet)** 2 transfer relation cụ thể là gì? negative sample là node khác hay cùng node relation khác? distance tính trước hay sau graph-conv? có normalize representation trước khi tính loss? margin 0.3 từ validation hay cố định theo baseline?

**(F-MTL-opt)** cross-entropy dùng mean hay sum? batch size mỗi task? mỗi batch có cùng số CA và EOA target? có task weighting / class weighting không?

**(F-Dataset)** nguồn từng dataset; block range / khoảng thời gian; nguồn gán nhãn Ponzi & phishing; tiêu chí bỏ duplicate address; xử lý internal transaction; dùng ERC-20 transfer hay chỉ native ETH; xử lý contract-creation; cách expand 1st/2nd-order; **graph build TRƯỚC hay SAU khi split**; **số node train/val/test chính xác**; #fraud/#normal mỗi split; **% overlap giữa 2 dataset**.
→ **Đơn vị của feature**: các threshold rule −10, −12.75, 102.04, 269.0 là theo **Wei / GWei / ETH** hay đã **log-transform**?

**(F-Preprocess)** có log1p? clipping outlier? standardization? robust scaling? xử lý missing value? scaler fit-chỉ-trên-train? signed balance giữ nguyên hay biến đổi?

**(F-Baseline-config)** với GCN, GAT, GraphSAGE, RGCN, HGT, LightGBM (và baseline khác): số layer, hidden dim, learning rate, dropout, neighbour sampling, class weighting, early stopping, search space, số lần chạy, **số parameters**. → **Và cho biết: baseline định tự chạy lại hay lấy số từ Meta-IFD?** (nếu lấy từ Meta-IFD thì báo để bên viết ghi footnote.)

---

## 2. THÍ NGHIỆM CẦN CHẠY (đưa số về theo khung mục 3)

> Mọi số: **Macro-F1 (+ metric mục E) mean±std qua 5 lần chạy**, **cùng seed** giữa các model để làm paired-test.

**(E-A) Ablation Generative (title nhấn "Generative Feature Augmentation" nên bắt buộc):**
1. Full
2. − ICVAE (bỏ sinh feature, chỉ raw)
3. − triplet loss
4. thay ICVAE bằng **Gaussian noise augmentation**
5. **VAE không condition** theo interaction type (CVAE thường)
6. **ICVAE có condition** interaction type (= full, làm mốc so sánh với #5)
7. − generative path (chỉ contrastive)
8. − contrastive path (chỉ generative)

**(E-B) Baseline cho thành phần RULE (chứng minh rule đóng góp riêng, không chỉ là "thêm 1 feature"):**
- Decision Tree standalone · DT-probability-appended · LightGBM/XGBoost-score-appended ·
  MLP số-params-tương-đương · **random binary feature cùng coverage** · hard-rule feature ·
  soft-rule probability · rule feature-mode · rule aux-loss-mode · no-rule.

**(E-C) Ablation cộng dần (thay cho ablation kiểu "trừ dần" hiện tại):**
- encoder → +ICVAE → +triplet → +CPA → +TCG → +distilled-rules; và **Full single-task** vs **Full multi-task**.
- **TCG fusion variants:** average · fixed-scalar · learnable-global-scalar · concat+MLP · vector-valued-gate · **node-conditioned-scalar (đề xuất)**.

**(E-D) Chạy thêm split (nếu làm được):** **temporal split** hoặc **inductive split** (train trên account cũ, test account mới) — để chứng minh phát hiện fraud mới, không chỉ transductive.

**(E-E) Metric mới cho mọi cấu hình chính** (thay/bổ sung Micro-F1 vốn ít ý nghĩa):
PR-AUC · ROC-AUC · balanced accuracy · **fraud-class F1** · false-positive rate · confusion matrix · precision@(recall cố định) hoặc recall@(precision cố định).

**(E-F) Kiểm định thống kê:** cùng seed → paired differences → **95% CI** → **paired t-test hoặc Wilcoxon** (đặc biệt UnifiedHMSL vs Meta-IFD trên **cả Ponzi lẫn phishing** — phishing hiện chỉ +0.19, cần biết có ý nghĩa không).

**(E-G) Compute cost (số thật, thay các câu định tính):**
GPU (loại) · VRAM · CPU · RAM · PyTorch/PyG/CUDA version · **tổng #params** · thời gian pre-train ICVAE · thời gian train GNN · **số epoch thực** · peak GPU mem · peak host mem · inference time · throughput · **thời gian 1 unified model vs 2 single-task model**.

---

## 3. KHUNG ĐIỀN KẾT QUẢ (paste số vào đây rồi gửi về)

### 3.1 Ablation Generative (E-A) — Macro-F1 (%)
```
Config                         | Ponzi Macro-F1 | Phish Macro-F1
Full                           |                |
- ICVAE                        |                |
- triplet                      |                |
Gaussian-noise thay ICVAE      |                |
VAE (no interaction-condition) |                |
ICVAE (interaction-condition)  |                |
- generative path              |                |
- contrastive path             |                |
```

### 3.2 Rule baselines (E-B) — Macro-F1 (%)
```
Config                          | Ponzi | Phish
DT standalone                   |       |
DT-prob appended                |       |
GBDT-score appended             |       |
MLP (equiv params)              |       |
Random-binary-feature           |       |
Hard-rule feature               |       |
Soft-rule prob                  |       |
Rule feature-mode               |       |
Rule aux-loss-mode              |       |
No-rule                         |       |
```

### 3.3 Ablation cộng dần (E-C) — Macro-F1 (%)
```
Config                    | Ponzi | Phish
Encoder (base)            |       |
+ ICVAE                   |       |
+ triplet                 |       |
+ CPA                     |       |
+ TCG                     |       |
+ distilled rules         |       |
Full single-task          |       |
Full multi-task           |       |

TCG fusion variant        | Ponzi | Phish
average                   |       |
fixed scalar              |       |
learnable global scalar   |       |
concat + MLP              |       |
vector-valued gate        |       |
node-conditioned (ours)   |       |
```

### 3.4 Metric mở rộng (E-E) — cho các model chính
```
Model        | Data  | PR-AUC | ROC-AUC | Bal-Acc | Fraud-F1 | FPR | Macro-F1
Meta-IFD     | Ponzi |        |         |         |          |     |
UnifiedHMSL  | Ponzi |        |         |         |          |     |
Meta-IFD     | Phish |        |         |         |          |     |
UnifiedHMSL  | Phish |        |         |         |          |     |
(thêm baseline nếu chạy lại)
```

### 3.5 Thống kê (E-F)
```
So sánh                          | Data  | mean diff | 95% CI | test | p-value | significant?
UnifiedHMSL vs Meta-IFD          | Ponzi |           |        |      |         |
UnifiedHMSL vs Meta-IFD          | Phish |           |        |      |         |
UnifiedHMSL vs MT-HGNN           | Ponzi |           |        |      |         |
```

### 3.6 Compute (E-G)
```
GPU: ...   VRAM: ...   CPU: ...   RAM: ...
PyTorch: ...  PyG: ...  CUDA: ...
#params (total): ...
ICVAE pretrain time: ...   GNN train time: ...   #epochs (thực): ...
peak GPU mem: ...   peak host mem: ...
inference time: ...   throughput: ...
1 unified vs 2 single-task (thời gian): ...
```

---

## 4. Ưu tiên nếu thiếu thời gian
1. **Q1 + Q2** (setup + leakage) — trả lời ngay, chỉ đọc code, gỡ nút thắt cho phần viết.
2. **E-A (ablation ICVAE/triplet)** — thầy nói đây là **thiếu sót lớn** vì title nhấn generative.
3. **E-E + E-F** (PR-AUC + kiểm định thống kê) — để phát biểu kết quả trung thực.
4. **F-Dataset + F-Preprocess + đơn vị threshold** — cần cho phần dữ liệu.
5. **E-B, E-C, E-G** — làm tiếp khi còn thời gian.
6. **E-D (temporal/inductive split)** — bonus, thuyết phục nhất nếu làm được.
