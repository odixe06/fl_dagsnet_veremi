# PerFed-SKD trên VeReMi NextGen / DAGSNet — quyết định, deviation, hợp đồng artifact

Bản dựng lại phương pháp của [`paper.md`](paper.md) cho bộ dữ liệu VeReMi NextGen với bộ phân
loại DAGSNet. File này là **nguồn duy nhất** của mọi lựa chọn; `proj/*.py` thực thi nó và
`tests/` kiểm nó. Chốt ngày **2026-09-21** với chủ dự án.

---

## 1. Cấu hình đã chốt

| hạng mục | giá trị | nguồn |
|---|---|---|
| Model của mọi device và của server | **DAGSNet 395.024 tham số**, `num_classes=16`, `n_features=66` | `knowledge/architecture.md`; bài báo không quy định kiến trúc |
| Khởi tạo | một `ω^0` duy nhất, seed **42**, dùng cho cả M device lẫn server | chủ dự án |
| Loss | `φ_m = CE(z, y) + λ·KL(p_teacher ‖ p_student)`, softmax nhiệt độ 1 | Eq. (2) + G3 |
| λ | **1,0** | G4 |
| Teacher V_m | trọng số của chính device đó ở cuối vòng trước; **đóng băng**, `eval()`, `no_grad` | Alg.2 dòng 12 + G9 |
| Ngưỡng τ | **τ_t = (1/M) Σ_m a_m** — trung bình accuracy của **mọi** device | Alg.1 dòng 11 (G1) |
| a_m | **accuracy** trên **tập test toàn cục** 10.761.343 dòng | Alg.1 + G8 |
| Chọn device | `S_{t+1} = { m : a_m^t < τ_t }`, so sánh **chặt** | Alg.1 dòng 5–6 |
| Vòng 1 | `S_1 = [M]` (chưa có accuracy nào), `V_m = ω^0` ∀m | G6, G7 |
| Ai train | **mọi** device, mỗi vòng | §IV + Alg.2 dòng 2–7 (G2) |
| Ai upload | **chỉ** device thuộc S_t | §IV "receives … from the selected edge devices" (G2) |
| Tổng hợp | `ω^t = (1/\|S_t\|) Σ_{m∈S_t} ω_m^t`, trung bình đều | Alg.1 dòng 10 |
| Optimizer | **AdamW**, `weight_decay = 1e-4`, `betas` mặc định, **tạo mới mỗi device mỗi vòng** | `knowledge/architecture.md` §4.1 (G5) |
| Learning rate | **cosine theo vòng 1e-3 → 1e-5, T = 50**, hằng trong một vòng (`perfedskd.lr_at`) | chủ dự án (G5) |
| Round × epoch | **50 × 1** | chủ dự án |
| Batch | **512 / 512 / 256** cho 20 / 50 / 100 client | chủ dự án + `knowledge/dataset.md` §4 |
| Clip | grad-norm **1,0** trên student | `knowledge/architecture.md` §4.1 |
| Precision | fp16 AMP + `GradScaler`; loss tính fp32 ngoài autocast; **không bao giờ bf16** (T4 là sm_75) | `knowledge/local-env.md` §3.3 |
| Eval | mỗi vòng, **mọi M** model cá nhân hoá trên **đủ** tập test + **thêm** model tổng hợp ω^t | chủ dự án |
| Metric | đủ **10 metric** cho từng device, cộng mean/std/min/max theo device; cột `global_*` cho ω^t | skill `METRIC_KEYS` |
| Checkpoint | **chỉ trọng số** (`state_dict`), `weights_only=True`; không pickle module, không optimizer | chủ dự án + skill |

**Điều KHÔNG dựng:** teacher phía server trên "predefined dataset" (G10 — không có trong bất
kỳ công thức nào), proximal term (§8.3 của `paper.md`), backbone trích xuất đặc trưng (chưa
từng tồn tại trong bài báo), MNIST/EMNIST, và bốn baseline FedAvg/FedProx/Fed-ensemble/PerFedAvg.

---

## 2. Deviation — phải công bố kèm mọi con số

| # | deviation | vì sao | hệ quả khi đọc kết quả |
|---|---|---|---|
| D1 | τ = trung bình accuracy các device, **không** phải accuracy của model tổng hợp | G1: chỉ bản này được viết thành công thức. Bản văn xuôi trên dữ liệu này sẽ chọn **mọi** device mọi vòng (ở dự án anh em, model tổng hợp đạt f1 0,79 còn mọi model cá nhân hoá ≤ 0,70 trên test toàn cục) → phương pháp suy biến thành FedAvg+SKD và mất hẳn claim tiết kiệm băng thông | Claim "giảm communication overhead" đo được là nhờ định nghĩa này; với định nghĩa kia nó bằng 0 |
| D2 | a_m là **accuracy**, không phải f1_macro | Đúng chữ bài báo | VeReMi mất cân bằng 41:1 → accuracy bị hai lớp lớn (44,5% test) chi phối. Quy tắc chọn device vì thế **không** tối ưu f1_macro. Cả 10 metric vẫn được lưu |
| D3 | a_m đo trên **test toàn cục dùng chung** | G8: phân hoạch này không có test theo client | Điểm số đo **khả năng tổng quát hoá toàn cục** của model cá nhân hoá, không đo hiệu năng trên phân bố cục bộ. Khác bài báo, vốn gọi là "personalized test accuracy" |
| D4 | Mọi device train, chỉ S_t upload | G2: theo văn xuôi §IV + Alg.2, và số chia \|S\| của dòng 10 | Không tiết kiệm **tính toán** cho device không được chọn; chỉ tiết kiệm **truyền tin** |
| D5 | L = KL(p_teacher ‖ p_student), nhiệt độ 1, λ = 1,0 | G3, G4 | Không có quét λ. Một λ khác cho kết quả khác; không có cơ sở nào trong bài báo để nói λ = 1 là đúng |
| D6 | Teacher chạy `eval()` dưới `no_grad` | G9 | Teacher dùng running stats của BatchNorm và tắt dropout → cố định trong cả vòng. Ở `train()` nó sẽ trả lời bằng thống kê batch và một mask dropout ngẫu nhiên, tức là teacher khác nhau ở mỗi batch |
| D7 | AdamW + cosine 1e-3 → 1e-5, tạo mới mỗi device mỗi vòng | G5: bài báo chỉ nói "SGD" và η = 0,01 cho MNIST 20 epoch | Không phải SGD. Không có moment nào đi qua ranh giới vòng — đó cũng là lý do checkpoint không cần state optimizer |
| D8 | 50 vòng × 1 epoch thay vì 200 × 20 | Ngân sách: 200×20 trên 43 triệu dòng là ~4.000 lượt quét dữ liệu | Không thể so số với bài báo |
| D9 | DAGSNet thay cho mạng ảnh; 16 lớp bảng thay cho 10/26 lớp ảnh | Bài báo không quy định kiến trúc | |
| D10 | `ω^0` dùng chung cho mọi device | Bài báo nói device **dị thể** nhưng không mô tả model dị thể nào, và Alg.1 dòng 10 trung bình trọng số — phép này chỉ định nghĩa được khi mọi model cùng hình dạng | "Heterogeneous" trong bản dựng này là **dị thể dữ liệu** (Dirichlet α = 0,5), không phải dị thể kiến trúc |
| D11 | fp16 AMP cho tính toán; đặc trưng thường trú ở fp16 | `knowledge/dataset.md` §1.3 | Đặc trưng đã bị lượng tử hoá về fp16 |
| D12 | Truyền tin được **tính**, không **đo** | Driver giữ bảng trọng số của mọi device trong tiến trình của nó, nên `M` model luôn quay về driver | Cột `comm_*` là chi phí của **giao thức** (\|S_t\| lên + \|S_t\| xuống), không phải lưu lượng của tiến trình |
| D13 | Một seed, một lần chạy | Ngân sách GPU | Bài báo lặp 10 lần và báo ± độ lệch. Bản này không có replication; `*_std` là độ lệch **giữa các device**, không phải giữa các lần chạy |

## 3. Ba điều cần nói trước khi chủ dự án nhìn đường cong W&B

1. **`global_*` ở vòng 1 sẽ rất tệ.** Vòng 1 là FedAvg trên M model vừa train 1 epoch đầy đủ
   từ khởi tạo ngẫu nhiên; trung bình trọng số của các mạng đã phân kỳ như thế gần như vô
   nghĩa. Đo được ở smoke test 4 client: `global f1 = 0,013` ở vòng 1. Từ vòng 2 mọi device
   thuộc S_t đều xuất phát từ **cùng** ω^{t-1}, nên phép trung bình trở lại là FedAvg hợp lệ
   và `global_*` đi lên.
2. **`kd` cao nhất ở vòng 1, không phải thấp nhất.** Teacher vòng 1 là ω^0 ngẫu nhiên: số hạng
   KD bằng **đúng 0 ở bước đầu tiên** (kiểm trong `tests/test_units.py`) rồi phình ra khi
   student rời xa một teacher gần-đồng-đều. Từ vòng 2 teacher là một model đã học và `kd` nhỏ lại.
3. **`|S|` sẽ dao động quanh M/2.** τ là trung bình nên xấp xỉ một nửa số device nằm dưới, trừ
   khi phân bố accuracy lệch mạnh. `comm_saving ≈ 0,5`. Vòng 1 có `|S| = M` và
   `comm_saving = 0` theo định nghĩa.

## 4. Thiết kế để chạy nhanh nhất trên 2×T4 — và cái đã cân nhắc rồi loại

Kế thừa nguyên khuôn đã đo trên 2×T4 (`skill references/per-client-fl-veremi.md`), cộng phần
riêng của phương pháp này.

**Đang dùng:**

| kỹ thuật | vì sao có lời | bằng chứng |
|---|---|---|
| Hai worker thường trú, mỗi worker một GPU, spawn **một lần** cho cả phiên | Toàn bộ train (5,29 GiB fp16) + test (1,32 GiB) nằm thường trú trên **cả hai** GPU ⇒ không có input pipeline, device nào cũng train được ở GPU nào | dự án anh em |
| Không có `DataLoader` | Vòng lặp là `slice + randperm` trên tensor thường trú | `dataset.md` §4 |
| `torch.compile(mode="reduce-overhead")` (CUDA graph) cho student, teacher và template eval | Model launch-bound. Đo trên T4 ở probe 21-09: **52,83 → 7,85 ms/step (6,73×)** ở batch 512 và **55,98 → 7,27 (7,70×)** ở batch 256 — cao hơn mức 2,86× của bước DAGSNet đơn vì bước SKD có ba graph | §5 |
| Cổng compile chạy trên **dữ liệu thật**, so với eager, dropout tắt cả hai phía, quy tắc argmax hàng quyết định | Triton sm_75 là rủi ro đã biết; fallback phải **ồn ào**, không im lặng | `_compile_train`, `_compile_eval` |
| `cudagraph_mark_step_begin()` ở đầu **mỗi** bước | Ba graph chạy trong một bước (teacher fwd, student fwd, student bwd); không khai báo ranh giới iteration thì cudagraph-tree coi lần gọi sau là iteration mới và làm hỏng output của lần trước | bẫy đã gặp ở pFedES |
| `recompile_limit = 64` | Student/teacher/template dùng chung **một** code object `DAGSNet.forward`; train/eval × dropout on/off × no_grad/grad × hai shape vượt hạn mức mặc định 8, và quá hạn Dynamo chạy **eager mà không báo** | bẫy đã gặp |
| Fold BatchNorm vào Conv1d cho eval + **một** template compiled, nạp trọng số bằng `copy_` | Bỏ 31 kernel launch mỗi forward; địa chỉ tham số không đổi nên graph vẫn hợp lệ cho mọi device | `evaluate.fold_bn` |
| Đếm confusion **trên GPU** bằng `bincount(y*C + pred)` | Một `.item()` mỗi batch là ~650 lần đồng bộ mỗi lượt | `evaluate.eval_model` |
| Điều phối LPT (device nhiều dòng nhất đi trước) | Gửi device lớn nhất sau cùng làm một GPU đứng không | |
| Ghim device `c` vào worker `c % W` khi eval | Hai worker là hai tiến trình, cuDNN benchmark chọn thuật toán khác nhau ⇒ vài hàng fp16 lệch; ghim giữ chuỗi số của mỗi device trên một GPU | đo ở pFedES |
| AdamW `fused=True` | Gộp bước cập nhật thành một kernel multi-tensor | |
| `preds` chỉ ghi ở vòng 50 | (M, 10,76 M) uint8 = 1,08 GB ở 100c | |

**Đã cân nhắc và LOẠI, kèm số học:**

* **Fold BatchNorm cho teacher lúc train.** Đúng về mặt toán (teacher ở `eval()`), và bỏ được
  31 kernel launch. Nhưng dưới CUDA graph launch overhead đã gần bằng 0, nên phần lời chỉ nằm
  ở thời gian chạy kernel BN — ước tính ≈ 25% của forward teacher ≈ **2–3% một vòng** ở 20c/50c
  và **~2%** ở 100c. Đổi lại: thêm một template, thêm một cổng compile, thêm một ca test, và
  logits teacher lệch ~1e-7 so với bản không fold (tức là đổi objective một chút). Không xứng.
* **Gộp nhiều model vào một eval bằng grouped convolution.** Eval chiếm 45% (20c) đến **64%**
  (100c) một vòng, nên đây là đòn bẩy lớn nhất còn lại — có thể 2–4×. Nhưng nó viết lại
  `evaluate.py`, cần cổng số học riêng, và ở local 4 GiB không kiểm được (OOM trước khi chạm
  vùng cần đo). Rủi ro cao, không có bản đo nào để dựa vào. Để ngoài bản dựng này; ghi lại làm
  việc mở.
* **Cache confusion matrix của device không đổi trọng số.** Ở pFedES đòn này cắt eval 100c từ
  30 phút xuống 3 phút. **Không dùng được ở đây:** D4 — mọi device train mỗi vòng, nên không
  device nào giữ nguyên trọng số. Đây là cái giá trực tiếp của việc theo văn xuôi §IV.
* **Giảm số dòng test hoặc số vòng eval.** Chủ dự án yêu cầu đủ 10 metric trên đủ tập test mỗi
  vòng, và τ được tính từ chính các con số đó — lấy mẫu sẽ làm quy tắc chọn device nhiễu.

## 5. Ngân sách — **đo trên 2×T4**, probe 2026-09-21

Cell calibration của hai probe (`reports/calibration.json`, cũng có trong W&B summary dưới
tiền tố `cal_`). Cả hai probe báo `backend = compiled/compiled`: cổng compile trên **sm_75**
chứng nhận cả graph student lẫn graph teacher, không rơi về eager.

| | 20c probe (batch 512) | 100c probe (batch 256) |
|---|---:|---:|
| bước SKD **compiled** | **7,85 ms/step** | **7,27 ms/step** |
| bước SKD eager | 52,83 | 55,98 |
| **tăng tốc compile** | **6,73×** | **7,70×** |
| compile + gate (một lần/phiên) | 103 s | 110 s |
| eval compiled-folded @**16384** | **400.053 rows/s** | **406.334 rows/s** |
| eval compiled-folded @32768 | 359.102 | 378.040 |
| eval eager-folded @16384 | 336.740 | 339.810 |

Hai điều số đo này quyết định:

* **`eval_batch = 16384`, không phải 32768.** Ở cả hai batch size, 32768 **chậm hơn** (−10 %
  và −7 %). Notebook giữ 16384 như đã sinh. (Ở dự án anh em, 8192 từng chậm 5× vì graph bị
  ghi lại mỗi batch — 16384 là điểm đã kiểm hai lần.)
* **Tăng tốc compile 6,7–7,7×**, cao hơn hẳn mức 2,9× lịch sử của bước DAGSNet đơn. Lý do:
  bước SKD có **ba** graph (teacher fwd, student fwd, student bwd), nên tỉ lệ thời gian là
  launch overhead lớn hơn và CUDA graph cắt được nhiều hơn. Hệ quả vận hành: **fallback về
  eager làm run chậm gần 7×** — nếu W&B báo `backend = eager` thì dừng, đừng chạy tiếp.

**Ngân sách 50 round suy từ các số trên** (train = steps ÷ 2 GPU × ms/step;
eval = ⌈(M+1)/2⌉ model/GPU × 10.761.343 ÷ rows-per-s):

| | steps/round | train (s) | eval (s) | vòng (phút) | 50 vòng (h) | phiên 11 h |
|---|---:|---:|---:|---:|---:|---|
| 20c | 84.083 | 330 | 282 (21 model) | **10,2** | **8,5** | **1 phiên** |
| 50c | 84.098 | 330 | 685 (51 model) | **16,9** | **14,1** | **2 phiên** |
| 100c | 168.200 | 611 | 1.372 (101 model) | **33,1** | **27,6** | **3 phiên** |

50c không có probe riêng: batch của nó bằng batch của 20c nên `ms/step` dùng chung, và chi phí
eval tỉ lệ thuận với số model. Đây là phép nội suy, không phải số đo — ghi rõ khi báo cáo.

**Hệ quả phân bổ tài khoản.** Ngưỡng đặt ra trước khi probe chạy là "50 round ≤ 28 h thì 100c
nằm gọn trên một tài khoản". **27,6 h < 28 h ⇒ không kịch bản nào cần bàn giao sang tài khoản
khác.** Mỗi kịch bản một tài khoản, phiên nối tiếp dùng `--kernel-source` cùng tài khoản (đường
đơn giản), và không cần handoff bundle nào:

| kịch bản | tài khoản | phiên | giờ |
|---|---|---|---:|
| 20c | `catbaochau` | 1 × 11 h | ~8,7 |
| 50c | `trietbackup` | 11 h + ~4 h | ~14,3 |
| 100c | `minhtriethihi` | 11 h + 11 h + ~6,5 h | ~27,9 |

`khanhmay0304` (≈ 28,4 h sau probe 100c) là dự phòng cho 100c nếu round thật chậm hơn dự phóng.
Dung lượng: một vòng lưu (M+1) × 1,58 MB trọng số ⇒ 20c 33 MB, 50c 81 MB, 100c 160 MB mỗi
vòng; 50 vòng = 1,7 / 4,0 / 8,0 GB, cộng `preds` 1,08 GB ở vòng 50 của 100c.

## 6. Hợp đồng artifact

`runs/<run_name>/` với `run_name = perfedskd_{20,50,100}c`:

| đường dẫn | nội dung | ghi khi nào |
|---|---|---|
| `weights/round_NNN.pt` | `{round, global, clients{0..M-1}, cfg, fingerprint, prev_sha, metrics, global_metrics, selected, selected_next, tau}` — **toàn tensor và kiểu nguyên thuỷ**, `weights_only=True` | mỗi vòng |
| `resume/round_NNN.pt` | `{round, rng, selected_next, note, fingerprint}` | mỗi vòng |
| `metrics/round_NNN.json` | hàng tổng hợp + `clients[]` (10 metric + per-class cho từng device) + `global{}` + `selected`, `selected_next`, `tau` | mỗi vòng |
| `confusion/round_NNN.npy` | `(M, 16, 16)` int64 | mỗi vòng |
| `confusion/global_NNN.npy` | `(16, 16)` int64 | mỗi vòng |
| `logs/round_NNN.json` | mỗi device: steps/applied/skipped/nonfinite/lr/seed/**selected**/sec/vram + 4 giá trị loss | mỗi vòng |
| `preds/round_NNN.u8.npy` | `(M, 10.761.343)` uint8 | chỉ vòng 50 |
| `preds/global_NNN.u8.npy` | `(10.761.343,)` uint8 | chỉ vòng 50 |
| `complete/round_NNN.done` | **marker, ghi sau cùng tuyệt đối** | mỗi vòng |
| `history.csv`, `clients.csv` | **dẫn xuất** từ `metrics/*.json`, keyed by round | mỗi vòng |
| `reports/manifest.json` | fingerprint, cfg, class_names, số dòng từng client, data_id, content_id, torch/cuda | một lần/phiên |
| `reports/y_true.u8.npy` | nhãn thật của tập test, đi theo run | một lần |
| `reports/handoff.json` | chuỗi sha `weights/round_001..r` khi bàn giao sang tài khoản khác | khi hop |

**Ba điều khiến checkpoint này là checkpoint chứ không phải một đống tensor:**
`load_weights` dựng lại model bằng `build_model(cfg)` lấy từ chính file, `load_state_dict(strict=True)`
(bắt được `running_mean`/`running_var` bị lọc mất — nếu không, `eval()` sẽ chuẩn hoá bằng 0/1
mà **không báo lỗi**), và assert đúng 395.024 tham số. Thêm một điều riêng của phương pháp
này: `selected_next` **nằm trong** file trọng số, nên một phiên nối tiếp không phải tự suy lại
tập device được chọn — suy lại sẽ khiến run được resume không còn là run bị ngắt.

`FINGERPRINT_KEYS` gồm cả `lam`, `select_metric` và `rounds`: λ đổi objective, quy tắc chọn
đổi thuật toán, và lịch cosine trải trên cả run nên trọng số ở vòng r phụ thuộc T đã hoạch định.

## 7. Bảng test

| file | kiểm gì | chạy ở đâu |
|---|---|---|
| `tests/test_units.py` | 46 check: số tham số, round-trip flat↔state_dict, fold BN chính xác, `eval_model` phủ cả tail batch, `skd_loss` đối chiếu Eq. (2) viết tay **và** cấu trúc gradient (teacher không nhận gradient), λ chỉ scale số hạng KD, teacher==student ⇒ KD=0, λ=0 ⇒ đúng bằng CE training, teacher **không đổi kể cả buffer BN** sau một `client_update`, `aggregate` chia \|S\| và từ chối S rỗng, `threshold`/`select_clients` đúng Alg.1, lịch LR, 10 metric vs sklearn | CPU, ~30 s |
| `tests/test_smoke_real.py` | dữ liệu VeReMi thật, 4 client, 3 vòng: artifact đủ, **chuỗi chọn device** (S_1 = tất cả; S_t = S_{t+1} của vòng trước; log từng device khớp; τ = trung bình accuracy), crash replay bit-identical, resume phiên sau bit-identical, resume với `rounds` khác bị chặn, phiên không-có-gì-để-làm | CPU, ~7 phút |
| `tests/test_two_workers.py` | 1 worker ≡ 2 worker bit-for-bit | CPU |
| `tests/test_compile_gate.py` | cổng compile bắt được backend sai | CPU |
| `tests/test_gpu_local.py` | Inductor chạy được ở local (sm_86 — **không** chứng minh sm_75) | GPU local |
| `tests/test_ckpt_verify.py` | verifier từ chối các ca giả mạo artifact | CPU |
| `tests/test_notebook_sim.py` | các cell sinh ra chạy được trên cây `/kaggle/input` giả | CPU (+`--gpu` cho cell calibration) |
| `scripts/validate_notebooks.py` | gate tĩnh: metadata, `docker_image` theo digest, module nhúng khớp `proj/`, giá trị CFG, thứ tự cell | CPU |

Mọi test local đi qua `scripts/run_local_checked.py` (watchdog RAM), **một cây một lúc**.
Local là sm_86 một GPU: nó chứng minh **đúng/sai**, không chứng minh **nhanh/chậm**, và không
chứng minh gì về sm_75. Xem `knowledge/local-env.md` §5.

## 8. Caveat bắt buộc kèm mọi con số công bố

Kế thừa `knowledge/dataset.md` §6 và `architecture.md` §8: split theo thời gian mô phỏng
(test chỉ có scenario `_7`); mất cân bằng 41:1 ⇒ **đọc `f1_macro`, không đọc `accuracy`**; rò
rỉ Sybil; `scaler.json` fit trên toàn bộ 43 M dòng train (rò rỉ thống kê toàn cục — trong FL
thật client không có); đặc trưng lượng tử hoá fp16; client = receiver unit, không phải xe thật.

Riêng phương pháp này, cộng thêm: D1–D13 ở §2; test **không** chia theo client nên điểm số là
tổng quát hoá toàn cục chứ không phải hiệu năng cá nhân hoá theo nghĩa của bài báo; và **không
đặt các con số này cạnh Table III–VI của bài báo** — khác dữ liệu, khác số lớp, khác ngân sách
huấn luyện, khác cả định nghĩa tập test.
