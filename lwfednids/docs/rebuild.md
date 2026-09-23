# Lightweight-Fed-NIDS trên VeReMi NextGen / DAGSNet — quyết định, deviation, hợp đồng artifact

Bản dựng lại phương pháp của [`paper.md`](paper.md) cho bộ dữ liệu VeReMi NextGen với bộ phân
loại DAGSNet. File này là **nguồn duy nhất** của mọi lựa chọn; `proj/*.py` thực thi nó và
`tests/` kiểm nó. Chốt ngày **2026-09-22** với chủ dự án (7 câu hỏi, 7 câu trả lời).

---

## 1. Cấu hình đã chốt

| hạng mục | giá trị | nguồn |
|---|---|---|
| Model của mọi client và của server | **DAGSNet** (395.024 tham số chưa cắt), `num_classes=16`, `n_features=66`, đầu vào là 66 đặc trưng bảng **trực tiếp** — **không** có backbone trích xuất đặc trưng | chủ dự án; `knowledge/ARCHITECTURE.md` |
| Khởi tạo θ₀ | mặc định PyTorch (Kaiming uniform a=√5), seed **42**, dựng **một lần** trên server | `knowledge/ARCHITECTURE.md` §3.2; bài báo nói "Kaiming He" (G7) |
| Sparsity | **0,7 duy nhất** — tỉ lệ **kênh** bị cắt ở mỗi lớp (nghĩa của `pruning_ratio` Torch-Pruning) | chủ dự án (câu 2); G2 |
| Mask 𝓜 | **zero-shot** từ θ₀, không dữ liệu; DepGraph gom nhóm; importance **L1 theo nhóm** (`GroupMagnitudeImportance(p=1)`); tỉ lệ **đồng đều mỗi lớp** (`global_pruning=False`); lớp Linear cuối (16 logit) **miễn cắt**; **xoá vật lý** kênh bị cắt | §III-B-1-b, Eq. (7)–(8); G3, G4 |
| Kết quả cắt | **35.891 tham số** (9,09 %), MACs 208.035 / 2.276.808 (9,14 %); concat 4 nhánh 544 → 157 kênh; plan id `524e7ab78439698c` (torch-pruning 1.6.1, đo local, kiểm lại trên Kaggle) | `tests/test_units.py` |
| Loss | `CrossEntropyLoss` 16 lớp trên logit fp32 | Eq. (5)/(6) đa lớp (G8) |
| Ai train / ai upload | **mọi** N client, mỗi vòng; mọi client upload | Alg. 6 dòng 6–8 |
| Tổng hợp | **θ^{t+1} = (1/N) Σⱼ θⱼ^t — trung bình đơn**, không trọng số theo Nⱼ | Eq. (9); chủ dự án (câu 5) |
| Mask theo vòng | **đồng nhất**: kênh đã xoá không mọc lại (Alg. 5 dòng 5 là identity) | G6 |
| Optimizer | **AdamW**, `weight_decay = 1e-4`, betas mặc định, **tạo mới mỗi client mỗi vòng** | chủ dự án (câu 4); `knowledge/ARCHITECTURE.md` §4.1 |
| Learning rate | **cosine theo vòng 1e-3 → 1e-5, T = 50**, hằng trong vòng (`lwfednids.lr_at`) | chủ dự án (câu 4) — chính sách chung các dự án anh em |
| Round × epoch | **50 × 1** | chủ dự án |
| Batch | **512 / 512 / 256** cho 20 / 50 / 100 client | chủ dự án; `knowledge/DATASET.md` §4 |
| Clip | grad-norm **1,0** | `knowledge/ARCHITECTURE.md` §4.1 |
| Precision | fp16 AMP + `GradScaler`; loss fp32 ngoài autocast; **không bao giờ bf16** (T4 = sm_75) | `knowledge/LOCAL_ENV.md` §3.3 |
| Eval | mỗi vòng, **mô hình tổng hợp θ^t** trên **đủ** 10.761.343 dòng test (chia hàng cho 2 GPU, cộng ma trận nhầm lẫn) | chủ dự án (câu 3a) |
| Metric | đủ **10 metric** của θ^t; cộng **mean/std/min/max theo client** của loss và grad-norm huấn luyện; `steps/applied/skipped`; `train_sec`, `eval_sec`; `model_mib`, `comm_mib` | skill `METRIC_KEYS`; chủ dự án |
| Checkpoint | **chỉ trọng số** (`state_dict` của θ^t) + `cfg` chứa **plan cắt** ⇒ dựng lại được **không cần torch-pruning**; `weights_only=True`; không pickle module, không optimizer | chủ dự án (câu 6) + skill |

**Điều KHÔNG dựng:** backbone ResNet/VGG trên ảnh flow (bài báo) — dữ liệu ta là bảng; các mức
sparsity 0 / 0,5 / 0,9 (chỉ đo **TA/IA so với mô hình chưa cắt** trong cell calibration của
probe, không train đủ 50 vòng mô hình chưa cắt); FedProx (Table 11); pipeline NFStream.

---

## 2. Deviation — phải công bố kèm mọi con số

| # | deviation | vì sao | hệ quả khi đọc kết quả |
|---|---|---|---|
| D1 | **Không có feature extractor**; DAGSNet nhận thẳng 66 đặc trưng | Chủ dự án (câu 1 của đề bài); dữ liệu VeReMi là bảng, không phải byte gói tin | Mask cắt tỉa tác động lên **bộ phân loại**, không lên backbone như bài báo. Mô hình gốc 395 k tham số là **launch-bound** trên T4 (6 ms/bước bất kể batch), nên tăng tốc TA/IA của bài báo (×2–3 trên ResNet 25 M tham số) **không** được kỳ vọng — TA/IA thật được đo trong probe |
| D2 | Sparsity = **tỉ lệ kênh** mỗi lớp (Torch-Pruning) | G2; đó là nghĩa của công cụ bài báo dùng | Tham số giảm mạnh hơn tỉ lệ kênh: 70 % kênh ⇒ **90,9 % tham số** bị xoá (mỗi lớp trong mất cả hàng lẫn cột). Ai đọc "sparsity 70 %" là "còn 30 % tham số" sẽ sai |
| D3 | AdamW + cosine theo vòng, tạo mới mỗi client mỗi vòng | G1: bài báo không nêu η, optimizer; Alg. 5 viết GD thuần | Không phải SGD. Không có moment nào qua ranh giới vòng ⇒ checkpoint không cần state optimizer |
| D4 | Tổng hợp **1/N đơn** trên phân hoạch **lệch** (870 k – 5,9 M dòng/client) | Đúng chữ Eq. (9); bài báo IID chia đều nên hai công thức trùng | Client nhỏ có ảnh hưởng **bằng** client lớn. Trên α = 0,5 điều này khác FedAvg gốc (trọng số Nⱼ/N); không có ablation |
| D5 | Chỉ **một** mức sparsity (0,7) | Ngân sách GPU; chủ dự án (câu 2) | Không có đường cong sparsity ↔ hiệu năng; baseline chưa cắt chỉ có **tốc độ** (probe), không có **điểm số** |
| D6 | 50 vòng × 1 epoch thay vì T = 5 | Chính sách chung các bản dựng; 43 M dòng | Không so số vòng với bài báo |
| D7 | Phân hoạch **non-IID** Dirichlet α = 0,5 thay vì IID | Bộ dữ liệu cố định của dự án | Bài toán khó hơn bài báo; f1_macro là số đáng đọc (mất cân bằng 41:1) |
| D8 | Mask **zero-shot trên trọng số ngẫu nhiên** = chọn kênh gần như ngẫu nhiên (tất định theo seed) | Đúng phương pháp bài báo (Cai et al.) | Kết quả phụ thuộc seed nhiều hơn một mask học từ dữ liệu; chỉ một seed |
| D9 | Lớp Linear cuối miễn cắt; các lớp Conv/Linear khác cắt đúng `int(n·0,3)` kênh; concat 544 → 157 theo tổng các nhánh | G4 | |
| D10 | fp16 AMP; đặc trưng thường trú fp16 | `knowledge/DATASET.md` §1.3 | Đặc trưng đã bị lượng tử hoá về fp16 |
| D11 | Truyền tin được **tính**, không **đo**: `comm_mib = 2·N·model_mib` | Driver giữ θ^t trong tiến trình | Là chi phí **giao thức**; mask được gửi **một lần** và không tính vào đây |
| D12 | Một seed, một lần chạy | Ngân sách GPU | Không có ± giữa các lần chạy; `*_client_std` là độ lệch **giữa các client** |
| D13 | Kênh cắt ra số lẻ (28, 9, 14, 38, 76, 157) | Tỉ lệ đúng 0,7 không làm tròn bội 8 | Tensor core fp16 thích bội 8; mô hình vốn launch-bound nên ảnh hưởng nhỏ; `round_to` của Torch-Pruning **không** dùng để trung thành với bài báo |

## 3. Những gì sẽ trông "sai" trên W&B mà thật ra là đúng

1. **Vòng 1 thấp**: mọi client xuất phát từ cùng θ' và train 1 epoch; FedAvg vòng 1 là trung
   bình của N mô hình vừa đi 1 epoch từ cùng điểm — hợp lệ, nhưng chưa hội tụ.
2. **`loss_client_std` lớn**: phân hoạch α = 0,5, client có phân bố lớp khác nhau, loss cục bộ
   khác nhau là bình thường. `loss_client_max` thuộc client có lớp khó (timeDelayAttack…).
3. **Mô hình 35,9 k tham số có thể đạt f1_macro thấp hơn đáng kể so với DAGSNet đầy đủ** (0,85
   ở bản centralized). Đó chính là câu hỏi bài báo đặt ra ở sparsity 70 %; không phải lỗi.
4. **`skipped` ≈ 1–2 bước/client mỗi vòng**: GradScaler khởi động ở 2¹⁶ và tự hiệu chỉnh;
   `max_skips_per_client = 16` là trần.

**Lý do thật sự phải dừng**: [`KAGGLE.md` §6](KAGGLE.md).

## 4. Thiết kế để chạy nhanh nhất trên 2×T4

Kế thừa khuôn đã đo trên 2×T4 ở các bản dựng anh em (`skill references/fl-veremi-dagsnet.md`),
rút gọn về **một** mô hình toàn cục:

| kỹ thuật | vì sao có lời |
|---|---|
| Prepack parquet → fp16 thường trú trên **cả hai** GPU (train 5,29 GiB + test 1,32 GiB) | bỏ hẳn DataLoader; mỗi bước chỉ là `randperm` + slice |
| 1 worker/GPU sống suốt phiên, dispatch client **dài nhất trước** (LPT) | không có tail idle; không spawn lại |
| `torch.compile(mode="reduce-overhead")` + CUDA graphs cho bước train và cho template eval | mô hình launch-bound: đo trên T4 22-09 **59,3 → 5,85 ms/bước (10×)** cho mô hình cắt; cổng compile so với eager trên hàng thật, rơi về eager **kêu to** |
| Fold BatchNorm vào Conv trước eval (exact) | bớt 31 kernel/forward |
| **Eval một mô hình chia hàng cho 2 GPU** (`driver.eval_bounds`), cộng ma trận nhầm lẫn | eval đo được **4 s/vòng** (mô hình cắt, 2 GPU) thay vì 7,3 s một GPU |
| Optimizer `fused=True`, loss fp32 ngoài autocast, `cudagraph_mark_step_begin` mỗi bước | |
| Trọng số qua queue dạng numpy phẳng (1 vector float + 1 vector int) | tránh 192 fd shared-memory mỗi lần |

**Đã cân nhắc rồi loại:** `round_to=8` cho kênh (D13 — lệch bài báo); eval từng client (câu 3b —
mọi client giữ cùng θ^t sau broadcast nên vô nghĩa và tốn N×13 s/vòng); gộp nhiều mức sparsity
vào một phiên (không cần vì chỉ một mức).

## 5. Ngân sách — dự báo trước probe; **số đo thật ở [`TESTS.md` §4](TESTS.md): 20c 255 s/round ⇒ 3,5 h**

Số bước/vòng (`knowledge/DATASET.md` §4): 20c 84.083 · 50c 84.098 · 100c 168.200. Với bước
compiled ≈ 5–6 ms (mô hình cắt, launch-bound), chia 2 GPU, cộng eval ~13 s và commit:

| kịch bản | train/vòng | vòng | 50 vòng | phiên 11,75 h |
|---|---|---|---|---|
| 20c (b=512) | ~4–5 phút | ~5 phút | **~4–5 h** | 1 |
| 50c (b=512) | ~4–5 phút | ~5 phút | **~4–5 h** | 1 |
| 100c (b=256) | ~8–9 phút (+20 % CPU-starved trên 4 vCPU) | ~10 phút | **~8–9 h** | 1 (có thể 2) |

Driver dừng khi `elapsed + 1,15·worst_round + 900 s > max_seconds`; phiên nối tiếp đi qua
`--require-resume` (kiểm ở `tests/test_smoke_real.py`).

## 6. Hợp đồng artifact (`/kaggle/working/runs/<run_name>/`)

| đường dẫn | nội dung | ghi bởi |
|---|---|---|
| `weights/round_NNN.pt` | `{round, global: state_dict, cfg (có prune_plan, plan_id, n_params), fingerprint, prev_sha, metrics}` — **tensor + kiểu thuần**, `weights_only=True` | `ckpt.save_round_weights` |
| `resume/round_NNN.pt` | `{round, rng, note, fingerprint}` — RNG driver (forensics; resume không cần) | ↑ |
| `confusion/round_NNN.npy` | ma trận 16×16 int64 của θ^t trên đủ test | driver |
| `preds/round_NNN.u8.npy` | dự đoán (10.761.343,) uint8, chỉ vòng trong `preds_rounds` (= [50]) | driver |
| `metrics/round_NNN.json` | hàng history + `per_class` + `clients` (thống kê train từng client) | driver |
| `logs/round_NNN.json` | `{round, clients}` — bản gốc của khối per-client | driver |
| `complete/round_NNN.done` | marker, **ghi cuối cùng** | `ckpt.mark_complete` |
| `history.csv` · `clients.csv` | **dẫn xuất** từ metrics json (một hàng/vòng · một hàng/client/vòng) | `ckpt.append_history` / `rebuild_history` |
| `reports/manifest.json` | cfg, fingerprint, data_id, content_id, client_rows, `prune` (bảng lớp), scaler, torch | `driver.write_manifest` |
| `reports/prune_plan.json` | plan (chỉ số kênh giữ) + plan_id + n_params | notebook |
| `reports/y_true.u8.npy` | nhãn test, để verify offline không cần dataset | ↑ |
| `reports/calibration.json` | chỉ probe: ms/bước, rows/s, TA/IA, dự báo vòng | cell calibration |

**Fingerprint** (`ckpt.FINGERPRINT_KEYS`, 24 khoá): kiến trúc + `sparsity`, `plan_id`,
`prune_importance` + lr/schedule/lr_min/rounds/wd/clip + n_clients/batch/local_epochs/seed +
`data_id`, `run_name`. Đổi bất kỳ khoá nào ⇒ không resume được (kiểm trong
`tests/test_ckpt_verify.py`, 16 khoá khoa học di chuyển, 6 khoá vận hành không).

**Resume**: `working/` trước, rồi input đính kèm (kernel output cùng tài khoản, hoặc dataset
checkpoint khác tài khoản), import qua staging, `prev_sha` xích các vòng, bundle handoff cho
phiên 3+ (`stage_ckpt_dataset.py --last-only`).

## 7. Caveat bắt buộc kèm mọi con số công bố

Kế thừa `knowledge/DATASET.md` §6 (split theo thời gian, rò rỉ Sybil, scaler fit toàn cục, test
không chia theo client) cộng D1–D13 ở trên. Đặc biệt:

* **f1_macro** là số đáng đọc, accuracy bị hai lớp lớn chi phối (44,5 % test).
* Điểm số là của **mô hình toàn cục** đã cắt 70 % kênh; không có điểm của mô hình chưa cắt
  trong cùng giao thức — chỉ có tốc độ của nó (probe).
* TA/IA đo trên **T4 với torch.compile**; bài báo đo trên phần cứng/khung khác; chỉ so hình dạng.
