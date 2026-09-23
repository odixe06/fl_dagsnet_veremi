# TESTS.md — cách kiểm và mọi số đã đo

Mọi con số ở đây **đo thật** và ghi kèm ngày, môi trường. Tách hai loại bằng chứng: **local**
(RTX 3050 sm_86, 4 GiB, WSL 7,6 GiB RAM — chứng minh đúng/sai) và **2×T4 sm_75** (Kaggle —
chứng minh tốc độ và compile). Cái trước không chứng minh cái sau (`knowledge/LOCAL_ENV.md` §5).

---

## 1. Cách chạy

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
# MỌI test local đi qua watchdog RAM (khoá chống chạy chồng; dừng khi RSS cây > trần)
conda run --no-capture-output -n nckh python scripts/run_local_checked.py python tests/test_units.py
conda run --no-capture-output -n nckh python scripts/run_local_checked.py python tests/test_smoke_real.py
conda run --no-capture-output -n nckh python scripts/run_local_checked.py python tests/test_two_workers.py
conda run --no-capture-output -n nckh python scripts/run_local_checked.py python tests/test_compile_gate.py
conda run --no-capture-output -n nckh python scripts/run_local_checked.py python tests/test_ckpt_verify.py
conda run --no-capture-output -n nckh python scripts/run_local_checked.py python tests/test_gpu_local.py
conda run --no-capture-output -n nckh python scripts/run_local_checked.py python tests/test_notebook_sim.py        # CPU: train + verify
conda run --no-capture-output -n nckh python scripts/run_local_checked.py python tests/test_notebook_sim.py --gpu  # calibration cell
python scripts/validate_notebooks.py        # gate tĩnh, không cần watchdog
```

Cần `torch-pruning==1.6.1` trong env `nckh` (cài `--no-deps` ngày 22-09; báo `__version__`
là 1.6.0). Watchdog cần ≥ ~3,5 GiB `MemAvailable`; chạy **một** cây test một lúc.

## 2. Bảy cây test — kiểm gì, và vì sao cần

| file | kiểm | vì sao cần |
|---|---|---|
| `test_units.py` | 48 check: 395.024 → 35.891 tham số; plan tất định, tái lập đúng torch-pruning từng tensor, sống qua JSON, `apply_plan` từ chối model lệch / chỉ số ngoài / thiếu module; θ' = M ⊙ θ₀ từng kênh; cắt đồng đều `int(n·0,3)`; concat 544 → 157; flat round-trip; fold BN exact **trên model đã cắt**; eval chia shard = eval nguyên; CE; `client_update` ≡ AdamW+clip thuần; `aggregate` 1/N từ chối thiếu client/không sắp/sai N; lịch cosine; 10 metric vs sklearn | mỗi công thức của phương pháp có một check trực tiếp |
| `test_smoke_real.py` | 4 client thật (100c) × 3 vòng, CPU: mọi artifact, thống kê client, replay bit-identical sau mất marker, resume phiên sau bit-identical, cổng từ chối đổi `rounds` **và** đổi sparsity/plan, nothing-to-do, require_resume rỗng, baseline sparsity 0 qua cùng driver; rebuild checkpoint **khi torch-pruning không import được** | hợp đồng commit/resume và "checkpoint chỉ là trọng số" |
| `test_two_workers.py` | 1 worker ≡ 2 worker (CPU) bit-for-bit: trọng số, ma trận nhầm lẫn, preds, thống kê | eval chia shard + dispatch LPT không đổi kết quả |
| `test_compile_gate.py` | cổng compile chứng nhận compiler đúng có RNG riêng, từ chối compiler sai, rơi về eager khi nổ, khôi phục dropout/trainable — trên model đã cắt | cổng là thứ quyết định backend production |
| `test_ckpt_verify.py` | 33 check: schema file trọng số (plan trong cfg, hình dạng đã cắt), từ chối `n_params` sai / plan tráo; fingerprint 16 khoá khoa học di chuyển, 6 khoá vận hành không; **24 ca tamper** verifier phải fail; import read-only; từ chối ghép hai lịch sử; bundle handoff + chain; merge 2 phiên ≡ 1 phiên | verifier chỉ có giá trị nếu bắt được giả mạo |
| `test_gpu_local.py` | Inductor thật sm_86: cổng train + eval pass trên model cắt; compiled ≈ eager (dropout 0); eval shard compiled = nguyên; TA local pruned/unpruned | sơ đồ CUDA graph hợp lệ (không phải bằng chứng sm_75) |
| `test_notebook_sim.py` | chạy đúng các cell của notebook probe trên cây `/kaggle` giả (parquet thật + sidecar `.stats.json`): CPU = train 2 vòng + verify; `--gpu` = tới hết cell calibration | luồng notebook (pip, plan, gate resume, prepack, manifest, prune_plan.json) |

## 3. Kết quả đo ở local — 2026-09-22, sm_86, 1 GPU, fixture nhỏ

### 3.1 `test_units.py` — 48/48 pass (RSS đỉnh 917 MiB)
* sparsity 0,7: **35.891** tham số (9,086 %), MACs 208.035 / 2.276.808 (9,137 %), plan
  `524e7ab78439698c`; `compute_plan` 0,1 s; plan JSON 11,2 KB.
* fold BN trên model cắt: max|Δlogit| = **5,96e-08**. Flat round-trip: 0. 10 metric vs sklearn: 5,6e-17.
* stem 0 giữ 28/96 kênh; L1 trung bình kênh giữ 1,453 > kênh bỏ 1,165.

### 3.2 `test_smoke_real.py` — pass (RSS đỉnh 2.046 MiB)
* 4 client × 12 k dòng, test 30 k dòng, 3 vòng: replay bit-identical, resume bit-identical,
  cổng chặn `rounds=4` và `sparsity=0.5`; file trọng số **0,21 MB** (model 0,141 MiB trên dây).
* Rebuild từ file với `sys.modules["torch_pruning"] = None`: OK.
* Baseline sparsity 0 (395.024 tham số) 1 vòng qua cùng driver: verify pass.

### 3.3 `test_two_workers.py` — pass (RSS 2.360 MiB): max|Δw| = 0,0; CM + preds giống hệt.
### 3.4 `test_compile_gate.py` — 7/7 pass.
### 3.5 `test_ckpt_verify.py` — 33/33 pass (24 ca tamper đều bị bắt).
### 3.6 `test_gpu_local.py` — pass (sm_86, batch 256, RSS 2.621 MiB)
* Cổng train + eval chứng nhận với Inductor thật trên model cắt.
* Compiled vs eager (dropout 0): max|Δw| 3,1e-03 (tương đối 4,4e-03), loss lệch < 0,1 %;
  **9,31 vs 19,95 ms/bước → 2,14×**.
* Eval compiled 605 k rows/s vs eager-folded 500 k (1,21×); 0/12.000 dự đoán khác; shard = nguyên.
* **Bước compiled: unpruned 7,68 ms vs pruned 7,06 ms → TA 1,09×** trên sm_86 (launch-bound,
  như dự báo D1). Số T4 ở §4.
### 3.7 `test_notebook_sim.py` — pass cả CPU (2 vòng + verify) và `--gpu` (tới calibration).

## 4. Kết quả đo trên 2×T4 (sm_75) — probe `lwfednids_20c_probe`, 22-09

Kernel `minhtran0601/lightweight-fed-nids-veremi-20-clients-probe` v1, push 07:32Z, W&B run
`21522798-uit/lwfednids-veremi/lwfednids_20c_probe`. Image `knowledge/runtime.json`,
torch-pruning 1.6.1 cài `--no-deps` trong notebook. **Backend: train `compiled`, eval `compiled`
trên cả hai rank** (cổng compile chứng nhận mô hình đã cắt trên sm_75 — điều local không thể).

### 4.1 Cell calibration (GPU 0, batch 512, client nhỏ nhất, 400 bước; eval đủ 10.761.343 dòng)

| | mô hình cắt (35.891) | chưa cắt (395.024) | TA / IA |
|---|---:|---:|---:|
| train eager, ms/bước | 59,29 | 30,94 | 0,52× |
| train **compiled**, ms/bước | **5,854** | 6,783 | **1,16×** |
| compile + gate, s | 95,1 | 86,8 | |
| eval eager-folded @16384, rows/s | 1.031.642 | 293.139 | 3,52× |
| eval **compiled**-folded @16384, rows/s | **1.468.885** | 397.276 | **3,70×** |
| eval compiled-folded @32768, rows/s | 1.519.366 | 384.660 | 3,95× |

Đọc: **train launch-bound** — cắt 91 % tham số chỉ tăng tốc bước compiled 1,16× (đúng dự báo
D1); eager của mô hình cắt còn **chậm hơn** mô hình đầy đủ (kênh lẻ 9/14/28/38 không phải bội 8
→ cuDNN chọn kernel kém) — production luôn chạy compiled nên không ảnh hưởng, nhưng là lý do
phải giữ compile. **Eval compute-bound ở batch lớn** → cắt tỉa cho **3,7×** đúng tinh thần IA
của bài báo. Eval một mô hình đủ test: 7,3 s một GPU, **3,7 s khi chia 2 GPU**.

Dự báo từ calibration: train 84.083 bước × 5,854 ms / 2 GPU = **246 s** + eval 3,7 s ≈
**250 s/round → 50 round ≈ 3,5 h** (chưa tính commit, LPT tail, startup ~12 phút).

### 4.2 Round thật của probe (COMPLETE 08:01Z; output kéo về `runs/pulls/20c_probe`, `verify_run --require-rounds 2` **pass**)

| | round 1 | round 2 |
|---|---:|---:|
| lr | 1e-3 | 1e-5 (cosine trên T = 2 của probe) |
| f1_macro / accuracy / f1_weighted | 0,0843 / 0,2627 / 0,2223 | **0,5827 / 0,7683 / 0,7243** |
| precision_macro / recall_macro | 0,3346 / 0,1165 | 0,6689 / 0,5707 |
| loss client mean ± std · gnorm mean | 0,3158 ± 0,1121 · 1,311 | 0,6167 ± 0,2628 · 1,419 |
| steps / skipped | 84.083 / 36 | 84.083 / 23 |
| train_sec / eval_sec / seconds | 319,4 / 5,5 / **325,0** | 250,4 / 4,1 / **254,6** |
| VRAM train / eval (GiB/GPU) | 6,94 / 6,97 | 6,94 / 6,73 |
| model_mib | 0,1406 | |

Mốc thời gian trong kernel (giây từ đầu phiên): GPU assert 23 s · pip torch-pruning 39 s · plan
40 s · prepack xong 190 s (143 s) · calibration 190 → 652 s · 2 worker ready **717 s** (12,0 phút
kể cả calibration ⇒ production ≈ 4–5 phút) · round 1 xong 1.064 s · round 2 xong 1.319 s.
Cổng compile production: rank0/rank1 max|Δlogit| 2,4e-04 / 1,8e-04, 0 flip decisive; eval
9/16.384 flip không decisive.

**Plan trên Kaggle (torch 2.10.0+cu128, torch-pruning 1.6.1) = `524e7ab78439698c`, đúng bằng
local (torch 2.13)** — mask tất định qua phiên bản torch. `data_id 29f492a531052d2b`,
`content_id 3aa70a5c51aacd51` (giá trị để đối chiếu với production 20c).

**Ngân sách suy ra:** 20c/50c ≈ 255 s/round × 50 = **3,5 h** + ~5 phút startup ⇒ 1 phiên.
100c (batch 256, 168.200 bước, +~20 % CPU-starved) ≈ 500–600 s/round ⇒ **7–8,5 h** ⇒ 1 phiên.

### 4.3 Round đầu của production (đọc W&B qua `poll_prod.sh`, 22-09 08:18–08:35Z; đang chạy)

| kịch bản | round | f1_macro | accuracy | loss client mean | skipped | train_sec | eval_sec | seconds | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20c (`minhtran0601`) | 1 | 0,0969 | 0,2941 | 0,3158 | 34 | 362,7 | 5,9 | 368,6 | 6,94 |
| | 2 | **0,6502** | 0,7636 | 0,2399 | 34 | 293,1 | 4,1 | **297,3** | 6,94 |
| 50c (`catbaochau`) | 1 | 0,1807 | 0,4897 | 0,3907 | 15 | 402,3 | 5,6 | 408,0 | 6,80 |
| | 2 | **0,6037** | 0,6982 | 0,2670 | 21 | 273,9 | 4,1 | **278,1** | 6,80 |
| 100c (`trietbackup`) | 1 | 0,1933 | 0,5314 | 0,3629 | 40 | 809,2 | 6,1 | 815,3 | 6,74 |
| | 2 | **0,5714** | 0,6622 | 0,2358 | 38 | 646,0 | 4,7 | **650,7** | 6,74 |

Đọc lại 08:45Z: 20c round 6 f1_macro **0,7735** (287,6 s/round), 50c round 7 **0,7308**
(236,1 s/round), 100c round 2 **0,5714** (650,7 s/round ⇒ 50 round ≈ **9,1 h**, một phiên).

Cả ba: backend `compiled/compiled` lúc 08:08Z (startup 5–6 phút). Round 1 luôn chậm hơn round 2
(warm-up CUDA graph theo client: probe 325 → 255 s). Dự báo: 20c ≈ 50 × 297 s ≈ **4,1 h**, 50c ≈
**3,9 h**; 100c ổn định ở ~650 s ⇒ ≈ **9,1 h** (một phiên; nếu trôi quá 815 s/round thì
driver tự dừng trước hạn và phiên 2 đi qua `--require-resume`, `KAGGLE.md` §5). Điểm round 2 của production cao hơn probe (0,650 vs 0,583) vì probe có
T = 2 ⇒ lr round 2 = 1e-5; production T = 50 ⇒ lr ≈ 1e-3.

**Kết thúc (23-09):** cả ba run COMPLETE 50/50 trong một phiên, backend compiled suốt 50 round,
`verify_run --require-rounds 50` pass. f1_macro round 50: 20c **0,7917** · 50c **0,7526** · 100c
**0,7557**; round trung vị 253,8 / 238,9 / 485,1 s; Σ 3,62 / 3,38 / 6,90 h. Bảng đầy đủ:
[`report.md`](report.md).

## 5. Ranh giới — cái gì các số trên **không** chứng minh

* Local pass ⇒ thuật toán, artifact, resume đúng; **không** ⇒ compile chạy trên sm_75, thời
  gian vòng, VRAM 2 GPU, hay ngân sách phiên.
* Fixture 4 client × 12 k dòng ⇒ f1 vô nghĩa; chỉ là kiểm pipeline.
* TA/IA trên sm_86 với batch 256 fixture ⇒ hình dạng "launch-bound", không phải con số công bố.
