# CONTEXT — Lightweight-Fed-NIDS trên VeReMi NextGen / DAGSNet

**File này là trung tâm điều hướng.** Đọc nó để nắm dự án trong 2 phút và biết mở đúng file nào
khi muốn sâu hơn. Nó **không** chứa bảng số liệu — chúng ở [`docs/`](docs/).

Cập nhật: **2026-09-23 01:30Z**.

---

## Dự án này là gì

Dựng lại phương pháp **Lightweight-Fed-NIDS** (Bouayad, Alami, Janati Idrissi, Berrada — *Lightweight
Federated Learning for Efficient Network Intrusion Detection*, IEEE Access 2024) trên bộ dữ liệu
**VeReMi NextGen** (16 lớp, 66 đặc trưng, 43.045.415 dòng train / 10.761.343 dòng test), với bộ
phân loại **DAGSNet** làm mô hình của mọi client và của server, **không** có backbone trích xuất
đặc trưng (66 đặc trưng vào thẳng). Phương pháp = **mask cắt tỉa có cấu trúc, zero-shot, tính một
lần trên server** (DepGraph + L1 theo nhóm, Torch-Pruning) + **FedAvg trung bình đơn 1/N**.

Ba kịch bản: **20, 50, 100 client** (Dirichlet α = 0,5), mỗi kịch bản 50 round × 1 epoch, **sparsity
0,7** (395.024 → **35.891** tham số), trên **2 × Tesla T4** của Kaggle, mỗi kịch bản một tài khoản.

**Trạng thái (23-09 ~01:30Z): XONG.** Ba run production **COMPLETE 50/50 round trong một phiên**
mỗi kịch bản, output kéo về `runs/pulls/{20,50,100}c`, `verify_run --require-rounds 50` **pass** cả
ba. f1_macro round 50 (cũng là đỉnh): **20c 0,7917 · 50c 0,7526 · 100c 0,7557**; Σ thời gian
3,62 / 3,38 / 6,90 h. Kết quả, phân tích, per-class, TA/IA: [`docs/report.md`](docs/report.md).
Việc còn mở: [§5](#5-việc-tiếp-theo).

---

## 1. Điều hướng — muốn biết gì thì mở file nào

| muốn biết | mở |
|---|---|
| **Bài báo nói gì**, và **để trống** ở đâu (G1–G8) | [`docs/paper.md`](docs/paper.md) |
| **Mọi lựa chọn của bản dựng**: cấu hình chốt, 13 deviation, thiết kế tốc độ, ngân sách, hợp đồng artifact, caveat | [`docs/rebuild.md`](docs/rebuild.md) |
| **Mọi con số đã đo**: 7 cây test kiểm gì, kết quả từng check, calibration 2×T4 | [`docs/tests.md`](docs/tests.md) |
| **Kết quả cuối**: 10 metric × 50 round × 3 kịch bản, per-class, TA/IA, phân tích | [`docs/report.md`](docs/report.md) |
| **Chạy trên Kaggle**: tài khoản, quota, dataset, kế hoạch phiên, runbook push/theo dõi/kéo/verify/hop, bẫy | [`docs/kaggle.md`](docs/kaggle.md) |
| **DAGSNet**: kiến trúc, Eq. (38)–(48), hợp đồng vào/ra, 66 cột | [`knowledge/architecture.md`](knowledge/architecture.md) |
| **Dữ liệu**: train đã z-score / test chưa, nhãn, ba kịch bản α = 0,5, steps/round | [`knowledge/dataset.md`](knowledge/dataset.md) |
| **Máy local**: RAM 7,6 GiB, VRAM 4 GiB, sm_86 ≠ sm_75, cái gì kiểm được ở local | [`knowledge/local-env.md`](knowledge/local-env.md) |
| **Bài báo gốc** | [`Lightweight_FL.md`](Lightweight_FL.md) |

**Ranh giới tài liệu — giữ đúng, vì một con số đặt sai chỗ sẽ bị phiên sau đọc như sự thật đã kiểm:**

| nơi | chứa | đừng bỏ vào đây |
|---|---|---|
| [`knowledge/`](knowledge/) | sự thật **không đổi** của dữ liệu, máy, kiến trúc — dùng chung nhiều phương pháp | bất kỳ con số của **một** phương pháp |
| [`docs/`](docs/) | tài liệu của **dự án này**: phương pháp, quyết định, số đo, vận hành | — |
| `CONTEXT.md` | điều hướng + trạng thái + những gì phải biết ngay | bảng số liệu dài |
| `../.claude/skills/` (gốc repo) | skill **dùng chung mọi dự án** (đã dọn sạch thông tin các bài báo cũ ngày 22-09) | ❌ bất kỳ thứ gì riêng của dự án này |

## 2. Mã nguồn và công cụ — file nào làm gì

**Vòng đời chuẩn:** sửa `proj/*.py` → chạy test → `gen_notebook.py` → `validate_notebooks.py`
→ nhúng key W&B → **validate lại** → push. ❌ Không bao giờ sửa `.ipynb` bằng tay.

| file | vai trò |
|---|---|
| [`proj/prune.py`](papers/lwfednids-bouayad-2024/proj/prune.py) | **mask**: `compute_plan` (Torch-Pruning, zero-shot, L1 nhóm, đồng đều mỗi lớp) → plan chỉ số kênh giữ; `apply_plan` dựng lại mô hình cắt **chỉ bằng torch**; hai đường được assert bằng nhau từng tensor |
| [`proj/lwfednids.py`](papers/lwfednids-bouayad-2024/proj/lwfednids.py) | **thuật toán**: `client_update` (CE, AdamW, clip, AMP skip accounting), `aggregate` (1/N, từ chối thiếu client), `lr_at`, flat layout |
| [`proj/model.py`](papers/lwfednids-bouayad-2024/proj/model.py) | DAGSNet; `build_full` (θ₀) và `build_model(cfg)` (θ₀ cắt theo plan trong cfg) |
| [`proj/driver.py`](papers/lwfednids-bouayad-2024/proj/driver.py) | 2 worker / 2 GPU, cổng compile, eval **chia hàng** 2 GPU, commit atomic, gate ngân sách phiên, manifest |
| [`proj/ckpt.py`](papers/lwfednids-bouayad-2024/proj/ckpt.py) | file trọng số (θ^t + cfg có plan), resume, marker, fingerprint 24 khoá, import/handoff |
| [`proj/verify.py`](papers/lwfednids-bouayad-2024/proj/verify.py) | dựng lại **mọi** con số từ artifact; rebuild mô hình từ file |
| [`proj/evaluate.py`](papers/lwfednids-bouayad-2024/proj/evaluate.py) · [`data.py`](papers/lwfednids-bouayad-2024/proj/data.py) · [`metrics.py`](papers/lwfednids-bouayad-2024/proj/metrics.py) | fold BN + eval compiled · parquet → fp16 thường trú · 10 metric từ confusion |
| [`scripts/gen_notebook.py`](scripts/gen_notebook.py) | sinh notebook; `SCENARIOS` (batch, dataset), `PAPER` (siêu tham số), cell calibration (TA/IA) |
| [`scripts/validate_notebooks.py`](scripts/validate_notebooks.py) | gate tĩnh (chặn cài lại torch, CFG, thứ tự plan → resume → decode, module nhúng khớp `proj/`) |
| [`scripts/run_local_checked.py`](scripts/run_local_checked.py) | **watchdog RAM — mọi test local phải đi qua đây** |
| [`scripts/verify_run.py`](scripts/verify_run.py) · [`pull_output.sh`](scripts/pull_output.sh) · [`poll_prod.sh`](scripts/poll_prod.sh) · [`watch_prod.py`](scripts/watch_prod.py) · [`stage_ckpt_dataset.py`](scripts/stage_ckpt_dataset.py) · [`gen_ckpt_probe.py`](scripts/gen_ckpt_probe.py) | verify offline · kéo output có retry · theo dõi · đóng gói checkpoint · probe cổng resume |
| [`scripts/rotate_kaggle_creds_wizard.sh`](scripts/rotate_kaggle_creds_wizard.sh) | xoay vòng token Kaggle (việc của người, browser) |
| [`tests/`](tests/) | 7 cây test — bảng ở [`docs/tests.md` §2](docs/tests.md) |
| `papers/lwfednids-bouayad-2024/notebook/` | `.ipynb` **sinh tự động**, nhúng key W&B ⇒ mode 600, **gitignore**; thư mục đã push có file `PUSHED` |
| `papers/lwfednids-bouayad-2024/runs/` | `pulls/` · `merged/` · `logs/` (poll) |

---

## 3. Nắm ngay — phương pháp và những gì đã chốt

**Khởi tạo (server, một lần):** `torch.manual_seed(42)` → θ₀ (DAGSNet 395.024) → mask zero-shot
(DepGraph, L1 theo nhóm, cắt 70 % kênh mỗi lớp, miễn Linear cuối) → θ' = M ⊙ θ₀ **xoá vật lý**
→ 35.891 tham số, plan `524e7ab78439698c` (local; Kaggle tự tính lại và ghi vào manifest).

**Một round:**
```
mọi client j:  θ_j ← θ^t ;  1 epoch AdamW(lr_t, wd 1e-4), CE 16 lớp, clip 1.0, fp16 AMP
server:        θ^{t+1} = (1/N) Σ_j θ_j        # Eq. (9), KHÔNG trọng số theo N_j
eval:          θ^{t+1} trên đủ 10.761.343 dòng test → 10 metric; mean/std/min/max loss & gnorm theo client
```
lr_t cosine 1e-3 → 1e-5 trên T = 50, hằng trong round. Batch **512 / 512 / 256**. Seed 42.

**Bảy câu hỏi đã hỏi và chốt với chủ dự án (2026-09-22)** — lý lẽ đầy đủ ở [`docs/rebuild.md`](docs/rebuild.md):

| | chốt |
|---|---|
| Bài báo | Lightweight-Fed-NIDS (Bouayad 2024), **không** phải NILM (dự án `nilm_fl` đã xong) |
| Sparsity | **chỉ 0,7** (= tỉ lệ kênh mỗi lớp; tham số còn 9,1 %) |
| Eval | mô hình tổng hợp trên test + trung bình thống kê train của mọi client (phương án a) |
| Optimizer | AdamW wd 1e-4, cosine theo round — giống các dự án anh em |
| Tổng hợp | 1/N như bài báo |
| Cắt tỉa | DepGraph + L1, tính một lần trên server, cắt vật lý, plan lưu trong cfg/checkpoint; `torch-pruning` cài vào env `nckh` và `pip --no-deps` trong notebook |
| Tài khoản | 20c → `minhtran0601`, 50c → `catbaochau`, 100c → `trietbackup` |

**Checkpoint chỉ lưu TRỌNG SỐ**: `state_dict` của θ^t + `cfg` (có `prune_plan`); nạp bằng
`weights_only=True`, dựng lại bằng `build_model(cfg)` **không cần torch-pruning**,
`load_state_dict(strict=True)`, assert `n_params` (35.891) — kiểm trong `test_smoke_real.py`.

### 3.1 Những thứ sẽ trông "sai" mà thật ra là đúng

Ở [`docs/rebuild.md` §3](docs/rebuild.md): vòng 1 thấp; `loss_client_std` lớn (α = 0,5); mô hình
35,9 k tham số có thể kém DAGSNet đầy đủ (đó là câu hỏi của bài báo); `skipped` ~1–2/client.
**Lý do thật sự phải dừng**: [`docs/kaggle.md` §6](docs/kaggle.md).

## 4. Kaggle — phân công tài khoản

Chi tiết quota, dataset, runbook, bẫy: [`docs/kaggle.md`](docs/kaggle.md).

| kịch bản | tài khoản | phiên dự kiến | trạng thái |
|---|---|---|---|
| 20c | `minhtran0601` | 1 (≈ 3,5 h đo từ probe) | probe + production `lightweight-fed-nids-veremi-20-clients` v1 **COMPLETE 50/50**, kéo về + verify pass |
| 50c | `catbaochau` | 1 (≈ 3,5 h) | production `lightweight-fed-nids-veremi-50-clients` v1 **COMPLETE 50/50**, kéo về + verify pass |
| 100c | `trietbackup` | 1 (651 s/round ⇒ ≈ 9,1 h) | production `lightweight-fed-nids-veremi-100-clients` v1 **COMPLETE 50/50 (1 phiên, 6,9 h)**, kéo về + verify pass |

⚠ `minhtrit06` và `odixeuit` là **ưu tiên sau cùng** (chủ dự án, 21-09).
⚠ Token Kaggle **không nằm trong repo** — ở `~/.kaggle/accounts/` mode 600. **Không dán token vào chat.**
⚠ Key W&B nhúng trong notebook là **vĩnh viễn** trong version history Kaggle ⇒ notebook phải
`is_private: true`, `.ipynb` không commit.

## 5. Việc tiếp theo

Ba run đã xong, kéo về, verify và viết báo cáo [`docs/report.md`](docs/report.md). Còn mở, **đều cần
chủ dự án quyết định** ([`docs/report.md` §8](docs/report.md)):

1. Baseline DAGSNet **chưa cắt** theo đúng giao thức này (D5): ≈ 4,2 / 3,9 / 8 h trên 2×T4, một
   phiên mỗi kịch bản. Đây là run duy nhất trả lời được "cắt 70 % có giữ hiệu năng không".
2. f1_macro vẫn tăng ở round 50: tăng T hoặc kéo dài đuôi lr thấp. Việc này đổi cấu hình đã chốt.
3. Dọn probe kernel trên Kaggle: thao tác xoá ngoài repo, **hỏi chủ dự án trước**.
