# CONTEXT — PerFed-SKD trên VeReMi NextGen / DAGSNet

**File này là trung tâm điều hướng.** Đọc nó để nắm dự án trong 2 phút và biết mở đúng file nào
khi muốn sâu hơn. Nó **không** chứa bảng số liệu — chúng ở [`docs/`](docs/).

Cập nhật: **2026-09-23 09:35Z**.

---

## Dự án này là gì

Dựng lại phương pháp **PerFed-SKD** (Singh, Rupchandani, Adhikari — *Personalized Federated
Learning for Heterogeneous Edge Device: Self-Knowledge Distillation Approach*) trên bộ dữ liệu
**VeReMi NextGen** (16 lớp, 66 đặc trưng, 43.045.415 dòng train / 10.761.343 dòng test), với bộ
phân loại **DAGSNet 395.024 tham số** làm model của mọi thiết bị và của server.

Ba kịch bản: **20, 50, 100 edge device** (Dirichlet α = 0,5), mỗi kịch bản 50 round × 1 epoch
trên **2 × Tesla T4** của Kaggle, mỗi kịch bản một tài khoản.

**Trạng thái (23-09, ~09:35Z): CẢ BA KỊCH BẢN XONG HẲN.** Code + test xong (**7/7 cây test local
pass**), bốn dự báo ở [§3.1](#31-bốn-thứ-sẽ-trông-sai-mà-thật-ra-là-đúng) đúng ở cả ba kịch bản.
Mọi cây `runs/merged/perfedskd_{20,50,100}c` **verify 50 round pass**. f1_macro mean ω_m / ω^t ở
r50: **20c 0,650 / 0,662 · 50c 0,690 / 0,720 · 100c 0,654 / 0,687**. 100c: phiên 3
`khanhmay0304/…-100-clients-s3` (nhập round 38 từ dataset checkpoint) COMPLETE round 39–50,
merge s1+s2+s3.
**REPORT chính thức:** [`docs/report.md`](docs/report.md) — đã điền đủ 100c, viết §4.5 (tăng dốc
cuối run, loss device tăng tập trung ở S_t) và §4.6 (số device không đơn điệu). Số liệu + hình ở
`papers/perfedskd-singh-2025/report_data/` (`report_data.py` chạy lại trên 3 cây 50 round).

**Kéo output (23-09):** ~3,4 MB/s. Output phiên nối tiếp **chứa lại mọi round đã import** ⇒ 100c chỉ
cần kéo đủ **cây phiên cuối** (s2 đã có 1–38, 6,0 GB), không kéo riêng s1; 50c s2 kéo bằng
`PULL_PATTERN` (biến môi trường mới của `pull_output.sh`) bỏ trọng số round 1–34 đã có. DNS WSL vẫn
chập chờn — vòng ngoài chờ `getent hosts`. **Watchdog RAM `run_local_checked.py` cần ≥ 3.584 MiB
khả dụng** — hai cửa sổ VS Code ăn ~3,5 GB; đóng bớt một cửa sổ trước khi verify (23-09 đã làm).

**Token (22-09):** token MCP `odixe0502` mới đã lưu + introspect OK. **Quét đĩa thấy KGAT của cả
8 tài khoản và refresh token `minhtran0601` nằm trong transcript Claude Code** (do dán vào chat)
⇒ cần **xoay vòng toàn bộ** — việc của người (browser): chạy
`scripts/rotate_kaggle_creds_wizard.sh` ([`docs/KAGGLE.md` §1.1](docs/KAGGLE.md)).
**Không dán token vào chat nữa.**

---

## 1. Điều hướng — muốn biết gì thì mở file nào

| muốn biết | mở |
|---|---|
| **Bài báo nói gì**, và bài báo **để trống / tự mâu thuẫn** ở đâu (G1–G10) | [`docs/paper.md`](docs/paper.md) |
| **Mọi lựa chọn của bản dựng**: 13 deviation, hợp đồng artifact, thiết kế tốc độ (và các tối ưu đã **loại** kèm số học), ngân sách, caveat bắt buộc | [`docs/rebuild.md`](docs/rebuild.md) |
| **Kết quả** (chính thức, cả ba kịch bản 50/50) | [`docs/report.md`](docs/report.md) + `papers/perfedskd-singh-2025/report_data/` |
| **Mọi con số đã đo**: 7 cây test kiểm gì, kết quả từng check, calibration 2×T4, ranh giới của từng loại bằng chứng | [`docs/TESTS.md`](docs/TESTS.md) |
| **Chạy trên Kaggle**: tài khoản, quota, dataset, kế hoạch phiên, runbook push / theo dõi / kéo / verify / hop, bẫy vận hành | [`docs/KAGGLE.md`](docs/KAGGLE.md) |
| **DAGSNet**: kiến trúc đầy đủ, Eq. (38)–(48), hợp đồng vào/ra, 66 cột đặc trưng | [`knowledge/ARCHITECTURE.md`](knowledge/ARCHITECTURE.md) |
| **Dữ liệu**: train đã z-score / test chưa, nhãn ở đâu, ba kịch bản α = 0,5, steps/round | [`knowledge/DATASET.md`](knowledge/DATASET.md) |
| **Máy local**: RAM 7,6 GiB, VRAM 4 GiB, sm_86 ≠ sm_75, cái gì kiểm được ở local | [`knowledge/LOCAL_ENV.md`](knowledge/LOCAL_ENV.md) |
| **Bài báo gốc** | [`Personalized_..._Approach.md`](Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.md) |

**Ranh giới tài liệu — giữ đúng, vì một con số đặt sai chỗ sẽ bị phiên sau đọc như sự thật đã kiểm:**

| nơi | chứa | đừng bỏ vào đây |
|---|---|---|
| [`knowledge/`](knowledge/) | sự thật **không đổi** của dữ liệu, máy, kiến trúc — dùng chung nhiều phương pháp | bất kỳ con số của **một** phương pháp |
| [`docs/`](docs/) | tài liệu của **dự án này**: phương pháp, quyết định, số đo, vận hành | — |
| `CONTEXT.md` | điều hướng + trạng thái + những gì phải biết ngay | bảng số liệu dài |
| `../.claude/skills/` (gốc repo) | skill **dùng chung mọi dự án** | ❌ bất kỳ thứ gì riêng của dự án này |

## 2. Mã nguồn và công cụ — file nào làm gì

**Vòng đời chuẩn:** sửa `proj/*.py` → chạy test → `gen_notebook.py` → `validate_notebooks.py`
→ nhúng key W&B → **validate lại** → push. ❌ Không bao giờ sửa `.ipynb` bằng tay.

| file | vai trò |
|---|---|
| [`proj/perfedskd.py`](papers/perfedskd-singh-2025/proj/perfedskd.py) | **thuật toán**: `skd_loss` (Eq. 2), `client_update`, `aggregate` (1/\|S\|), `threshold`, `select_clients`, `lr_at` |
| [`proj/driver.py`](papers/perfedskd-singh-2025/proj/driver.py) | 2 worker / 2 GPU, cổng compile, eval M+1 model, commit atomic, gate ngân sách phiên |
| [`proj/ckpt.py`](papers/perfedskd-singh-2025/proj/ckpt.py) | file trọng số, resume, marker, fingerprint, handoff bundle |
| [`proj/verify.py`](papers/perfedskd-singh-2025/proj/verify.py) | dựng lại **mọi** con số từ artifact, kể cả τ và chuỗi chọn device |
| [`proj/model.py`](papers/perfedskd-singh-2025/proj/model.py) | DAGSNet; `build_model(cfg)` cho mọi model |
| [`proj/evaluate.py`](papers/perfedskd-singh-2025/proj/evaluate.py) · [`data.py`](papers/perfedskd-singh-2025/proj/data.py) · [`metrics.py`](papers/perfedskd-singh-2025/proj/metrics.py) | fold BN + eval compiled · parquet→fp16 thường trú · 10 metric từ confusion |
| [`scripts/gen_notebook.py`](scripts/gen_notebook.py) | sinh notebook; chứa `SCENARIOS` (batch, dataset) và `PAPER` (siêu tham số) |
| [`scripts/validate_notebooks.py`](scripts/validate_notebooks.py) | gate tĩnh trước khi tốn một version Kaggle |
| [`scripts/run_local_checked.py`](scripts/run_local_checked.py) | **watchdog RAM — mọi test local phải đi qua đây** |
| [`scripts/verify_run.py`](scripts/verify_run.py) · [`pull_output.sh`](scripts/pull_output.sh) · [`stage_ckpt_dataset.py`](scripts/stage_ckpt_dataset.py) · [`gen_ckpt_probe.py`](scripts/gen_ckpt_probe.py) · [`report_data.py`](scripts/report_data.py) · [`poll_prod.sh`](scripts/poll_prod.sh) | verify offline · kéo output có retry · đóng gói checkpoint · probe cổng resume · số liệu báo cáo · theo dõi |
| [`tests/`](tests/) | 7 cây test — bảng ở [`docs/TESTS.md` §2](docs/TESTS.md) |
| `papers/perfedskd-singh-2025/notebook/` | `.ipynb` **sinh tự động**, nhúng key W&B ⇒ mode 600, **gitignore**, không commit |
| `papers/perfedskd-singh-2025/runs/` | `pulls/` · `merged/` (cây chuẩn) · `ckpt_ds/` (staging hop) |

---

## 3. Nắm ngay — phương pháp và những gì đã chốt

**Một round của PerFed-SKD:**

```
teacher  V_m = ω_m^{t-1}   model của chính device đó ở round trước, ĐÓNG BĂNG
student  ω_m = ω^{t-1}     nếu m ∈ S_t    ← device được chọn nhận model tổng hợp
         ω_m = ω_m^{t-1}   nếu m ∉ S_t    ← device còn lại giữ trọng số riêng
train 1 epoch:  φ_m = CE(z, y) + 1,0 · KL(p_V ‖ p_ω)               Eq. (2)
upload:         chỉ device ∈ S_t
server:         ω^t = (1/|S_t|) Σ_{m∈S_t} ω_m^t                    Alg. 1 dòng 10
                τ_t = (1/M) Σ_m a_m^t   →   S_{t+1} = {m : a_m^t < τ_t}
```

**Bốn câu hỏi đã hỏi và chốt với chủ dự án (2026-09-21)** — ba câu đầu là chỗ bài báo **tự mâu
thuẫn**; lý lẽ đầy đủ ở [`docs/rebuild.md` §2](docs/rebuild.md):

| | chốt | vì sao không chọn cách kia |
|---|---|---|
| **Ngưỡng τ** | trung bình accuracy của **mọi** device (Alg.1 dòng 11) | dùng accuracy của model tổng hợp (văn xuôi §III-A) sẽ chọn **mọi** device **mọi** round ⇒ suy biến thành FedAvg+SKD, mất hẳn claim tiết kiệm băng thông |
| **Metric chọn** | **accuracy**, trên **tập test toàn cục dùng chung** | đúng chữ bài báo; f1_macro hợp dữ liệu hơn nhưng là deviation |
| **Ai train / ai upload** | **mọi** device train, **chỉ** S_t upload | theo văn xuôi §IV + Alg.2; số chia \|S\| của Alg.1 dòng 10 chỉ nhất quán với cách đọc này |
| **Loss SKD** | KL(teacher ‖ student), T = 1, λ = 1,0; teacher `eval()` + `no_grad` | bài báo không cho dạng L, không cho λ, không cho nhiệt độ |

**Cấu hình:** 50 round × 1 epoch · batch **512 / 512 / 256** · LR **cosine 1e-3 → 1e-5, T = 50**
· AdamW **wd 1e-4**, tạo mới mỗi device mỗi round · clip 1,0 · fp16 AMP (**không bao giờ bf16**
— T4 là sm_75) · seed 42 · một `ω^0` duy nhất cho mọi device.

**Bỏ hẳn:** teacher phía server trên "predefined dataset" (bài báo nhắc ở §III-A nhưng **không
có** trong bất kỳ công thức hay dòng giả mã nào) · proximal term (§III-A nhắc, Eq. 2 không có) ·
backbone trích xuất đặc trưng (bài báo **chưa từng có**) · 4 baseline.

**Eval:** mỗi round, **mọi M** model cá nhân hoá trên **đủ** 10.761.343 dòng test — 10 metric
mỗi device + mean/std/min/max, **cộng** model tổng hợp ω^t (cột `global_*`). Chính các con số
này nuôi τ, nên quy tắc chọn device được đo trên đúng artifact được công bố.

**Checkpoint chỉ lưu TRỌNG SỐ**, không lưu module: `state_dict` của ω^t + M ω_m, cộng
`selected_next` và `tau` trong **cùng** file; nạp bằng `weights_only=True`, dựng lại bằng
`build_model(cfg)` lấy từ chính file, `load_state_dict(strict=True)`, assert **395.024** tham số.

### 3.1 Bốn thứ sẽ trông "sai" mà thật ra là đúng

**Đừng dừng run vì bốn thứ này** — chúng được dự báo **trước** khi chạy
([`docs/rebuild.md` §3](docs/rebuild.md)):

1. `global_*` **rất tệ ở round 1** — round 1 là FedAvg trên M model vừa train 1 epoch từ init
   ngẫu nhiên. Từ round 2, mọi device ∈ S_t xuất phát từ cùng ω^{t-1} nên phép trung bình hợp lệ.
2. `kd` **cao nhất ở round 1** — teacher round 1 là ω^0 gần-đồng-đều; KD bằng 0 đúng ở **bước
   đầu tiên** rồi phình ra.
3. `|S|` **dao động quanh M/2**, `comm_saving ≈ 0,5`; round 1 có `|S| = M` và saving 0.
4. `f1_macro` **giảm** ở một số round — model cá nhân hoá được đo trên test **toàn cục**.

**Lý do thật sự phải dừng**: [`docs/KAGGLE.md` §6](docs/KAGGLE.md). Ngắn gọn: `backend = eager`
(chậm gần 7×), `skipped` tăng dần, f1 về 0/NaN, `train_sec` +30 %, VRAM > 14 GiB, kernel ERROR.

---

## 4. Kaggle — phân công tài khoản

Chi tiết quota, dataset, runbook, bẫy: [`docs/KAGGLE.md`](docs/KAGGLE.md).

| kịch bản | tài khoản | phiên | trạng thái |
|---|---|---|---|
| 20c | `catbaochau` | 1 (8,42 h đo) | **XONG**: COMPLETE 50/50, kéo đủ, merge, **verify pass** |
| 50c | `trietbackup` | 2 (10,46 + ~4,9 h đo) | **XONG**: phiên 1 34 + phiên 2 16 round, merge, **verify 50 pass** |
| 100c | `minhtriethihi` (phiên 1–2) → **`khanhmay0304`** (phiên 3, dataset checkpoint) | 3 (11,19 + 11,09 + 6,75 h đo) | **XONG**: 18 + 20 + 12 round (phiên 3 qua dataset checkpoint), merge, **verify 50 pass** |

**Ba quyết định của chủ dự án 21-09 ~17:00Z** (lý lẽ và số ở [`docs/KAGGLE.md` §3](docs/KAGGLE.md)):
`--max-hours` được **> 11 miễn < 12 h** · **tối thiểu số phiên** (3 là sàn của 100c: 25 round =
13,7 h > 12 h) · **biên quota tổng 5–10 phút là đủ** ⇒ phiên 3 của 100c khai
`--max-hours = quota còn lại − 0,15` để driver tự dừng thay vì bị Kaggle giết (mất cả output).

⚠ `minhtrit06` và `odixeuit` là **ưu tiên sau cùng** (chủ dự án, 21-09).
⚠ Token Kaggle **không nằm trong repo** — ở `~/.kaggle/accounts/` mode 600.
⚠ Agent **tự đổi tài khoản** được nhưng **phải nói rõ**; ❌ không bao giờ `kaggle auth login --force`.
⚠ **Không dán token vào chat** (22-09: mọi KGAT từng dán đều nằm trong transcript trên đĩa) — ghi
thẳng vào `~/.kaggle/accounts/<user>.mcp-token` hoặc chạy `scripts/rotate_kaggle_creds_wizard.sh`.
⚠ Key W&B nhúng trong notebook là **vĩnh viễn** trong version history Kaggle ⇒ notebook phải
`is_private: true`, `.ipynb` không commit.

---

## 5. Việc tiếp theo

1. **Xoay vòng token — việc của chủ dự án**: `bash scripts/rotate_kaggle_creds_wizard.sh` trong
   terminal thật (browser login 8 lần). Sau đó `/mcp` để reconnect.
2. Dọn probe kernel (`…-20-clients-probe`, `…-100-clients-probe`, `khanhmay0304/perfedskd-100c-ckpt-probe`)
   và dataset `khanhmay0304/perfedskd-100c-ckpt-s3` trên Kaggle — thao tác xoá ngoài repo, **hỏi chủ
   dự án trước** (điều kiện "phiên 3 COMPLETE + verify 50 pass" của dataset đã thoả 23-09).

**Việc mở, không bắt buộc** (chi tiết ở [`docs/report.md` §8](docs/report.md)): ablation τ = accuracy của model tổng hợp (phản thực tế
ở REPORT §4.4: chọn 80 % device ở 100c, 26 % ở 20c; ~2 h GPU ở 20c × 10 round) · λ = 0 · quét λ · baseline Local/FedAvg cùng cấu hình · gộp
nhiều model vào một eval bằng grouped convolution (đòn bẩy lớn nhất còn lại cho 100c: eval
chiếm 64 % một round — [`docs/rebuild.md` §4](docs/rebuild.md)).
