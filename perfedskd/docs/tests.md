# tests.md — cách kiểm và mọi số đã đo

Bằng chứng của dự án. `CONTEXT.md` chỉ giữ 3–4 con số đầu bảng và trỏ về đây.

**Nguyên tắc đọc file này:** ba loại bằng chứng **không** chứng minh cho nhau —
① kiểm ở **local** (sm_86, 1 GPU, fixture nhỏ) chứng minh **đúng/sai**, không chứng minh
nhanh/chậm; ② đo trên **2×T4** (sm_75) chứng minh tốc độ và tính khả thi của CUDA graph;
③ **run production hoàn tất** mới chứng minh kết quả khoa học. Luôn ghi rõ đang trích loại nào.

Cập nhật: **2026-09-22**.

---

## 1. Cách chạy

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
cd ~/nckh/veremi/perfedskd
```

| lệnh | thời gian |
|---|---|
| `CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_units.py` | ~30 s |
| `CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_smoke_real.py` | ~7 phút |
| `CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_two_workers.py` | ~3 phút |
| `CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_compile_gate.py` | ~1 phút |
| `CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_ckpt_verify.py` | ~2 phút |
| `python scripts/run_local_checked.py python tests/test_gpu_local.py` | ~4 phút, **dùng GPU local** |
| `CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python tests/test_notebook_sim.py` | ~3 phút (`--gpu` để chạy cell calibration) |
| `CUDA_VISIBLE_DEVICES="" python scripts/validate_notebooks.py` | vài giây |

⚠ **Ba quy tắc cứng của máy local** (`knowledge/local-env.md`): mọi test đi qua
`scripts/run_local_checked.py` (watchdog RAM); **một cây test một lúc**; watchdog dừng thì
**giảm bài test, không nới ngưỡng**. RAM WSL 7,6 GiB — vượt RAM làm đơ cả WSL, không phải
`MemoryError` sạch sẽ.

---

## 2. Bảy cây test — kiểm gì, và vì sao cần

| file | kiểm | vì sao mục này tồn tại |
|---|---|---|
| [`tests/test_units.py`](../tests/test_units.py) | **46 check** (§3.1) | mọi công thức của phương pháp, đối chiếu bản viết tay |
| [`tests/test_smoke_real.py`](../tests/test_smoke_real.py) | end-to-end trên **dữ liệu VeReMi thật** (§3.2) | pipeline + hợp đồng resume; fixture ngẫu nhiên không chứng minh được |
| [`tests/test_two_workers.py`](../tests/test_two_workers.py) | 1 worker ≡ 2 worker **bit-for-bit** | thứ tự cộng float và lịch điều phối không được ảnh hưởng kết quả |
| [`tests/test_compile_gate.py`](../tests/test_compile_gate.py) | **7 check**: cổng compile chứng nhận / từ chối / fallback | `torch.compile` lazy ⇒ `try` quanh lời gọi **không bắt được gì**; phải so số thật |
| [`tests/test_ckpt_verify.py`](../tests/test_ckpt_verify.py) | **36 check**, trong đó **27 ca giả mạo bị từ chối** | mỗi ca từng **lọt** qua một verifier ở dự án anh em |
| [`tests/test_gpu_local.py`](../tests/test_gpu_local.py) | Inductor + CUDA graph **thật** | sm_86 chứng minh **lược đồ graph**, ❌ **không** chứng minh sm_75 |
| [`tests/test_notebook_sim.py`](../tests/test_notebook_sim.py) | chạy **chính các cell đã sinh** trên cây `/kaggle/input` giả (có sidecar `.stats.json`) | notebook là thứ thật sự chạy; module pass không đảm bảo cell pass |

---

## 3. Kết quả đo ở local — sm_86, 1 GPU (RTX 3050 4 GiB), fixture nhỏ

**Trạng thái 2026-09-21: 7/7 cây test PASS trên code hiện tại.**

### 3.1 `test_units.py` — 46 check

Những check đáng nêu tên:

| check | số đo |
|---|---|
| DAGSNet đúng **395.024** tham số, `(B,66) → (B,16)` | ✔ |
| round-trip `flatten` ↔ `unflatten_into` | `max\|Δlogit\| = 0` |
| `fold_bn` chính xác trong `eval()` | `max\|Δlogit\| = 2,4 × 10⁻⁷` |
| `eval_model` phủ **mọi** dòng kể cả tail batch; confusion = `bincount(y·C + pred)` | ✔ |
| `skd_loss` = Eq. (2) viết tay: `ce`, `kd = KL(p_teacher ‖ p_student)`, `φ = ce + λ·kd` | ✔ |
| **không** gradient nào tới logit của teacher | ✔ |
| một backward của φ ≡ gradient của `CE + λ·KL` với teacher là hằng | ✔ |
| teacher ≡ student ⇒ `kd = 0` (đúng trạng thái round 1) | `\|kd\| < 10⁻⁷` |
| λ chỉ scale số hạng KD, không đụng CE | ✔ |
| **λ = 0 ⇒ đúng bằng CE training thuần** (không có số hạng ẩn nào) | `allclose, atol 1e-6` |
| teacher **không đổi** sau một `client_update`, **kể cả buffer BatchNorm** | `torch.equal` |
| student ở `train()`, teacher ở `eval()` sau khi xong | ✔ |
| `aggregate` chia **\|S\|** chứ không chia số client hệ thống; `aggregate([])` **raise** | ✔ |
| `threshold` = trung bình accuracy; `select_clients` = `{m : a_m < τ}`, đã sort | ✔ |
| accuracy-đều ⇒ chọn **rỗng** (driver phải giữ global model cũ) | ✔ |
| lịch LR cosine: hai đầu, đơn điệu giảm, trung điểm, ngoài `1..T` **raise** | ✔ |
| 10 metric vs **sklearn** | `max\|Δ\| = 5,6 × 10⁻¹⁷` |

### 3.2 `test_smoke_real.py` — 4 client thật × 3 round

| kiểm | kết quả |
|---|---|
| train đã z-score / test **chưa** (canh bằng `mean(f_snd_spd)` thô = 7,81) | ✔ |
| fp16 an toàn | `max\|x\|` train 120,9 / test 355,0 |
| artifact đủ mọi round (weights, resume, metrics, confusion ×2, logs, marker) | ✔ |
| **chuỗi chọn device**: `S_1` = tất cả; `S_t` = `S_{t+1}` round trước; log từng device khớp; `τ` = trung bình accuracy; `comm_saving` khớp | ✔ |
| file trọng số có **11 khoá** gồm `selected` / `selected_next` / `tau`; `selected_next` là `int` thuần, `tau` là `float` thuần ⇒ `weights_only=True` nạp được | ✔ |
| dựng lại: aggregate + 4 × 395.024, `strict=True` | file 8,26 MB |
| **crash replay**: xoá marker round cuối rồi chạy lại | `max\|Δweight\| = 0,0`, `history.csv` không trùng dòng |
| **resume phiên sau**: xoá hẳn round cuối rồi chạy phiên mới | `max\|Δweight\| = 0,0` so với run không bị ngắt |
| resume khai `rounds` khác ⇒ chặn ở gate | `SystemExit` |
| phiên không-có-gì-để-làm ⇒ 0 round, artifact không bị chạm | ✔ |
| `require_resume` mà không có checkpoint ⇒ chết **trước** khi train | `SystemExit` |

Số khoa học của fixture này **vô nghĩa** (4 client × 12k dòng): `f1_macro` ≈ 0,02.
Nó kiểm **pipeline**, không kiểm chất lượng model.

### 3.3 `test_two_workers.py`

`max|Δweight| = 0,0`; `f1_macro` 0,013777 trùng khít; `τ` và **tập chọn device** giống nhau
(3 device). ⇒ kết quả không phụ thuộc lịch điều phối hay số worker.

### 3.4 `test_compile_gate.py` — 7 check

| tình huống | cổng phải làm gì | kết quả |
|---|---|---|
| compiler đúng nhưng **có dòng RNG riêng** (đúng kiểu Inductor) | **chứng nhận** | ✔ |
| compiler lệch logit +0,3 (dropout 0,1 và 0,0) | **từ chối**, phục hồi `p`, model vẫn trainable | ✔ |
| compiler **raise** | fallback eager **ồn ào**, phục hồi `p` | ✔ |
| cổng eval: ba tình huống trên | chứng nhận / từ chối / fallback | ✔ |

Thêm: sau mọi nhánh, student về `train()` và teacher về `eval()` — một cổng để teacher ở
`train()` sẽ khiến teacher trả lời khác nhau ở **mỗi batch**.

### 3.5 `test_ckpt_verify.py` — 36 check

Hợp đồng checkpoint: chỉ tensor + kiểu nguyên thuỷ, có buffer BN, `weights_only=True`,
dựng lại `strict=True`, checkpoint **không có `selected_next` dùng được ⇒ raise**.

Fingerprint: **15 khoá khoa học** làm nó đổi (gồm `lam`, `select_metric`, `rounds`,
`lr_schedule`, `lr_min`), **5 khoá vận hành** thì không (`eval_batch`, `max_seconds`,
`compile`, `world_size`, `preds_rounds`); thiếu khoá ⇒ `KeyError`, **không** default.

**27 ca giả mạo, tất cả bị từ chối:** metric per-client = NaN · mean bị ghi đè · metric
global bị ghi đè ở json / ở hàng / trong file trọng số · confusion global bị sửa · xoá một
dòng `history.csv` · sửa một giá trị `clients.csv` · per-class f1 = 999 · per-class recall
global = 999 · thiếu một client trong log · client applied 0 bước · client applied bước có
gradient không hữu hạn · sửa predictions / predictions global · thiếu predictions đã khai ·
thiếu log client · client train ở LR lệch lịch · hàng LR lệch lịch · NaN trong trọng số ·
run ngắn hơn yêu cầu · **`tau` bị ghi đè ở json / ở file trọng số** · **`selected_next` bị
nới ra mọi device** · **chuỗi chọn bị đứt (`S_t` ≠ `S_{t+1}` của round trước)** · **cờ
`selected` của một log client bị đảo** · thiếu file confusion global.

Import/resume: mount **read-only** chỉ nhập round đã commit ở nguồn, file ghi lại được,
history dựng lại; hai history cùng fingerprint ⇒ **từ chối ghép**; handoff bundle neo hai
đầu (bytes ↔ `chain[r]`, `prev_sha` ↔ `chain[r-1]`); chain sai / round sai / thiếu handoff ⇒
không nhập gì; bundle không ghép được lên cây đã có round riêng; **merge s1 + bundle s2 ⇒
byte-identical với cây một phiên**, và overlap bị train lại ⇒ merge từ chối.

### 3.6 `test_gpu_local.py` — Inductor thật, sm_86

| | |
|---|---|
| cổng compile train trên **dữ liệu thật** | student `max\|Δlogit\| = 9,8 × 10⁻⁴`, `rel\|Δgrad\| = 4,6 × 10⁻³`, **0 flip / 235 hàng quyết định**; teacher `max\|Δlogit\| = 4,9 × 10⁻⁴`, **0 flip / 251** |
| teacher **không** dịch chuyển qua một bước compiled | `torch.equal` |
| bước SKD compiled vs eager | 12,14 vs 23,21 ms/step @256 ⇒ **1,91×** |
| compiled vs eager sau một epoch (dropout 0) | `max\|Δw\|` = 3,0 × 10⁻² = **3,8 %** của chính bước cập nhật |
| eval compiled-folded vs eager-folded | 296.198 vs 243.827 rows/s (**1,21×**), **0 / 12.000** dự đoán lệch |
| model thứ hai qua **cùng** template eval compiled | confusion khác, đủ số dòng |
| peak VRAM @256 | **0,12 GiB** |

### 3.7 `test_notebook_sim.py`

CPU: chạy **mọi** cell trừ calibration — hardware, CFG, ghi module, prepack (cache +
`content_id`), train 2 round, cell verifier ⇒ `ok`, artifact đủ.
GPU: chạy cell calibration, `train_backend = eager` như mong đợi ở fixture nhỏ.

---

## 4. Kết quả đo trên 2×T4 (sm_75) — probe 2026-09-21

Nguồn: `reports/calibration.json` của mỗi probe, và W&B summary dưới tiền tố `cal_`.
Kernel: `catbaochau/perfed-skd-veremi-20-clients-probe`,
`khanhmay0304/perfed-skd-veremi-100-clients-probe`.

⚠ `proj/perfedskd.py` đã đổi **sau** khi push probe: một đường dẫn trong docstring, và đổi tên
tham số teacher của `client_update` từ `Tc/Te` sang `Vc/Ve`. Cell calibration gọi hàm đó
**theo vị trí**, nên số đo dưới đây vẫn là số của code hiện tại — phép tính không đổi một bit.
`validate_notebooks.py` in dòng `note ... has moved on since the push` cho đúng chuyện này.

**Cả hai probe báo `backend = compiled/compiled`** ⇒ cổng compile trên sm_75 chứng nhận **cả**
graph student **lẫn** graph teacher; không rơi về eager. Đây là rủi ro lớn nhất của bản dựng
và nó đã được loại.

| | 20c probe (batch 512) | 100c probe (batch 256) |
|---|---:|---:|
| bước SKD **compiled** | **7,85 ms/step** | **7,27 ms/step** |
| bước SKD eager | 52,83 | 55,98 |
| **tăng tốc compile** | **6,73×** | **7,70×** |
| compile + gate (một lần/phiên) | 103 s | 110 s |
| eval compiled-folded @**16384** | **400.053 rows/s** | **406.334 rows/s** |
| eval compiled-folded @32768 | 359.102 | 378.040 |
| eval eager-folded @16384 | 336.740 | 339.810 |
| eval eager-folded @32768 | 266.170 | 281.916 |

**Hai quyết định vận hành rút ra:**

1. **`eval_batch = 16384`, không phải 32768.** 32768 chậm hơn ở cả hai batch size (−10 % và
   −7 %). Notebook giữ 16384 như đã sinh ⇒ **không** sinh lại.
2. **Fallback eager làm run chậm gần 7×.** Cao hơn hẳn mức 2,86× của bước DAGSNet đơn, vì bước
   SKD có **ba** graph (teacher fwd, student fwd, student bwd) nên tỉ trọng launch overhead lớn
   hơn. ⇒ W&B báo `backend = eager` là **lý do phải dừng run**, không phải chuyện nhỏ.

### 4.1 Ngân sách suy từ số đo trên

`train = steps ÷ 2 GPU × ms/step`; `eval = ⌈(M+1)/2⌉ model trên GPU đông nhất × 10.761.343 ÷ rows-per-s`.

| | steps/round | train (s) | eval (s) | vòng (phút) | 50 vòng (h) | phiên 11 h |
|---|---:|---:|---:|---:|---:|---|
| 20c | 84.083 | 330 | 282 (11 model/GPU) | **10,2** | **8,5** | **1** |
| 50c | 84.098 | 330 | 685 (26 model/GPU) | **16,9** | **14,1** | **2** |
| 100c | 168.200 | 611 | 1.351 (51 model/GPU) | **32,7** | **27,2** | **3** |

⚠ **50c không có probe riêng.** Batch của nó bằng batch 20c nên dùng chung `ms/step`, và chi
phí eval tỉ lệ thuận số model. Đây là **nội suy**, không phải số đo — phải ghi rõ khi báo cáo.

⚠ Bảng này là **dự phóng từ micro-benchmark**, chưa phải thời gian round thật. Dự án anh em đo
được 100c bị **nghẽn CPU** trên 4 vCPU (+20 % round time). Nếu điều đó lặp lại thì 100c thành
~35 phút/round ⇒ 29 h, sát trần quota 30 h. Round thật của probe là số chốt.

### 4.2 Round thật của probe

**20c — COMPLETE, 2 round** (`catbaochau/perfed-skd-veremi-20-clients-probe`, 21-09):

| round | train (s) | eval (s) | **tổng (s)** | \|S\| → kế | τ | f1_macro mean | global f1 | ce | kd | skip/steps | VRAM train / eval |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 400,4 | 316,1 | **717,2** | 20 → 9 | 0,5818 | 0,5152 | 0,2020 | 0,801 | 0,485 | 4 / 84.083 | 7,08 / 7,09 GiB |
| 2 | 359,4 | 314,2 | **674,3** | 9 → 7 | 0,5915 | 0,5015 | 0,6155 | 0,594 | 0,180 | 9 / 84.083 | 7,06 / 7,07 GiB |

Round 2 là trạng thái dừng (round 1 tốn thêm cho lần chạm đầu của cuDNN benchmark).
**674 s/round** so với 612 s dự phóng ⇒ micro-benchmark **lạc quan 10 %**; phần thiếu là commit
I/O, chi phí `fold_bn` + `load_folded` mỗi model, và eval bị lệch 11 vs 10 model giữa hai GPU.

**Bốn dự báo ở `rebuild.md` §3 đều đúng, đo được:**

| dự báo | đo được |
|---|---|
| `global_*` rất tệ ở round 1, lên từ round 2 | global f1 **0,202 → 0,615**, global acc 0,402 → 0,754 |
| `kd` cao nhất ở round 1 | **0,485 → 0,180** |
| `\|S\|` về ≈ M/2 sau round 1 | 20 → **9** → 7 |
| `f1_macro` của model cá nhân hoá có thể giảm | 0,5152 → 0,5015 |

**Suy ra ngân sách thật** (dùng round 2 làm trạng thái dừng; `ms/step` thật = 359,4 s ÷ 42.042
bước = **8,54 ms @512**; eval thật ≈ **28,5 s/model/GPU**):

| | train (s) | eval (s) | vòng (phút) | 50 vòng (h) | phiên |
|---|---:|---:|---:|---:|---|
| 20c | 359 | 314 (11 model/GPU) | **11,2** | **9,4** | **1** |
| 50c | 359 | 741 (26 model/GPU) | **18,4** | **15,4** | **2** |
| 100c | ~665–800¹ | ~1.454 (51 model/GPU) | **35,3–37,6** | **29,4–31,3** | **3** |

¹ `ms/step` @256 suy từ tỉ lệ calibration (7,27/7,85) ⇒ 7,91 ms; khoảng trên đã cộng dự phòng
+20 % cho **nghẽn CPU** mà dự án anh em đo được ở 100c trên 4 vCPU.

⚠ **100c rơi đúng vào vùng 29–31 h, tức sát hoặc vượt trần quota 30 h của một tài khoản.**
Probe 100c là số chốt — xem quy tắc quyết định ở [`kaggle.md` §3](kaggle.md).

**100c — COMPLETE, 2 round** (`khanhmay0304/perfed-skd-veremi-100-clients-probe`, 21-09):

| round | train (s) | eval (s) | **tổng (s)** | \|S\| → kế | τ | f1_macro mean | global f1 | ce | kd | skip/steps | VRAM train / eval |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 753,4 | 1.362,3 | **2.118,2** | 100 → 40 | 0,4595 | 0,3647 | 0,2070 | 0,798 | 0,490 | 3 / 168.200 | 7,01 / 7,33 GiB |
| 2 | 611,8 | 1.359,2 | **1.974,5** | 40 → 43 | 0,5148 | 0,3746 | 0,5363 | 0,731 | 0,204 | 9 / 168.200 | 7,15 / 7,31 GiB |

**1.974 s = 32,9 phút/round** ở trạng thái dừng. Dự phóng 35,3–37,6 phút đã cộng +20 % cho
nghẽn CPU — **không xảy ra**: train 612 s khớp dự phóng 611 s, eval 1.359 s khớp 1.351 s, tức
micro-benchmark 100c **chính xác** (khác 20c lạc quan 10 %). Eval chiếm **69 %** một round.

Bốn dự báo cũng đúng ở 100c: global f1 0,207 → 0,536 · kd 0,490 → 0,204 · |S| 100 → 40 → 43 ·
f1_macro mean 0,365 → 0,375 (dự báo là "có thể giảm", không bắt buộc).

**Per-client** (`metrics/round_00X.json` và `clients.csv` trong output kernel; W&B **không** có
per-client): τ và S_{t+1} dựng lại **khớp 100 %** từ accuracy per-client ở cả hai round.
Accuracy trải 0,18–0,63 (r1), 0,32–0,66 (r2). Client được chọn ở r2 (nhận ω^1) tăng accuracy
trung bình **+0,140**; client không được chọn **−0,001** — cơ chế chọn thiết bị làm đúng việc
nó phải làm. 18 client bị chọn cả r2 lẫn r3, 22 chỉ r2, 25 chỉ r3.

### 4.3 Round thật của production (đọc W&B 21-09 ~16:50Z, đang chạy)

| | khởi động trước round 1 (s)¹ | round đầu phiên (s) | round ổn định (s) | eval s/model/GPU | 50 round + khởi động (h) | phiên |
|---|---:|---:|---:|---:|---:|---|
| 20c (19 round) | **903** | 639 | **593** | 22,4 | **8,5** | 1 |
| 50c (10 round) | **1.413** | 1.199 | **1.090** | 28,3 | **16,0** (33 + 17 round) | 2 |
| 100c (probe) | ~1.800 (ước từ 50c) | 2.118 | **1.974** | 26,7 | **~29,2** (3 phiên) | 3 |

¹ `_runtime − seconds` ở hàng W&B đầu tiên: decode 43 M dòng + spawn 2 worker + compile, tính
từ `wandb.init`. Chi phí này trả **mỗi phiên**, nên nó là lý do phiên phải dài nhất có thể.

20c production chạy **nhanh hơn probe** (593 vs 674 s: eval 246 vs 314 s) — bảng "ngân sách
thật" ở trên (11,2 / 18,4 / 35,3–37,6 phút) vì thế **lạc hậu**; số ở đây là số chốt. Cả hai
run `compiled/compiled`, `skipped` ≤ 29/84.083 và ≤ 19/84.098, không tăng dần; VRAM ~7,0 GiB.

### 4.4 Phiên đã xong (đọc W&B 22-09 02:12Z; verify offline ở cuối mục)

| | round | khởi động (s) | round đầu (s) | round median / max (s) | phiên (h, `_runtime` cuối) | quota tốn (h) | `skipped` max | VRAM max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20c phiên 1 (`catbaochau`) | **50/50** | 903 | 639 | **593** / 654 | 8,42 | 8,99 | 32 / 84.083 | 7,08 GiB |
| 50c phiên 1 (`trietbackup`) | **34/50** | 1.413 | 1.199 | **1.088** / 1.136 | 10,46 | 10,47 | 24 / 84.098 | 7,17 GiB |
| 100c phiên 1 (`minhtriethihi`, đang chạy) | 14+ | **2.560** | **2.345** | **2.199** / 2.249 | 9,30 (r14) | — | 48 / 168.200 | 7,15 GiB |

Cả ba `compiled/compiled`; `skipped` dao động, không tăng dần; không có dấu hiệu dừng nào ở
[`kaggle.md` §6](kaggle.md). 50c dừng đúng như cổng driver dự tính (34 round ở 10,46 h, 33
dự kiến). **100c chậm hơn probe 11 %** (2.199 vs 1.974 s) — chênh nằm ở **eval** (1.586 vs 1.359
s/round), `train_sec` 611 s **khớp** probe 612 s ⇒ không phải nghẽn CPU; khởi động 43 phút thay
vì 30 ⇒ 17 round/phiên thay vì 19. Hệ quả cho kế hoạch: [`kaggle.md` §3](kaggle.md).

Chất lượng ở round cuối của phiên (mean qua M model cá nhân hoá trên test toàn cục / model tổng hợp):

| | round | f1_macro mean | accuracy mean | global f1_macro | global accuracy | \|S\| | τ | f1_macro mean tốt nhất | global f1 tốt nhất |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 20c | 50 | **0,6504** | 0,6229 | 0,6616 | 0,5539 | 6 | 0,6229 | r50 = 0,6504 | r41 = 0,6622 |
| 50c | 34 | **0,5655** | 0,5619 | 0,6387 | 0,5629 | 20 | 0,5619 | r34 = 0,5655 | r30 = 0,6554 |
| 100c | 14 | 0,4112 | 0,4566 | 0,5941 | 0,5413 | 47 | 0,4566 | r14 = 0,4112 | r7 = 0,6123 |

Bốn hình dạng dự báo ([`rebuild.md` §3](rebuild.md)) đều lặp lại ở production: global f1 round 1
= 0,204 / 0,222 / 0,217 rồi lên; `|S|` = M ở round 1 rồi về ~M/3–M/2 (20c kết thúc ở 6/20,
tức τ cao hơn phần lớn device — hệ quả tự nhiên khi accuracy hội tụ sát nhau); f1_macro mean
vẫn đang **tăng** ở round cuối của cả ba, chưa bão hoà.

---

## 5. Ranh giới — cái gì các số trên **không** chứng minh

| |
|---|
| Local sm_86 pass ⇒ **không** chứng minh sm_75. Chỉ probe trên T4 chứng minh được. |
| Local 1 GPU ⇒ **không** chứng minh topology 2 worker / 2 GPU trên phần cứng thật. |
| Micro-benchmark ⇒ **không** chứng minh thời gian round thật (thiếu commit I/O, nghẽn CPU, biến động queue của Kaggle). |
| Fixture nhỏ ⇒ **không** nói gì về chất lượng model. Mọi `f1_macro` ở §3 là số của pipeline. |
| Verifier chứng nhận **đúng những round có trên đĩa**. "Kịch bản đã xong" chỉ được nói sau `verify_run --require-rounds 50` trên cây đã merge. |
| Một seed, một lần chạy. Không có replication. |
