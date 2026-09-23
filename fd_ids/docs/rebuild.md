# FD-IDS trên VeReMi NextGen / DAGSNet — quyết định, deviation, hợp đồng artifact

Mọi lựa chọn dưới đây là của **bản dựng** (chủ dự án chốt 2026-09-10 hoặc ghi rõ là lựa chọn
triển khai), không gán cho tác giả bài báo. Phương pháp gốc: [`paper.md`](paper.md). Mã:
[`../papers/fd-ids-2025/proj/`](../papers/fd-ids-2025/proj/). Nhật ký đầy đủ từng phiên:
[`../CONTEXT.md`](../CONTEXT.md); mọi số đo: [`tests.md`](tests.md); kết quả: [`report.md`](report.md).

> File này được viết lại 2026-09-23 từ `CONTEXT.md` §1, §2, §8, §14–15 và `tests.md` để mọi
> phương pháp có cùng bộ `paper.md` / `rebuild.md` / `report.md`. Không có quyết định mới nào ở
> đây; mọi con số trích từ artifact đã verify hoặc từ `tests.md`.

## 1. Cấu hình đã chốt

Client update theo Eq. (6), tổng hợp theo Eq. (2):

```
L = λ·L_hard + (1−λ)·L_soft + β·L_proximal
    L_hard     = CrossEntropy(logit student, nhãn)
    L_soft     = T²·KL( softmax(Z_t/T) ‖ softmax(Z_s/T) )      Eq. (4), teacher = w_G^t
    L_proximal = (μ/2)·‖w_k − w_G^t‖²                          Eq. (3)
w_G^{t+1} = Σ_k (n_k/n)·w_k^{t+1}                              Eq. (2)
```

| tham số | giá trị | nguồn |
|---|---|---|
| Optimizer / lr | Adam / **1e-3, hằng** | bài báo Table 3 |
| μ / λ / β / T | **0,01 / 0,5 / 0,1 / 3** | bài báo Table 3 |
| KD interval | **round-wise**, KD trong loss của mọi batch cục bộ | bài báo §4.3.2 + Algorithm 1 (G1) |
| Tham gia | **toàn bộ** client mỗi round | Algorithm 1 dòng 4 |
| Batch | **512 / 512 / 256** (20c / 50c / 100c) | `knowledge/dataset.md` §4 |
| Round × epoch | **50 × 1** | chủ dự án chốt 2026-09-10 |
| Bộ phân loại | **DAGSNet**, 395.024 tham số | `knowledge/architecture.md` |
| Seed / clip / precision | 42 / grad-norm 1,0 / fp16 AMP, loss tính fp32 ngoài autocast | `knowledge/architecture.md` §4.1 |
| Phần cứng | Kaggle 2 × T4; mỗi GPU một worker, client train tuần tự, không DDP | |

⚠ **Lịch lr hằng**, không phải cosine 1e-3 → 1e-5 mà các dự án sau thống nhất từ 13-09. Ba run
của FD-IDS chạy **trước** quyết định đó và chưa chạy lại. Mọi so sánh với các phương pháp khác
trong repo phải ghi rõ khác biệt này.

## 2. Deviation — phải công bố kèm mọi con số

**Do chủ dự án chỉ định hoặc do dữ liệu:**

| hạng mục | bài báo | bản này | vì sao |
|---|---|---|---|
| Bộ phân loại | DNN 5 lớp 32-64-128-64-32, 22.095 tham số | **DAGSNet** 395.024 | chủ dự án chỉ định |
| Dữ liệu | Edge-IIoT / N-BaIoT | **VeReMi NextGen**, 16 lớp, 66 đặc trưng | chủ dự án chỉ định |
| Số client | 9 | **20 / 50 / 100** | chủ dự án chỉ định |
| Non-IID | Dirichlet θ = 1 và 0,1 | **α = 0,5 cố định** | phân mảnh dựng sẵn, không sinh lại |
| Round × epoch | 40 × 2 | **50 × 1** | chốt 2026-09-10, khớp ngân sách |
| Batch | 128 | **512 / 512 / 256** | batch 128 ≈ 75 h mỗi cấu hình, vượt quota |
| Tiền xử lý | one-hot + chọn top-k bằng MI | **không**, dùng đủ 66 cột `f_*` đã z-score | dữ liệu giao ở trạng thái đã xử lý |
| Metric | Accuracy/Precision/Recall/F1 + FPR/FNR nhị phân | **10 metric đa lớp**, không FPR/FNR | chưa chốt quy ước nhị phân hoá 16 lớp (G9) |
| Đánh giá client (Table 5, cột B/W) | có | **không**, chỉ global model | G8 và §3 dưới đây |

**Lấp chỗ trống của bài báo** (G-số ở [`paper.md`](paper.md) §6):

| | lựa chọn | ghi chú |
|---|---|---|
| **D1 (G1)** | KD **trong** loss cục bộ của mọi batch, theo Algorithm 1; không có pha KD riêng sau tổng hợp. Round 1 cũng dùng teacher = w_G^0 mới khởi tạo, không warm-up | thêm warm-up không KD là đổi phương pháp |
| **D2 (G2)** | Teacher = w_G^t, **đóng băng cả round**, chạy `eval()`. Logit teacher tính một lần cho mọi dòng của client trước khi train, lưu **fp16**, batch 16.384 | sai số so với tính online: logit < 1e-3, gradient student < 1e-3 tương đối (`tests.md` §1c.5) |
| **D3 (G3)** | Proximal theo đúng chữ: β·(μ/2)·‖w − w_G‖², cài bằng gradient `β·μ·(w − w_G)` cộng thẳng vào grad; **hệ số hiệu dụng 5e-4**. Chỉ áp lên **tham số**, không lên buffer BN | |
| **D4 (G4)** | **Adam** (betas 0,9/0,999, eps 1e-8, wd 0, không amsgrad), **tạo mới mỗi client mỗi round**; GradScaler cũng tạo mới | bài báo không nói vòng đời moment; reset là quy ước triển khai |
| **D5 (G5)** | Mô hình xuất **logit**; CE và KD tính trên logit (không softmax hai lần) | |
| **D6 (G10)** | BN trong tổng hợp: `running_mean/var` trung bình theo n_k/n như trọng số; `num_batches_tracked` lấy max | trung bình các variance không phải pooled variance; DNN của bài báo không có BN |
| **D7** | Client được xử lý theo **thứ tự cid** khi cộng Eq. (2), không theo thứ tự hoàn thành | thứ tự cộng float quyết định bit của kết quả |
| **D8** | **Hợp đồng RNG**: nguồn ngẫu nhiên của một client dẫn xuất từ `(seed, round, client)`; kết quả không phụ thuộc worker nào nhận client | đo trên CPU `compile=False`: 1 vs 2 worker cho Δ trọng số = 0; chưa suy rộng sang CUDA compiled |
| **D9** | Một seed, một lần chạy mỗi cấu hình | không có replication |

## 3. Vì sao chỉ đo global model

Algorithm 1 dòng 14: `w_k = w_G^t`. **FD-IDS không có bước cá thể hoá.** Mỗi round client bị ghi
đè bằng global model, train, gửi lên, rồi lại bị ghi đè. Không trọng số client nào tồn tại qua
ranh giới round, nên model duy nhất tồn tại liên tục là **global model**. Bản dựng đo **10 metric
của w_G^t mỗi round, trên đủ 10.761.343 dòng test**. Cột B/W của Table 5 không được tái lập: bài
báo không mô tả đủ giao thức đánh giá từng client (G8).

## 4. Hiệu năng trên 2×T4 — số đo

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| bước / round (tổng mọi client) | 84.083 | 84.098 | 168.200 |
| round steady (s), backend compiled | 509–524 | 528–538 | 671–981 |
| Σ `seconds` 50 round (h, không gồm startup) | 7,22 | 7,37 | 10,54 |
| phiên Kaggle | 2 (42 + 8 round) | 1 | 2 (25 + 25 round) |

* **Compile:** `torch.compile` chứng nhận trên T4 (`max|Δlogit| ≤ 9,8e-4`), **2,06×** so với eager
  (17,9 → 8,7 phút/round). Ba run đầu chạy eager vì lỗi #39 trong gate compile của bản dựng (gate
  so hai mask dropout khác nhau), không phải vì sm_75; chi tiết ở `tests.md` §2 và §3.4.
* **100c đắt gấp đôi mỗi round**, vì batch 256 cho gấp đôi số bước. Chênh lệch giữa 20c/50c và 100c
  vì thế không chỉ do số client.
* **Tiếp nối phiên:** cây run upload làm dataset của tài khoản chạy tiếp (vì `kernel_sources` không
  qua được ranh giới tài khoản), probe CPU kiểm `last == expected` trước khi push GPU. Đây là thủ
  tục chuẩn mà các dự án sau kế thừa.

## 5. Hợp đồng artifact

```
runs/fdids_<K>c_v2/
  weights/round_NNN.pt      chỉ trọng số: {round, model, cfg, fingerprint, metrics}
  resume/round_NNN.pt       round + RNG của driver (truy vết; không cần để tiếp tục — D8)
  confusion/round_NNN.npy   16×16, tổng == 10.761.343
  preds/round_NNN.u8.npy    nhãn dự đoán, đúng thứ tự test
  metrics/round_NNN.json    10 metric + ce/kd/gnorm/steps/applied/skipped/vram/seconds + per-class
  logs/round_NNN.json       từng client: n_k, rank, steps, applied, skipped, nonfinite, seed, ce, kd, gnorm
  reports/manifest.json     cfg hiệu lực, fingerprint, thứ tự cột, tên lớp, số dòng từng client
  history.csv               dẫn xuất từ metrics/*.json
  complete/round_NNN.done   ghi cuối cùng
```

* **Fingerprint 21 khoá** (`proj/ckpt.py::FINGERPRINT_KEYS`): kiến trúc, lr, λ, β, μ, T, clip,
  số client, batch, epoch, seed, `data_id`, `run_name`. `rounds` cố ý **không** nằm trong đó: lr
  hằng nên trọng số ở round r không phụ thuộc tổng số round.
* Dựng lại model ở bất kỳ round: `proj/ckpt.py::load_weights(..., expect_params=395_024)`,
  `strict=True`, kiểm số tham số, fingerprint và tính hữu hạn; đã đo `max|Δ logits| = 0,0`.
* `proj/verify.py` dựng lại 10 metric từ ma trận nhầm lẫn, và dựng lại ma trận từ `preds/`.
  `scripts/verify_run.py --require-rounds 50` pass ở mode full cho cả ba cấu hình.
* Trọng số, preds, resume và toàn bộ `runs/` **không** có trong git; bản gốc nằm ở output Kaggle
  và máy local.

## 6. Kiểm chứng

Suite local: **13 file test, 88 phép kiểm**, pass toàn bộ, luôn chạy qua watchdog RAM
`scripts/run_local_checked.py`. Ba vòng rà soát (R01–R18) đều đóng, mỗi mục có một ca chèn lỗi
riêng. Tổng 41 lỗi được ghi ở `tests.md` §2; #36–#38 chỉ lộ ra trên phần cứng thật. Validator
notebook kiểm 5 lớp: metadata, CFG qua AST, module nhúng ↔ `proj/*.py` **từng byte**, thứ tự
luồng thực thi, tên biến qua các cell.

## 7. Hình dạng kết quả phải biết trước khi đọc số

`f1_macro` **đạt đỉnh sớm** (round 6–9) rồi giữ một plateau nhiễu thấp hơn đỉnh 2–5 %; `accuracy`
và `f1_weighted` xói mòn 8–11 điểm, trong khi loss train của client giảm đơn điệu. Headline của
báo cáo là **round 50** theo đúng lịch đã chốt. Round đỉnh chỉ là quan sát hậu kiểm, vì không có
tập validation để chọn checkpoint. Giả thuyết "BatchNorm dưới FedAvg" (D6) **chưa được kiểm** và
không được trích như kết luận.

## 8. Caveat bắt buộc kèm mọi con số công bố

1. Split theo **thời gian mô phỏng**, không theo xe; test chỉ có scenario `_7`.
2. Mất cân bằng **41:1**: đọc `f1_macro`, không đọc `accuracy`.
3. **Rò rỉ Sybil:** 100 % dòng `trafficCongestionSybil` nằm trong flow toàn cùng nhãn.
4. `scaler.json` fit trên **toàn bộ** 43 M dòng train, tức trên dữ liệu mọi client gộp lại.
5. Đặc trưng lưu **fp16**.
6. Test **không** chia theo client; điểm test đo tổng quát hoá toàn cục.
7. **Không** đặt số của bản này cạnh số của bài báo (xem [`paper.md`](paper.md) §8).
8. Một run, một seed; lr hằng (§1).
