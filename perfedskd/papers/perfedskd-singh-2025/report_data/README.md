# Số liệu cho REPORT.md — PerFed-SKD trên VeReMi / DAGSNet

Sinh tự động bởi `scripts/report_data.py` từ artifact đã kéo về, ghép phiên và verify local; không có số nào lấy từ W&B. Metric **cá nhân hoá** là trung bình không trọng số trên N model ω_m, mỗi model đo trên **toàn bộ 10 761 343 dòng test toàn cục**; metric **global** là của một model ω^t (server) trên cùng tập test. Caveat bắt buộc: `docs/rebuild.md` §2 (D1–D13) và §8.

## 1. Tình trạng run

| kịch bản | client | round | trạng thái | phiên Kaggle | round-time trung vị | tổng giờ round |
|---|---:|---:|---|---|---:|---:|
| 20c | 20 | **50/50** | HOÀN TẤT | 1 phiên (50 round) | 9.9 phút (train 5.8 + eval 4.1) | 8.3 h |
| 50c | 50 | **50/50** | HOÀN TẤT | 2 phiên (34 round, 16 round) | 18.1 phút (train 5.9 + eval 12.1) | 14.9 h |
| 100c | 100 | **50/50** | HOÀN TẤT | 3 phiên (18 round, 20 round, 12 round) | 33.1 phút (train 9.7 + eval 22.7) | 28.5 h |

## 2. Round cuối — đủ 10 metric (mean trên client; std/min/max của f1_macro; server aggregate ω^t)

accuracy = precision_micro = recall_micro = recall_weighted = f1_micro (đồng nhất thức đa lớp đơn nhãn).

| metric | 20c ω_m (r50) | 20c ω^t | 50c ω_m (r50) | 50c ω^t | 100c ω_m (r50) | 100c ω^t |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.6229 | 0.5539 | 0.6619 | 0.6735 | 0.6354 | 0.6627 |
| precision_macro | 0.7022 | 0.6887 | 0.7169 | 0.7198 | 0.6774 | 0.6685 |
| precision_micro | 0.6229 | 0.5539 | 0.6619 | 0.6735 | 0.6354 | 0.6627 |
| precision_weighted | 0.7446 | 0.7268 | 0.7548 | 0.7564 | 0.7228 | 0.7210 |
| recall_macro | 0.7114 | 0.7534 | 0.7391 | 0.7743 | 0.7060 | 0.7492 |
| recall_micro | 0.6229 | 0.5539 | 0.6619 | 0.6735 | 0.6354 | 0.6627 |
| recall_weighted | 0.6229 | 0.5539 | 0.6619 | 0.6735 | 0.6354 | 0.6627 |
| f1_macro | 0.6504 | 0.6616 | 0.6901 | 0.7202 | 0.6541 | 0.6871 |
| f1_micro | 0.6229 | 0.5539 | 0.6619 | 0.6735 | 0.6354 | 0.6627 |
| f1_weighted | 0.6077 | 0.5343 | 0.6565 | 0.6658 | 0.6315 | 0.6588 |
| f1_macro std / min / max | 0.0485 / 0.5578 / 0.7689 | — | 0.0295 / 0.6320 / 0.7443 | — | 0.0393 / 0.5323 / 0.7453 | — |
| client yếu nhất / mạnh nhất (f1_macro) | #0 0.558 / #10 0.769 | — | #23 0.632 / #16 0.744 | — | #56 0.532 / #99 0.745 | — |

## 3. Hình dạng đường cong (round 1 → đỉnh hậu kiểm → cuối)

Đỉnh là quan sát *sau khi chạy*, không phải checkpoint được chọn; số công bố là round cuối.

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| f1_macro ω_m round 1 | 0.5153 | 0.4256 | 0.3652 |
| f1_macro ω_m đỉnh (round) | 0.6504 (r50) | 0.6901 (r50) | 0.6541 (r50) |
| f1_macro ω_m cuối | 0.6504 | 0.6901 | 0.6541 |
| chênh so đỉnh | −0.0 % | −0.0 % | −0.0 % |
| accuracy ω_m round 1 → cuối | 0.5790 → 0.6229 | 0.5127 → 0.6619 | 0.4594 → 0.6354 |
| f1_macro ω^t round 1 → 2 → cuối | 0.2039 → 0.5491 → 0.6616 | 0.2222 → 0.5853 → 0.7202 | 0.2166 → 0.5489 → 0.6871 |
| accuracy ω^t round 1 → cuối | 0.3761 → 0.5539 | 0.4718 → 0.6735 | 0.4464 → 0.6627 |

## 4. Thành phần loss (trung bình client, các bước áp dụng) — round 1 / 2 / cuối

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| loss | 1.287 / 0.537 / 0.061 | 1.305 / 0.556 / 0.125 | 1.288 / 0.535 / 0.159 |
| ce | 0.802 / 0.381 / 0.039 | 0.821 / 0.392 / 0.081 | 0.798 / 0.369 / 0.096 |
| kd | 0.485 / 0.156 / 0.022 | 0.485 / 0.165 / 0.045 | 0.490 / 0.166 / 0.063 |
| gnorm | 0.501 / 0.507 / 0.590 | 0.533 / 0.581 / 1.588 | 0.612 / 0.644 / 2.209 |

## 5. Chọn thiết bị và truyền tin (quy tắc τ của PerFed-SKD)

|S_t| = số device được chọn ở round t (round 1 = M theo định nghĩa); comm_saving = 1 − |S_t|/M của chi phí giao thức (|S_t| lên + |S_t| xuống, **tính**, không đo — D12).

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| \|S\| mean (min–max, bỏ round 1) | 7.4 (6–9) / 20 | 20.4 (17–24) / 50 | 46.3 (40–55) / 100 |
| \|S\|/M mean (bỏ round 1) | 0.369 | 0.408 | 0.463 |
| comm_saving mean (mọi round) | 0.618 | 0.580 | 0.526 |
| truyền lên / xuống tổng (MiB) | 580 / 580 | 1595 / 1595 | 3601 / 3601 |
| τ round 1 → cuối | 0.5790 → 0.6229 | 0.5127 → 0.6619 | 0.4594 → 0.6354 |

## 6. Kiểm soát chất lượng pipeline

| | 20c | 50c | 100c |
|---|---:|---:|---:|
| backend train | compiled | compiled | compiled |
| evaluated = N mọi round | có | có | có |
| bước/round, skip AMP tổng (max một round) | 84083, 1192 (32) | 84098, 943 (41) | 168200, 2523 (102) |
| VRAM train max (GiB) | 7.08 | 7.18 | 7.16 |
| fingerprint | `90566629a718b902` | `20ffa11c1e56036c` | `f923c7feb193d427` |
| torch / CUDA | 2.10.0+cu128 / 12.8 | 2.10.0+cu128 / 12.8 | 2.10.0+cu128 / 12.8 |

## 7. File

| file | nội dung |
|---|---|
| `rounds_Kc.md`, `history_Kc.csv` | **mọi round**, đủ 10 metric (mean) + std/min/max, global, |S|, tau, skip, lr, train/eval giây — bảng bắt buộc của report |
| `per_client_final_Kc.csv` | 10 metric của từng client ở round cuối |
| `per_class_final_Kc.csv` | mỗi lớp ở round cuối: support, P/R/F1 trung bình trên client, trên confusion gộp, và của ω^t |
| `confusion_final_Kc.csv`, `confusion_global_Kc.csv` | confusion gộp (tổng N client) và của ω^t, round cuối, 16×16 |
| `summary.json` | mọi số ở trên dạng máy đọc + cấu hình từ `reports/manifest.json` + provenance phiên |
| `convergence.png` | f1_macro / accuracy (mean ± std, nét liền) và server aggregate ω^t (nét đứt) theo round, cả ba K; lịch LR |
| `skd_loss.png` | ce và kd (Eq. 2) trung bình client theo round |
| `client_spread.png` | phân bố f1_macro theo client ở round cuối |
| `per_class_f1_Kc.png`, `confusion_Kc.png` | F1 từng lớp (theo tỉ lệ test) và confusion gộp chuẩn hoá theo hàng |
| `selection.png` | \|S_t\|/M và τ_t theo round |
| `round_time.png` | phút train / eval mỗi round trên 2×T4 |
