# KAGGLE.md — tài khoản, dữ liệu, kế hoạch phiên, và runbook vận hành

Mọi thứ liên quan đến việc **chạy** trên Kaggle. `CONTEXT.md` chỉ giữ bảng phân công tài khoản
và trỏ về đây. Cập nhật: **2026-09-22 08:10Z**.

---

## 1. Tài khoản

Tám tài khoản dùng chung với các dự án anh em (`~/.kaggle/accounts/`, mode 600; token **không**
nằm trong repo). `health` pass cả 8 (22-09 07:30Z). Quota GPU đọc **22-09 06:40Z** (đã trừ phiên
đang chạy của `perfedskd`; refresh **26-09 00:00Z**):

| tài khoản | quota còn | phân công dự án này | ghi chú |
|---|---:|---|---|
| `minhtran0601` | 22,48 h | **20c** (probe + production) | |
| `catbaochau` | 21,01 h | **50c** | perfedskd 20c đã xong trên tài khoản này |
| `trietbackup` | 19,12 h | **100c** | perfedskd 50c phiên 2 đang chạy (đến ~11:40Z 22-09) |
| `khanhmay0304` | 28,68 h | dự phòng | perfedskd cần ~9,3 h cho 100c phiên 3 |
| `minhtriethihi` | 18,31 h | dự phòng | perfedskd 100c phiên 2 đang chạy (đến ~17:20Z 22-09) |
| `odixe0502` | 6,36 h | chủ 4 dataset đầu vào | |
| `minhtrit06`, `odixeuit` | ~0 h | ⚠ **ưu tiên sau cùng** (chủ dự án, 21-09) | |

Chủ dự án chốt (22-09, câu 7): **mỗi kịch bản trọn trong một tài khoản**, `--max-hours 11.75`.

**Chính sách (chủ dự án, 2026-09-10):** agent tự đổi tài khoản (`kaggle_account.py use/ensure
--confirm`) nhưng phải nói rõ; push/kéo dưới tài khoản khác **không cần đổi** — dùng
`kaggle_as.py <user> -- kaggle ...`. ❌ Không bao giờ `kaggle auth login --force`.
❌ **Không dán token vào chat** (22-09: mọi KGAT từng dán đều nằm trong transcript trên đĩa).

**Giới hạn dịch vụ:** tối đa **2 GPU session đồng thời / tài khoản**, **12 h / session**,
`/kaggle/working` **bị xoá** khi session mới bắt đầu.

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
H=../.claude/skills/kaggle-training-notebook/scripts
python $H/kaggle_account.py list     # tài khoản nào có creds gì, ai đang active
python $H/kaggle_account.py quota    # giờ GPU còn lại (đã trừ phiên đang chạy), thời điểm refresh
python $H/kaggle_account.py health   # refresh token còn sống không
```

## 2. Dữ liệu

Bốn dataset **public** của `odixe0502` (`knowledge/KAGGLE_DATASETS.md`): `veremi-fl-{20,50,100}client`
+ `veremi-nextgen2026-centralized` (test + `scaler.json`). Mount đo được:
`/kaggle/input/datasets/odixe0502/<slug>/…`; notebook phân giải bằng sentinel
(`find_root("train/client_id=000")`, `find_root("upload/test")`), không hard-code.

Image: `knowledge/runtime.json` — digest đã kiểm cho 2×T4 (torch 2.10.0+cu128, Python 3.12).
Notebook `pip install --no-deps torch-pruning==1.6.1` ở cell đầu (cần `enable_internet`, vốn đã
bật cho W&B); validator chặn mọi lệnh cài lại `torch`.

## 3. Kế hoạch phiên

Dự báo trước probe ở [`rebuild.md` §5](rebuild.md): 20c/50c ≈ 4–5 h, 100c ≈ 8–9 h ⇒ **mỗi kịch
bản một phiên 11,75 h**. Nếu 100c không kịp: phiên 2 cùng tài khoản qua `--session 2
--require-resume --kernel-source trietbackup/lightweight-fed-nids-veremi-100-clients`
(driver tự dừng trước hạn, `finalize_reserve_seconds = 900`). Số đo thật: [`TESTS.md` §4](TESTS.md).

## 4. Runbook — phiên đầu của một kịch bản

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
H=../.claude/skills/kaggle-training-notebook/scripts; NB=papers/lwfednids-bouayad-2024/notebook
# 1. sinh notebook
python scripts/gen_notebook.py --owner minhtran0601 --clients 20 --max-hours 11.75
# 2. gate tĩnh (metadata, digest image, module nhúng khớp proj/ từng byte, CFG, thứ tự cell)
python scripts/validate_notebooks.py
# 3. nhúng key W&B — notebook PHẢI là is_private: true; key sẽ VĨNH VIỄN trong version history
python $H/embed_wandb_key.py $NB/20c/lwfednids_20c.ipynb --metadata $NB/20c/kernel-metadata.json
# 4. validate LẠI (bước 3 sửa nội dung cell)
python scripts/validate_notebooks.py
# 5. push, rồi ghi lại đã push (validator coi thư mục có PUSHED là đã đóng băng)
python $H/kaggle_as.py minhtran0601 -- kaggle kernels push -p $NB/20c
echo "pushed $(date -u +%FT%TZ) v1 minhtran0601/lightweight-fed-nids-veremi-20-clients" > $NB/20c/PUSHED
```

Probe (2 vòng + cell calibration đo TA/IA so với mô hình chưa cắt): thêm `--probe --max-hours 2`;
thư mục `20c_probe`, run_name `lwfednids_20c_probe`, slug `…-probe`.

## 5. Runbook — phiên nối tiếp

```bash
# cùng tài khoản: output kernel trước làm nguồn
python scripts/gen_notebook.py --owner trietbackup --clients 100 --session 2 --require-resume \
    --kernel-source trietbackup/lightweight-fed-nids-veremi-100-clients --max-hours 11.75
# rồi validate → embed → validate → push -p $NB/100c_s2

# khác tài khoản: kéo output → (merge) → đóng gói bundle → dataset của tài khoản mới → probe CPU
python scripts/stage_ckpt_dataset.py <run_dir> --owner <acct> --slug lwfednids-100c-ckpt-s3 \
    --out papers/lwfednids-bouayad-2024/runs/ckpt_ds/100c_s3 --last-only
python $H/kaggle_as.py <acct> -- kaggle datasets create -p papers/lwfednids-bouayad-2024/runs/ckpt_ds/100c_s3 -r zip -t
python scripts/gen_ckpt_probe.py --owner <acct> --dataset <acct>/lwfednids-100c-ckpt-s3 \
    --run-name lwfednids_100c --expect-round <r> --out $NB/100c_ckpt_probe
# probe CPU: 0 GPU, ~1 phút, PHẢI in "PROBE_OK lwfednids_100c <r>" trước khi push kernel GPU
python scripts/gen_notebook.py --owner <acct> --clients 100 --session 3 --require-resume \
    --dataset-source <acct>/lwfednids-100c-ckpt-s3 --max-hours 11.75
```

## 6. Runbook — theo dõi, kéo, verify

```bash
# theo dõi (một dòng khi trạng thái/số hàng W&B đổi; thoát khi mọi kernel terminal)
bash scripts/poll_prod.sh 300 minhtran0601/lightweight-fed-nids-veremi-20-clients:lwfednids_20c
python scripts/watch_prod.py lwfednids_20c --kernel minhtran0601/lightweight-fed-nids-veremi-20-clients
# kéo (retry, watchdog byte mạng) → verify offline
bash scripts/pull_output.sh minhtran0601 minhtran0601/lightweight-fed-nids-veremi-20-clients \
    papers/lwfednids-bouayad-2024/runs/pulls/20c
python scripts/verify_run.py papers/lwfednids-bouayad-2024/runs/pulls/20c/runs/lwfednids_20c --require-rounds 50
# nhiều phiên → một cây
python $H/merge_sessions.py --out papers/lwfednids-bouayad-2024/runs/merged/lwfednids_100c \
    papers/lwfednids-bouayad-2024/runs/pulls/100c_s1 papers/lwfednids-bouayad-2024/runs/pulls/100c_s2
```

**Dừng run khi** (đọc W&B): `summary.backend = eager` (chậm ~2–3×, không đúng ngân sách);
`skipped` **tăng dần** theo vòng (không phải ~1–2/client hằng); `f1_macro` về 0 hoặc NaN;
`train_sec` +30 % so với vòng trước mà không có lý do; `vram_train_gb` > 14; kernel ERROR.
Cách dừng: `kaggle_as.py <acct> -- kaggle kernels status`, rồi cancel qua MCP
`cancel_notebook_session` hoặc push bản thay thế (xem skill `references/stop-session.md`).

## 7. Bẫy vận hành đã biết (kế thừa từ các dự án anh em)

* Kaggle lấy slug từ **title**, bỏ qua `id` nếu lệch → validator so hai bên.
* `nb.metadata["kaggle"]` thiếu ⇒ **0 GPU** dù `kernel-metadata.json` đúng.
* Zero GPU có thể là **theo tài khoản**: chạy lại nguyên xi trên tài khoản khác trước khi sửa code.
* `kernel_sources` **không** mount output của tài khoản khác — im lặng; muốn hop phải qua dataset.
* CLI `kaggle kernels output` chết/treo giữa chừng → `pull_output.sh` có retry + watchdog byte mạng.
* Stdout của kernel đang chạy **không** tải được → W&B là mắt duy nhất trong lúc chạy.
* `conda run -n nckh python - <<PY` **nuốt stdout** — ghi script ra file rồi chạy.
* Watchdog RAM local cần ≥ ~3,5 GiB `MemAvailable`; VS Code server chiếm ~2 GB.

## 8. Kernel đã push

| ngày | tài khoản | kernel | vai trò | trạng thái |
|---|---|---|---|---|
| 22-09 07:32Z | `minhtran0601` | `lightweight-fed-nids-veremi-20-clients-probe` v1 | probe 2 vòng + calibration | **COMPLETE** 08:01Z, kéo về + verify pass (`TESTS.md` §4) |
| 22-09 08:02Z | `minhtran0601` | `lightweight-fed-nids-veremi-20-clients` v1 | production 20c, `--max-hours 11.75` | **COMPLETE 50/50, 3,62 h**, kéo về + verify pass ([`report.md`](report.md)) |
| 22-09 08:02Z | `catbaochau` | `lightweight-fed-nids-veremi-50-clients` v1 | production 50c, `--max-hours 11.75` | **COMPLETE 50/50, 3,38 h**, kéo về + verify pass ([`report.md`](report.md)) |
| 22-09 08:02Z | `trietbackup` | `lightweight-fed-nids-veremi-100-clients` v1 | production 100c, `--max-hours 11.75` | **COMPLETE 50/50 trong 1 phiên, 6,90 h**, kéo về + verify pass ([`report.md`](report.md)) |
