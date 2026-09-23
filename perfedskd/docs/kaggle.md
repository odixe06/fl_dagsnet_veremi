# kaggle.md — tài khoản, dữ liệu, kế hoạch phiên, và runbook vận hành

Mọi thứ liên quan đến việc **chạy** trên Kaggle. `CONTEXT.md` chỉ giữ bảng phân công tài khoản
và trỏ về đây. Cập nhật: **2026-09-23 02:10Z**.

---

## 1. Tài khoản

**Tám tài khoản, `health` pass toàn bộ, quota đọc lúc 2026-09-21 ~12:45Z (refresh 26-09 00:00Z):**

| tài khoản | email | quota GPU | vai trò |
|---|---|---:|---|
| `catbaochau` | thismailuse4chill@gmail.com | 30,00 h | **20c** (probe + production) |
| `trietbackup` | minhtriet0502.backup@gmail.com | 30,00 h | **50c** |
| `minhtriethihi` | — | 30,00 h | **100c** (cả ba phiên, kế hoạch A) |
| `khanhmay0304` | — | 30,00 h | probe 100c; **tài khoản tràn** cho 100c |
| `minhtran0601` | minhtriet0502.work@gmail.com | 22,48 h | dự phòng |
| `odixe0502` | — | 19,04 h | **chủ 4 dataset đầu vào**; dự phòng |
| `minhtrit06` | — | 3,65 h | ⚠ **ưu tiên sau cùng** (chủ dự án, 21-09) |
| `odixeuit` | — | 0,12 h | ⚠ **ưu tiên sau cùng** (chủ dự án, 21-09) |

`catbaochau` và `trietbackup` là hai tài khoản thêm ngày 21-09: token MCP đã **introspect trên
server** (`active=True`, username khớp), và chủ dự án đã browser-login nên cả hai có snapshot
OAuth đầy đủ ⇒ `quota`/`health`/`plan`/`use`/`ensure` dùng được. Token MCP của `minhtran0601`
được thay mới cùng ngày.

**Đọc lại 21-09 16:50Z:** `minhtriethihi` 30,00 (trước khi push 100c) · `khanhmay0304` 28,68
(probe 100c tốn 1,32 h) · `trietbackup` 26,44 · `catbaochau` 25,89. Refresh vẫn 26-09 00:00Z.

**Đọc lại 22-09 02:12Z** (20c xong, 50c phiên 1 xong, 100c phiên 1 đang chạy giờ 9,3):
`khanhmay0304` 28,68 · `minhtran0601` 22,48 · `catbaochau` **21,01** (20c tốn 8,99 h) ·
`minhtriethihi` **20,73** (đang trừ dần) · `trietbackup` **19,53** (50c phiên 1 tốn 10,47 h) ·
`odixe0502` 6,36 · `minhtrit06` 2,68 · `odixeuit` 0,07.

**Đọc lại 23-09 01:20Z** (20c, 50c xong; 100c phiên 2 xong, trước khi push phiên 3):
`khanhmay0304` **28,68** · `minhtran0601` 18,40 · `catbaochau` 17,55 · `minhtriethihi` **7,72** ·
`trietbackup` **7,70** · `odixe0502` 6,36 · `minhtrit06` 0,00 · `odixeuit` 0,00.
(`minhtran0601` và `catbaochau` giảm ~3,5–4 h so với 22-09 mà dự án này không dùng — tài khoản
dùng chung với dự án khác.)

**Tài khoản CLI đang active: `trietbackup`** (do `add-account` để lại).

⚠ **Không ghi token vào bất kỳ file nào trong repo.** Token ở `~/.kaggle/accounts/<user>.mcp-token`
(mode 600), snapshot OAuth ở `~/.kaggle/accounts/<user>.credentials.json`, thư mục mode 700.

### 1.1 Token — xoay vòng 22-09

- **`odixe0502`**: token MCP mới được chủ dự án cấp 22-09 ~02:10Z, đã introspect trên server
  (`active=True`, username khớp) và lưu vào `odixe0502.mcp-token` (600). Token cũ (03-09) introspect
  `active=False` — đã bị thu hồi phía Kaggle trước đó.
- **Phát hiện rò rỉ (22-09):** quét byte `~/.claude`, `~/.vscode-server`, `~/nckh` thấy **giá trị
  KGAT của cả 8 tài khoản** nằm trong transcript Claude Code (`~/.claude/projects/*/*.jsonl`, và
  log extension VS Code sao chép chúng) — vì mỗi token từng được **dán vào chat** khi thêm tài khoản
  — và **refresh token của `minhtran0601`** trong một transcript (`edl-cmso`). Cả `.mcp.json` lẫn
  `.vscode/mcp.json` của repo này còn mode 644 (đã siết về 600); các repo anh em trong
  `fl_dagsnet_veremi1/` cũng mang bearer cũ.
- **Xoay vòng là việc của người**: KGAT mới tạo trên kaggle.com, refresh token mới chỉ có qua
  `kaggle auth login --force` (browser, đúng tài khoản). Agent không tự làm được; công cụ:
  `scripts/rotate_kaggle_creds_wizard.sh` (10 stage: preflight → 8 tài khoản → khôi phục active +
  resync host config) gọi `scripts/rotate_kaggle_creds.py`, nhập token bằng **input ẩn**, introspect
  giá trị mới, cài đặt 600, **thu hồi giá trị cũ qua API** (`POST /api/v1/tokens/revoke` — chính
  lời gọi sau `kaggle auth revoke`), rồi ghi lại mọi `.mcp.json`/`.vscode/mcp.json` dưới `~/nckh`.
  ❌ Không dùng `KaggleCredentials.revoke_token()`: `delete()` của nó xoá `~/.kaggle/credentials.json`
  (login đang sống) bất kể snapshot nào được nạp.
- **Từ nay không dán token vào chat** — ghi thẳng vào file hoặc chạy wizard. Transcript cũ chứa
  token chết sau khi xoay; muốn xoá khỏi đĩa thì bảo agent, đó là thao tác xoá dữ liệu của chủ dự án.

**Chính sách (chủ dự án, 2026-09-10):** agent **tự** đổi tài khoản
(`kaggle_account.py use/ensure --confirm`) nhưng **phải nói rõ đã đổi sang đâu và vì sao**;
push/kéo dưới tài khoản khác **không cần đổi** — dùng `kaggle_as.py <user> -- kaggle ...`.
❌ Không bao giờ `kaggle auth login --force` (phá refresh token không lấy lại được).

**Giới hạn dịch vụ:** tối đa **2 GPU session đồng thời / tài khoản**, **12 h / session**,
`/kaggle/working` **bị xoá** khi session mới bắt đầu.

```bash
H=../.claude/skills/kaggle-training-notebook/scripts
python $H/kaggle_account.py list     # tài khoản nào có creds gì, ai đang active
python $H/kaggle_account.py quota    # giờ GPU còn lại, thời điểm refresh
python $H/kaggle_account.py health   # refresh token còn sống không (cảnh báo sớm)
python $H/kaggle_account.py plan --gpu-hours 28 --session-hours 11
```

---

## 2. Dữ liệu

**Bốn dataset đầu vào đều PUBLIC**, chủ `odixe0502` ⇒ mọi tài khoản mount được, **không cần
share tay** (`knowledge/kaggle-datasets.md`).

| kịch bản | Kaggle | local (WSL) |
|---|---|---|
| 20 client | [`odixe0502/veremi-fl-20client`](https://www.kaggle.com/datasets/odixe0502/veremi-fl-20client) | `~/nckh/veremi/dataset/fl_client/alpha05/20_client/train/client_id=NNN/` |
| 50 client | [`odixe0502/veremi-fl-50client`](https://www.kaggle.com/datasets/odixe0502/veremi-fl-50client) | `…/alpha05/50_client/…` |
| 100 client | [`odixe0502/veremi-fl-100client`](https://www.kaggle.com/datasets/odixe0502/veremi-fl-100client) | `…/alpha05/100_client/…` |
| test toàn cục | [`odixe0502/veremi-nextgen2026-centralized`](https://www.kaggle.com/datasets/odixe0502/veremi-nextgen2026-centralized), thư mục `upload/test/` | `~/nckh/veremi/dataset/centralized/test/` |

Từ Windows: `\\wsl.localhost\Ubuntu\home\odixe\nckh\veremi\dataset\fl_client` và
`\\wsl.localhost\Ubuntu\home\odixe\nckh\veremi\dataset\centralized\test`.

⚠ **Không hard-code tiền tố mount.** Đường dẫn thật là
`/kaggle/input/datasets/<owner>/<slug>/…`, không phải `/kaggle/input/<slug>/`.
`proj/data.py::find_root` phân giải bằng **sentinel** (`train/client_id=000`, `upload/test`).

⚠ **train/ ĐÃ z-score, test/ thì CHƯA.** Áp `scaler.json` **chỉ** cho test, **đúng một lần**.
Cả hai lỗi ngược nhau đều **không báo lỗi gì** (`knowledge/dataset.md` §1.1).

---

## 3. Kế hoạch phiên

Giờ dưới đây suy từ **round thật** của production 20c/50c và probe 100c
([`tests.md` §4.2–4.3](tests.md)), không phải từ micro-benchmark:

| kịch bản | tài khoản | phiên | `--max-hours` | round | giờ | trạng thái |
|---|---|---|---|---|---:|---|
| 20c | `catbaochau` | 1 | 11 | 1–50 | 8,42 (đo) | **COMPLETE** 50/50, **verify pass** — xong hẳn |
| 50c | `trietbackup` | 2 | 11 · 6,5 | 1–34 · 35–50 (đo) | 10,46 + ~4,9 (đo) | **COMPLETE 50/50**, merge s1+s2, **verify 50 pass** — xong hẳn |
| 100c | `minhtriethihi` → `khanhmay0304` | 3 | 11,75 · 11,75 · 11,75 | 1–18 · 19–38 · 39–50 (đo) | 11,19 + 11,09 + 6,75 (đo) | **COMPLETE 50/50**, merge s1+s2+s3, **verify 50 pass** — xong hẳn |

**100c sau phiên 2 (23-09):** phiên 2 chạy **20 round (19–38)**, không phải 18: khởi động chỉ
**709 s** (import + compile; phiên 1: 2.560 s) và round ổn định **1.930 s** (eval 1.358 s — khớp
probe; phiên 1 eval 1.600 s ⇒ chênh là do máy được cấp). Driver dừng đúng cổng sau r38 (10,97 h;
kernel tổng 11,09 h). Còn **12 round**: 12 × 1.930–2.250 s + khởi động 12–43 phút ≈ **7,2–8,2 h**.
`minhtriethihi` còn 7,72 h ⇒ đi cùng tài khoản quá sát trần (hết quota giữa chừng = mất cả output)
⇒ giữ quyết định hop sang `khanhmay0304` (28,68 h), `--max-hours 11.75`, vẫn **3 phiên**.

**Sửa kế hoạch 100c, 22-09 02:30Z — theo quy tắc quyết định bên dưới, hàng "> 36 phút".**
Production 100c chậm hơn probe: khởi động **2.560 s** (43 phút, probe ước 30), round 1 **2.345 s**,
round ổn định **2.199 s median / 2.249 s max = 36,7 phút** (probe 1.974 s = 32,9; eval 1.586 s
so với 1.359 s — chênh nằm ở eval, `train_sec` 611 s khớp probe nên không phải nghẽn CPU và không
phải lý do dừng). Cổng driver (reserve 900 s) cho **17 round/phiên** (round 17 kết thúc ~11,1 h),
không phải 19 ⇒ sau hai phiên còn **16 round × 36,7 phút + 43 phút ≈ 10,5 h**, trong khi
`minhtriethihi` sau phiên 2 chỉ còn ≈ 30 − 2 × 11,3 ≈ **7,3 h**. Kế hoạch A (một tài khoản
gánh cả ba phiên) **không còn khả thi**; theo hàng 3 của bảng quyết định: phiên 3 chạy trên
**`khanhmay0304`** (28,68 h) qua **dataset checkpoint + probe CPU** (§5), `--max-hours 11.75`
(10,5 h ước tính ≤ 11,75 h, một phiên vẫn đủ ⇒ **vẫn 3 phiên**, không tăng số phiên). Phiên 2
vẫn `minhtriethihi`, `--max-hours 11.75`, đường `--kernel-source` cùng tài khoản.

**Đã chốt với chủ dự án 21-09 ~17:00Z**, sau round thật 100c = 32,9 phút (kế hoạch A theo quy
tắc bên dưới) và ba yêu cầu mới của chủ dự án:

1. `--max-hours` **được > 11**, miễn là **không chạm trần 12 h** ⇒ phiên 1–2 của 100c dùng
   **11,75**. Với khởi động ~30 phút, round đầu 2.118 s, round ổn định 1.974 s: cổng driver
   (dừng khi `elapsed + 1,15 × round_chậm_nhất + 900 s > max`) cho **19 round**, round cuối kết
   thúc ~11,0 h, còn ≥ 1 h cho cell verify + Kaggle lưu ~3 GB (xấu nhất ~25 phút). 11,9 h chỉ
   thêm được 1 round khi khởi động ≤ 31 phút và chỉ cách trần 20–29 phút — **không chọn**.
2. **Tối thiểu số phiên.** Không giảm được: 25 round × 32,9 phút = 13,7 h > 12 h ⇒ **3 phiên là
   sàn** của 100c (50c: 2, 20c: 1). Kéo dài ngân sách chỉ dời round giữa các phiên, **không**
   tiết kiệm quota vì tổng công việc không đổi.
3. **Biên quota tổng chỉ cần 5–10 phút.** Ước tính 100c ≈ 29,2 h / 30 h (biên ~45 phút với
   khởi động 30 phút/phiên; ~20 phút nếu khởi động 40 phút) ⇒ **một tài khoản gánh cả ba
   phiên**. Khi vượt: quota hết giữa chừng ⇒ Kaggle giết session ⇒ **mất cả output phiên đó**,
   nên phiên 3 **phải** khai `--max-hours = quota còn lại − 0,15` để driver tự dừng thay vì bị
   giết; round còn dư (nếu có) đi đường dataset checkpoint sang `khanhmay0304` (§5) hoặc chờ
   refresh 26-09 00:00Z.

**Số round một phiên** = 1 + max{m : S + T₁ + (m−1)·T + 1,15·T₁ + 900 ≤ max_seconds} với S
khởi động, T₁ round đầu phiên, T round ổn định — tính trước khi chọn `--max-hours`.

Phiên nối tiếp **cùng tài khoản** ⇒ đường đơn giản
`--session N --require-resume --kernel-source <acct>/<slug>`; **không** cần handoff bundle,
**không** cần probe CPU, **không** cần đổi tài khoản.

**Quy tắc quyết định sau khi có round thật của probe** (đã áp dụng 21-09: 32,9 phút ⇒ hàng đầu):

| round 100c đo được | 50 round | hành động |
|---|---:|---|
| ≤ 33,5 phút | ≤ 28,0 h | kế hoạch A, `minhtriethihi` gánh cả ba phiên |
| 33,5 – 36 phút | 28 – 30 h | vẫn kế hoạch A nhưng **biên < 2 h**: chuẩn bị `khanhmay0304` làm tài khoản tràn và stage handoff bundle sau phiên 2 |
| > 36 phút | > 30 h | lên kế hoạch hop **từ đầu**: phiên 1–2 `minhtriethihi`, phiên 3 `khanhmay0304` qua dataset checkpoint + probe CPU (§5) |

⚠ **`rounds = 50` nằm trong fingerprint** (lịch cosine trải cả run) ⇒ **mọi phiên của một run
phải khai cùng T = 50**. Phiên nối tiếp khai khác sẽ bị chặn ở gate resume (đã kiểm ở
`tests/test_smoke_real.py`).

✔ Notebook 100c sinh lại với `--max-hours 11.75`, key nhúng, validate pass ×2, push 16:57Z 21-09,
RUNNING sau 45 s.

**Dung lượng output:** một round lưu (M+1) × 1,58 MB trọng số ⇒ 20c 33 MB, 50c 81 MB,
100c 160 MB mỗi round; 50 round = 1,7 / 4,0 / 8,0 GB, cộng `preds` 1,08 GB ở round 50 của 100c.

---

## 4. Runbook — phiên đầu của một kịch bản

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate nckh
cd ~/nckh/veremi/perfedskd
H=../.claude/skills/kaggle-training-notebook/scripts
NB=papers/perfedskd-singh-2025/notebook

# 1. sinh notebook
python scripts/gen_notebook.py --owner <acct> --clients <K> --max-hours 11
# 2. gate tĩnh (metadata, digest image, module nhúng khớp proj/, giá trị CFG, thứ tự cell)
CUDA_VISIBLE_DEVICES="" python scripts/validate_notebooks.py
# 3. nhúng key W&B — notebook PHẢI là is_private: true
python $H/embed_wandb_key.py $NB/<K>c/perfedskd_<K>c.ipynb --metadata $NB/<K>c/kernel-metadata.json
# 4. validate LẠI (bước 3 sửa nội dung cell)
CUDA_VISIBLE_DEVICES="" python scripts/validate_notebooks.py
# 5. push, rồi ghi lại đã push
python $H/kaggle_as.py <acct> -- kaggle kernels push -p $NB/<K>c
date -u +"pushed %FT%TZ" > $NB/<K>c/PUSHED
```

⚠ **Key W&B nhúng vào notebook là VĨNH VIỄN trong version history của Kaggle.** Uỷ quyền của
chủ dự án (2026-09-07) chỉ áp dụng cho notebook **`is_private: true`** của chính họ. `.ipynb`
nằm trong `.gitignore` và mode 600 — **không commit, không chia sẻ**. Muốn thu hồi key thì
**revoke ở W&B trước**; dựng lại notebook ở local **không** xoá được gì trên Kaggle.

## 5. Runbook — phiên nối tiếp

**Cùng tài khoản** (đường đơn giản, không cần probe):

```bash
python scripts/gen_notebook.py --owner <acct> --clients <K> --session 2 --require-resume \
    --kernel-source <acct>/perfed-skd-veremi-<K>-clients --max-hours 11
# rồi validate → embed → validate → push -p $NB/<K>c_s2
```

**Khác tài khoản** — `kernel_sources` **không** qua được ranh giới tài khoản: push **báo thành
công**, kernel chạy với `/kaggle/input` **rỗng**, và không có exception nào. Phải đi đường
dataset checkpoint:

```bash
P=papers/perfedskd-singh-2025/runs
bash scripts/pull_output.sh A A/<slug-phiên-N> $P/pulls/<K>c_sN          # chờ PULL_OK
rm -rf $P/merged/perfedskd_<K>c && python $H/merge_sessions.py \
    --out $P/merged/perfedskd_<K>c $P/pulls/<K>c_s1 … $P/pulls/<K>c_sN
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python scripts/verify_run.py \
    $P/merged/perfedskd_<K>c --require-rounds <r>
CUDA_VISIBLE_DEVICES="" python scripts/stage_ckpt_dataset.py $P/merged/perfedskd_<K>c \
    --owner B --slug perfedskd-<K>c-ckpt-sN --out $P/ckpt_ds/<K>c_sN --last-only
python $H/kaggle_as.py B -- kaggle datasets create -p $P/ckpt_ds/<K>c_sN -r zip -t
# probe CPU: 0 GPU, ~1 phút, PHẢI in "PROBE_OK perfedskd_<K>c <r>" trước khi push kernel GPU
python scripts/gen_ckpt_probe.py --owner B --dataset B/perfedskd-<K>c-ckpt-sN \
    --run-name perfedskd_<K>c --expect-round <r> --out $NB/<K>c_ckpt_probe_sN
python $H/kaggle_as.py B -- kaggle kernels push -p $NB/<K>c_ckpt_probe_sN
python scripts/gen_notebook.py --owner B --clients <K> --session N+1 --require-resume \
    --dataset-source B/perfedskd-<K>c-ckpt-sN --max-hours 11
```

⚠ `kaggle datasets create` **phải** có `-r zip -t`: mặc định `--dir-mode skip` **bỏ hết thư
mục**, và chuyển đổi tabular **ghi lại mọi CSV** (lỗi này chỉ lộ ra nhiều giờ sau, ở bước đối
chiếu hash giữa lúc resume). Stage vào `<out>/runs/<run_name>/…` vì Kaggle **bỏ một cấp thư
mục** khi giải nén; mount tại `/kaggle/input/datasets/<owner>/<slug>/<run_name>/…`.

## 6. Runbook — theo dõi, kéo, verify

```bash
python $H/kaggle_as.py <acct> -- kaggle kernels status <owner>/<slug>
bash scripts/poll_prod.sh 300 <owner>/<slug>:perfedskd_<K>c     # trạng thái + hàng W&B
bash scripts/pull_output.sh <acct> <owner>/<slug> $P/pulls/<name>   # chờ PULL_OK
CUDA_VISIBLE_DEVICES="" python scripts/run_local_checked.py python scripts/verify_run.py \
    $P/merged/perfedskd_<K>c --require-rounds 50
python scripts/report_data.py --out papers/perfedskd-singh-2025/report_data \
    20c=$P/merged/perfedskd_20c 50c=$P/merged/perfedskd_50c 100c=$P/merged/perfedskd_100c
```

**W&B:** entity `21522798-uit`, project `perfedskd-veremi`, run id = `run_name`
(`perfedskd_{K}c`). Log của kernel Kaggle **không đọc được khi đang chạy** — W&B là cửa sổ duy
nhất vào một run đang sống. W&B chỉ nhận **mean/std/min/max** qua client; **accuracy từng
client mỗi round** (thứ nuôi τ) chỉ có trong output kernel sau khi phiên kết thúc:
`metrics/round_XXX.json` (`clients[]`, `selected`, `selected_next`, `tau`) và `clients.csv`
(round × cid × 10 metric). Kéo **riêng** các file nhỏ này, không kéo trọng số:

```bash
python $H/kaggle_as.py <acct> -- kaggle kernels output <owner>/<slug> -p <dest> --page-size 200 \
    --file-pattern '.*(clients\.csv|history\.csv|round_[0-9]{3}\.json|calibration\.json|manifest\.json)$'
```
(880 KB cho probe 100c thay vì ~1,5 GB.) `proj/verify.py` dựng lại τ và S_{t+1} từ đúng các file này.

**Dấu hiệu PHẢI dừng run:**

| dấu hiệu | vì sao |
|---|---|
| `backend = eager` (train hoặc eval) | chậm **gần 7×** — xem [`tests.md` §4](tests.md) |
| `skipped` tăng dần theo round | GradScaler không hội tụ, không phải calibration |
| `f1_macro` về 0 hoặc NaN | phân kỳ |
| `train_sec` tăng > 30 % | nghẽn CPU hoặc graph bị ghi lại |
| `vram_train_gb` > 14 | sắp OOM |
| kernel `ERROR` | |

**KHÔNG dừng vì:** `f1_macro` giảm · `global_*` tệ ở round 1 · `kd` cao ở round 1 · `|S|` dao
động. Cả bốn là **hình dạng đã dự báo** — [`rebuild.md` §3](rebuild.md).

✔ `scripts/report_data.py` chạy thật lần đầu 23-09 trên 20c/50c/100c(38): đã sửa tiêu đề
`skd_loss.png` (sót từ mutual learning), trỏ caveat về `rebuild.md` §2/§8, legend/tiêu đề hình, và
thêm §5 "Chọn thiết bị và truyền tin" + `selection.png`. `--report docs/report.md` sinh phụ lục.

## 7. Bẫy vận hành đã biết (kế thừa từ các dự án anh em)

| bẫy | biểu hiện | cách tránh |
|---|---|---|
| `kernel_sources` qua tài khoản khác | push **báo thành công**, `/kaggle/input` rỗng, không exception | dataset checkpoint (§5) |
| `kaggle datasets create` mặc định | bỏ thư mục, ghi lại CSV | luôn `-r zip -t`, đối chiếu kích thước **mọi trang** của `datasets files` |
| Kaggle bỏ một cấp thư mục khi giải nén zip | `run_name` mất khỏi đường mount ⇒ discovery không tìm thấy | stage `<out>/runs/<run_name>/…` |
| `kaggle kernels output` treo / chết giữa chừng | file 0 byte, tải dở | `scripts/pull_output.sh` (retry, kill khi treo, in `PULL_OK`) |
| Thiếu block `nb.metadata["kaggle"]` | kernel lên với **0 GPU** | `validate_notebooks.py` kiểm; cell 1 assert đúng 2 × T4 `(7,5)` |
| 0 GPU dù metadata đúng | có thể **theo tài khoản** | chạy lại **y nguyên** ở tài khoản khác trước khi sửa code; ❌ không bao giờ xoá cái assert |
| W&B mất heartbeat ở phiên resume | `resume="allow"` cùng id, hàng mới không lên | kernel vẫn chạy — theo dõi bằng trạng thái kernel; W&B là **màn hình theo dõi, không phải phụ thuộc** |
| Cổng resume chỉ kiểm "có tìm thấy gì không" | một round hỏng ở giữa ⇒ phiên GPU **train lại** các round thiếu dưới cùng run name | probe CPU **assert đúng số round** trước khi push kernel GPU |
| Watchdog "file 0 byte > 4 phút" của `pull_output.sh` (22-09) | CLI đệm **cả file trong RAM** rồi mới ghi ⇒ file đích đứng ở 0 byte suốt lúc tải; băng thông ở đây ~1–1,7 MB/s nên weights 81–160 MB và preds 215 MB–1,08 GB "đứng yên" > 4 phút **hợp lệ** → bị giết, lượt sau tải lại từ đầu, vòng lặp không bao giờ xong | đã đổi watchdog sang **"không nhận byte nào qua mạng > 4 phút"** (`/proc/net/dev`) và trần lượt 3 h; **kéo tuần tự** từng kernel — hai pull song song chia đôi băng thông và không nhanh hơn |
| Output phiên nối tiếp chứa lại **mọi** round đã import (23-09) | kéo đủ cây phiên 2 = kéo lại toàn bộ phiên 1 | phiên cùng tài khoản: chỉ kéo đủ **cây phiên cuối**; hoặc `PULL_PATTERN='^(?!.*(weights\|resume\|confusion\|preds)/(round\|global)_0(…)\.)'` bỏ trọng số round đã có (`--file-pattern` dùng `re.search` trên đường dẫn). Phiên nhập từ handoff bundle chỉ chứa round bundle + round mới |
| `pkill -f <chuỗi>` trong tool call của agent | chuỗi khớp cả dòng lệnh của chính shell đang chạy ⇒ shell tự chết (exit 144), lệnh sau `pkill` không chạy | kill theo PID, hoặc mẫu neo vào đường dẫn script |
| DNS WSL chập chờn (22-09 02:30Z) | `NameResolutionError api.kaggle.com`, `curl` resolve timeout, ping IP vẫn được | vòng retry phải **chờ `getent hosts api.kaggle.com`** trước mỗi lượt, không đốt số lượt vào lỗi DNS |
| `run_local_checked.py` từ chối verify (22-09 06:13Z) | `Refusing local test: only 3139 MiB available` — VS Code server trên WSL ăn ~2,2 GB | cần ≥ 3.584 MiB khả dụng; đóng bớt cửa sổ VS Code rồi chạy lại; ❌ không bỏ qua watchdog (verify 50c/20c chỉ tốn 0,7–0,9 GB RSS) |

## 8. Kernel đã push

| kernel | tài khoản | mục đích | trạng thái |
|---|---|---|---|
| `catbaochau/perfed-skd-veremi-20-clients-probe` | catbaochau | 2 round + micro-benchmark T4, batch 512 | **COMPLETE** (2 round, [`tests.md` §4.2](tests.md)) |
| `khanhmay0304/perfed-skd-veremi-100-clients-probe` | khanhmay0304 | 2 round + micro-benchmark T4, batch 256 | **COMPLETE** (2 round, 1,32 h quota, [`tests.md` §4.2](tests.md)) |
| `catbaochau/perfed-skd-veremi-20-clients` | catbaochau | **production 20c**, 50 round, 1 phiên | pushed 13:16Z 21-09; **COMPLETE 50/50** (8,42 h; 593 s/round); output kéo đủ 06:20Z 22-09 (`runs/pulls/20c_s1`), merge → `runs/merged/perfedskd_20c`, **`verify_run --require-rounds 50` pass** (1 file preds ở round 50) |
| `trietbackup/perfed-skd-veremi-50-clients` | trietbackup | **production 50c**, phiên 1/2, `--max-hours 11` | pushed 13:16Z 21-09; **COMPLETE 34/50** (10,46 h; 1.088 s/round); output kéo đủ 06:12Z 22-09 (`runs/pulls/50c_s1`, 269 file), merge → `runs/merged/perfedskd_50c`, **`verify_run --require-rounds 34` pass** |
| `trietbackup/perfed-skd-veremi-50-clients-s2` | trietbackup | **production 50c**, phiên 2/2, `--max-hours 6.5`, `--require-resume --kernel-source …-50-clients` | pushed 06:16Z 22-09; **COMPLETE round 35–50**; kéo 01:50Z 23-09 (`runs/pulls/50c_s2`, `PULL_PATTERN` bỏ trọng số r1–34), merge s1+s2 → `runs/merged/perfedskd_50c`, **`verify_run --require-rounds 50` pass** |
| `minhtriethihi/perfed-skd-veremi-100-clients` | minhtriethihi | **production 100c**, phiên 1/3, `--max-hours 11.75` | pushed **16:57Z 21-09**; **COMPLETE 18/50** (11,19 h; 2.198 s/round — 18 round, không phải 17 như dự báo); file nhỏ đã kéo về `runs/pulls/100c_s1_small`, cây đủ **chưa** kéo |
| `minhtriethihi/perfed-skd-veremi-100-clients-s2` | minhtriethihi | **production 100c**, phiên 2/3, `--max-hours 11.75`, `--kernel-source …-100-clients` | pushed 06:10Z 22-09; **COMPLETE round 19–38** (khởi động 709 s, 1.930 s/round, dừng cổng ngân sách sau r38, 11,09 h); kéo đủ 01:43Z 23-09 (`runs/pulls/100c_s2`, 6,0 GB, chứa cả r1–18 đã import), merge với `100c_s1_small`, **`verify_run --require-rounds 38` pass** |
| dataset `khanhmay0304/perfedskd-100c-ckpt-s3` (private) | khanhmay0304 | handoff bundle round 38 + `handoff.json` (chuỗi sha 1..38), 144 MB zip | created 01:55Z 23-09, `ready`, 9 file khớp kích thước local |
| `khanhmay0304/perfedskd-100c-ckpt-probe` | khanhmay0304 | probe CPU cổng resume | pushed 02:06Z 23-09, COMPLETE sau ~30 s, **`PROBE_OK perfedskd_100c 38`** |
| `khanhmay0304/perfed-skd-veremi-100-clients-s3` | khanhmay0304 | **production 100c**, phiên 3/3, `--max-hours 11.75`, `--require-resume --dataset-source khanhmay0304/perfedskd-100c-ckpt-s3` | pushed **02:07Z 23-09**; **COMPLETE round 39–50** (khởi động ~8 phút, 1.978 s/round, 6,75 h đến hết r50); kéo đủ 09:25Z 23-09 (`runs/pulls/100c_s3`, 3,1 GB, chỉ r38–50), merge `100c_s1_small` + `100c_s2` + `100c_s3`, **`verify_run --require-rounds 50` pass** |

Probe kernel và dataset checkpoint nên **xoá sau khi xong** — nhưng đó là thao tác xoá ngoài
repo, **hỏi chủ dự án trước**.
