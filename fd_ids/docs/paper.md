# FD-IDS — trích xuất phương pháp và những chỗ bài báo để trống

Nguồn: Huaiyuan Peng, Yanfeng Xiao, Chunming Wu, *FD-IDS: A Federated Learning and Knowledge
Distillation-Based Intrusion Detection System for Non-IID IoT Environments*, **Sensors 2025, 25,
4309**. Bản đầy đủ nằm ở [`../sensors-25-04309.md`](../sensors-25-04309.md).

File này **chỉ** ghi lại những gì bài báo thật sự nói, cộng với danh sách chỗ nó để trống. Mọi
lựa chọn để lấp chỗ trống nằm ở [`rebuild.md`](rebuild.md), không nằm ở đây.

---

## 1. Ký hiệu

| ký hiệu | nghĩa trong bài báo |
|---|---|
| K | số client (thí nghiệm: 9) |
| D_k, n_k | dữ liệu và số mẫu của client k; n = Σ n_k |
| w_G^t | tham số global model ở round t |
| w_k | tham số model của client k trong lúc train cục bộ |
| Z_t, Z_s | logit của teacher (global model) và student (client model) |
| T | nhiệt độ của KD |
| λ | trọng số giữa loss nhãn cứng và loss mềm |
| μ | hệ số điều chuẩn trong proximal term của FedProx |
| β | trọng số của proximal term trong loss tổng |
| R, E, η | số round, số epoch cục bộ, learning rate |
| θ | tham số Dirichlet của phân hoạch non-IID |

## 2. Ba thành phần của phương pháp

FD-IDS = **FedAvg có trọng số theo số mẫu + proximal term của FedProx + knowledge distillation
từ global model xuống client**, chạy với **toàn bộ client mỗi round**.

**Tổng hợp — Eq. (2):**

$$w_G^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n}\, w_k^{t+1}$$

**Proximal term — Eq. (3):**

$$L_{\text{proximal}} = \frac{\mu}{2}\,\| w_k - w_G^t \|^2, \qquad \mu = 0{,}01$$

**Loss KD — Eq. (4):** global model là teacher, model của client là student.

$$L_{\text{soft}} = T^2 \cdot KL\big(\text{softmax}(Z_t/T)\ \|\ \text{softmax}(Z_s/T)\big), \qquad T = 3$$

**Loss của client — Eq. (5) (biến thể FedAvg) và Eq. (6) (biến thể FedProx, là FD-IDS):**

$$L_{\text{distill}} = \lambda\, L_{\text{hard}} + (1-\lambda)\, L_{\text{soft}} \quad (5)$$

$$L_{\text{distill}} = \lambda\, L_{\text{hard}} + (1-\lambda)\, L_{\text{soft}} + \beta\, L_{\text{proximal}} \quad (6)$$

với L_hard là cross-entropy trên nhãn thật, **λ = 0,5**, **β = 0,1**.

Bài báo nói về từng thành phần:

* Teacher là **global model của round** — "the global model functions as the teacher model, while
  the local models on the clients serve as the student models" (§3.5.2).
* KD là **một chiều**, từ teacher xuống student; chiều KL được viết rõ trong Eq. (4).
* Proximal term lấy nguyên từ FedProx; §3.5.1 nêu μ = 0,01, Table 3 nêu thêm β = 0,1.

## 3. Algorithm 1 — nguyên văn (rút gọn ký hiệu)

```
Require: R, K, D_k, E, η, λ, β          Ensure: w_G^R
Server:
 2  Initialize w_G^0
 3  for t = 1..R:
 4    m = K                                   (mọi client tham gia)
 5    for each client k (song song):
 6      send w_G^t to client k
 7      receive w_G^{t+1} from client k       (sic — xem G7)
 9    w_G^{t+1} = Σ_k (n_k/n) w_k^{t+1}
11  return w_G^R
Client — ClientUpdate(w_G^t, D_k):
14  w_k = w_G^t
15  for e = 1..E:
16    for batch b ⊆ D_k:
17      L = λ·L_hard + (1−λ)·L_soft + β·L_proximal
18      w_k ← w_k − η∇L
21  return w_k
```

## 4. Tiền xử lý và mô hình của bài báo

* Bỏ các cột dễ bị giả mạo (`udp.port`, `ip.src_host`, `ip.dst_host`), bỏ dòng thiếu/vô hạn,
  one-hot cột định danh, **z-score** mọi đặc trưng (§3.3).
* Chọn đặc trưng bằng **mutual information** (Eq. 1): giữ top **25/95** (Edge-IIoT) và **71/115**
  (N-BaIoT).
* Bộ phân loại: **DNN 5 lớp ẩn 32-64-128-64-32**, ReLU, softmax ở đầu ra (§3.4). Tự kiểm: 25 đầu
  vào → 32-64-128-64-32 → 15 lớp cho đúng **22.095** tham số mà Table 14 công bố (Edge-IIoT có 14
  loại tấn công + benign).

## 5. Cấu hình thực nghiệm bài báo công bố

| | |
|---|---|
| Dữ liệu | Edge-IIoT, N-BaIoT; chia train/test **80/20** |
| Số client | **9**, đồng bộ, mọi client mỗi round |
| Non-IID | Dirichlet θ = **1** (low) và θ = **0,1** (high), chỉ áp lên tập train (Eq. 7) |
| Round × epoch | **40 × 2** |
| Batch / optimizer / lr | **128** / **Adam** / **0,001** (Table 3, chọn bằng grid search ở Table 2) |
| μ / λ / β / T | 0,01 / 0,5 / 0,1 / 3 |
| Metric | Accuracy, Precision, Recall, F1 (§4.2) + FPR, FNR; thời gian chạy |
| Baseline | FedAvg-only, FedProx-only, SIM-FED; CL cùng mô hình (§4.3.7) |

Kết quả headline ở round 40 (Table 5, accuracy của global model): Edge-IIoT **94,82** (θ = 1) và
**93,86** (θ = 0,1); N-BaIoT 87,70 và 83,81.

Bài báo cũng so **ba khoảng KD** (§4.3.2): *round-wise* (sau mỗi round), *periodic* (mỗi 8 round)
và *end-of-training* (round 31–40). Round-wise tốt nhất ở cả hai dữ liệu (Table 6–7). Ablation KD
(Table 8–9) cho mức tăng lớn nhất ở θ = 0,1.

---

## 6. Những chỗ bài báo để trống hoặc tự mâu thuẫn

Đánh số để `rebuild.md` §2 tham chiếu. Không mục nào ở đây được "suy ra" — chúng là chỗ trống
thật, và mọi cách lấp đều là lựa chọn của bản dựng.

| # | chỗ trống | bài báo nói gì | vì sao phải quyết |
|---|---|---|---|
| **G1** | **Thời điểm KD** | Algorithm 1 dòng 17 tính L_soft **trong từng batch** của local training. §4.3.2 lại mô tả round-wise KD là "performed immediately **after** each communication round". | Hai cách đọc cho hai thuật toán khác nhau: KD trong loss cục bộ, hay một pha KD riêng sau khi tổng hợp. |
| **G2** | **Teacher ở chế độ nào, cập nhật khi nào** | Chỉ nói "global model". Không nói teacher có đóng băng suốt round không, chạy `train()` hay `eval()`. | BatchNorm/Dropout cho output khác nhau ở hai chế độ. |
| **G3** | **Hệ số proximal chồng hai lần** | Eq. (3) đã có μ/2; Eq. (6) nhân thêm β. Không nói β·μ/2 là cố ý hay β thay cho μ. | Hệ số hiệu dụng là 0,1 × 0,01 / 2 = **5e-4** nếu đọc theo chữ. |
| **G4** | **Optimizer so với giả mã** | Table 3 và §3.4: **Adam**, lr 0,001. Algorithm 1 dòng 18: `w_k ← w_k − η∇L` (bước SGD). Không nói state của Adam giữ hay reset giữa các round. | Quyết định optimizer và vòng đời moment. |
| **G5** | **Softmax ở đầu ra + cross-entropy** | §3.4 đặt softmax ở output layer; Table 3 ghi "Cross-Entropy Loss". | Trong PyTorch, `CrossEntropyLoss` trên output đã softmax là softmax hai lần. Cần quyết L_hard và L_soft tính trên logit hay xác suất. |
| **G6** | **"λ = distillation weight" nhưng λ nhân loss cứng** | Table 3 gọi λ là "Distillation Weight"; Eq. (5)–(6) nhân λ với **L_hard**, (1−λ) với L_soft. | Với λ = 0,5 hai cách đọc trùng nhau; chỉ quan trọng khi quét λ. |
| **G7** | **Lỗi ký hiệu ở Algorithm 1 dòng 7** | "Receive w_G^{t+1} from client k". | Phải là w_k^{t+1}; dòng 9 mới tạo w_G^{t+1}. |
| **G8** | **Đánh giá "client" ở Table 5** | Cột B/W là accuracy của client tốt nhất/tệ nhất, nhưng không nói đo model nào (trước hay sau local update), trên tập test nào (test chung hay test cục bộ). | FD-IDS không có bước cá thể hoá: client bị ghi đè bằng w_G mỗi round. |
| **G9** | **Metric đa lớp và FPR/FNR** | §4.2 định nghĩa từ ma trận nhầm lẫn **nhị phân** (Table 4). Thí nghiệm là đa lớp; kiểu trung bình (macro/weighted) không nêu. Trong Table 8 recall luôn bằng accuracy. | Recall = accuracy chỉ xảy ra với trung bình **weighted**; FPR/FNR đa lớp cần một quy ước nhị phân hoá. |
| **G10** | **BatchNorm trong phép tổng hợp** | Không nhắc: DNN của bài báo không có BN. | Chỉ thành chỗ trống khi đổi sang bộ phân loại có BN. |

## 7. Ba điều bài báo **không** yêu cầu, hay bị gán nhầm

1. **Không có bước cá thể hoá.** Algorithm 1 dòng 14 `w_k = w_G^t`: không trọng số client nào tồn
   tại qua ranh giới round. Model duy nhất tồn tại liên tục là **global model**.
2. **Không có chọn client.** Dòng 4 `m = K`: mọi client tham gia mọi round.
3. **Không có teacher riêng ở server hay dữ liệu công khai.** Teacher chính là w_G^t mà client vừa
   nhận; KD không cần thêm dữ liệu.

## 8. Vì sao không so số của bản dựng này với số của bài báo

Bài báo đo trên Edge-IIoT/N-BaIoT (15 lớp, 25/71 đặc trưng sau MI), 9 client, Dirichlet θ ∈
{1; 0,1}, 40 round × 2 epoch, DNN 22.095 tham số, bộ metric có FPR/FNR nhị phân. Bản dựng này đo
trên VeReMi NextGen (66 đặc trưng dạng bảng, 16 lớp, 43 triệu dòng train, Dirichlet α = 0,5),
20/50/100 client, 50 round × 1 epoch, DAGSNet 395.024 tham số, trên 2×T4. Cái được kế thừa là
**phương pháp** — Eq. (2)–(6) và Algorithm 1 — không phải con số. Xem thêm
[`../knowledge/architecture.md`](../knowledge/architecture.md) §8.
