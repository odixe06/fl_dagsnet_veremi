# Lightweight-Fed-NIDS — trích xuất phương pháp và những chỗ bài báo để trống

Nguồn: [`../Lightweight_FL.md`](Lightweight_FL.md) — Bouayad, Alami, Janati Idrissi, Berrada,
*Lightweight Federated Learning for Efficient Network Intrusion Detection*, IEEE Access, 2024,
DOI 10.1109/ACCESS.2024.3494057. Chỉ trích phần **phương pháp** (§III) và cấu hình thực nghiệm
(§IV-B) ở mức bài báo công bố; mọi lựa chọn của bản dựng nằm ở [`rebuild.md`](rebuild.md).

---

## 1. Ký hiệu (Table 1)

| ký hiệu | nghĩa |
|---|---|
| θ₀ | tham số khởi tạo của mô hình |
| θ | tham số mô hình toàn cục |
| θⱼ | mô hình cục bộ của client j |
| C, N | tập client và số client tham gia **một** vòng |
| T | số vòng truyền tin |
| 𝓜 | **mask cắt tỉa**, tensor nhị phân cùng hình dạng θ |
| Dⱼ, Nⱼ | dữ liệu của client j và kích thước của nó |
| B, η, 𝓛 | batch size, learning rate, hàm loss |

## 2. Kiến trúc hệ thống (§III-A)

* **Client** (Fig. 1, §III-A-1): thu dữ liệu cục bộ → **áp mask 𝓜 lên θ nhận được** → tối ưu
  θⱼ trên Dⱼ → gửi lên server.
* **Server** (§III-A-2): (i) *khởi tạo hệ thống* — đặt θ₀ và **tính mask 𝓜 một lần**, phát cả
  hai xuống mọi client ("we dispatch the mask only once"); (ii) *tổng hợp* bằng FedAvg cho tới
  khi đủ T vòng.

## 3. Khởi tạo (§III-B-1)

### 3.a Định nghĩa mô hình
Kaiming He init; kiến trúc = **feature extractor** (ResNet-50 / ResNet-101 / VGG-19 trên ảnh
3 kênh dựng từ 40 gói × 300 byte của một flow) + **một lớp fully-connected + softmax**. Loss là
cross-entropy (Eq. 5–6, dạng nhị phân).

### 3.b Tính mask cắt tỉa — Eq. (7)–(8)
* **Zero-shot** (Cai et al. [36]): tính từ trọng số vừa khởi tạo, **không cần dữ liệu**, không lặp.
* **Có cấu trúc** theo DepGraph (Fang et al. [15]): gom các lớp phụ thuộc nhau (liên lớp + nội
  lớp) thành **nhóm** g; điểm quan trọng của chiều cắt được k trong nhóm:

  $$\hat I_{g,k} = N \cdot I_{g,k} \big/ \sum \mathrm{TopN}(I_g), \qquad
    I_{g,k} = \sum_{\theta\in g}\|\theta[k]\|_1^2, \qquad I(\theta)=\|\theta\|_1 \tag{7}$$

  tức xếp hạng theo **chuẩn L1 theo nhóm**; nhóm điểm thấp bị cắt theo **pruning rate** định trước.
* Mask 𝓜 ∈ {0,1} cùng hình dạng θ; mô hình cắt tỉa: $\theta' = \mathcal M \odot \theta_0$ (Eq. 8).
* Bài báo nhấn mạnh zero-hoá đơn thuần **không** giảm kích thước/tính toán, nên dùng
  DepGraph ("DeepGraph" trong bài) để **xoá vật lý** tham số bị cắt. Công cụ: **Torch-Pruning**.
* Cắt tỉa là tất định và phụ thuộc kiến trúc; server làm rồi phát mask **một lần**.

## 4. Tối ưu cục bộ (§III-B-2, Algorithm 5)

```
Require: θ (toàn cục), 𝓜
1  B ← các batch; E ← số epoch; η ← learning rate
4  nhận θ và 𝓜 từ server
5  θ ← θ ⊙ 𝓜                        # áp mask
6  for epoch = 1..E:
7    for b ∈ B:
8      θ ← θ − η ∇𝓛(θ, b)            # gradient descent thuần
```

Gửi θ đã cập nhật (và đã cắt) về server.

## 5. Tổng hợp (§III-B-3, Algorithm 6, Eq. 9)

$$\theta_{global}^{t+1} = \frac{1}{N}\sum_{i=1}^{N}\theta_i^{t} \tag{9}$$

Trung bình **đơn** (không trọng số theo Nⱼ). Server phát θ^{t+1} xuống **mọi** client; client
**thay** mô hình cục bộ bằng mô hình toàn cục rồi train tiếp. Lặp T vòng.

## 6. Cấu hình thực nghiệm bài báo công bố (§IV)

| hạng mục | giá trị |
|---|---|
| Dữ liệu | UNSW-NB15, USTC-TFC2016, CIC-IDS-2017; ảnh flow 40×300 byte ×3 kênh |
| Phân chia client | **IID**, chia đều normal/attack cho mọi client; 70 % train / 30 % test |
| Số client | 10 (Table 5–7), 50 (Table 8), 100 (Table 9–10) |
| Số vòng | **T = 5** (Fig. 11–21) |
| Sparsity | **0 / 50 / 70 / 90 %** |
| Backbone | ResNet-50, ResNet-101, VGG-19 |
| Tổng hợp | FedAvg (Eq. 9); FedProx thêm ở Table 11, không tốt hơn |
| Metric | Accuracy, F1-score, Training Time, Inference Time, TA = T₀/Tₛ, IA = I₀/Iₛ, ΔAcc, ΔF1 |
| Thư viện | Python 3.9.15, NFStream, NumPy, Pandas, PyTorch, Torch-Pruning |

Kết luận bài báo: sparsity 70 % với ResNet-101 cho F1 cao nhất (0,9993, 10 client) với TA ×2,03;
90 % nhanh nhất (TA ×3,62 với VGG-19) nhưng có thể sập (accuracy 0,5 ở UNSW-NB15 50 client,
USTC-TFC2016 100 client).

## 7. Chỗ bài báo để trống hoặc tự mâu thuẫn — G1..G8

| # | chỗ trống | hệ quả cho bản dựng |
|---|---|---|
| G1 | **Không có η, B, E, optimizer, scheduler** nào được nêu (§IV-B bị cắt sau "with the …") | phải chọn ([`rebuild.md`](rebuild.md) D3) |
| G2 | "Sparsity" không được định nghĩa: tỉ lệ **kênh** cắt mỗi lớp (tham số `ch_sparsity`/`pruning_ratio` của Torch-Pruning) hay tỉ lệ **tham số** bị xoá? Abstract nói "reduces model size by 90 %" | dùng nghĩa của công cụ bài báo dùng: **tỉ lệ kênh mỗi lớp** (D2) |
| G3 | Chuẩn hoá Eq. (7) (`N · I / Σ TopN`) chỉ ảnh hưởng khi so **giữa các nhóm** (global pruning); bài báo không nói cắt local hay global | cắt **local** (tỉ lệ đồng đều mỗi lớp) — với local, mọi chuẩn hoá đơn điệu đều cho cùng thứ tự trong nhóm |
| G4 | Lớp nào được miễn cắt (lớp phân loại cuối?) | miễn lớp Linear cuối để giữ đủ 16 logit |
| G5 | Eq. (9) chia đều 1/N; với IID chia đều thì bằng FedAvg có trọng số. Với phân hoạch lệch thì hai công thức khác nhau | giữ **1/N** đúng chữ bài báo (D4) |
| G6 | Algorithm 5 dòng 5 áp mask mỗi vòng, nhưng nếu tham số đã bị **xoá vật lý** thì bước này là đồng nhất; nếu chỉ zero-hoá thì gradient sẽ làm tham số bị cắt mọc lại **trong** vòng và mask chỉ được áp lại ở đầu vòng sau | xoá vật lý (đúng như bài báo làm với DepGraph) ⇒ dòng 5 là đồng nhất |
| G7 | Kaiming He init: bài báo không nói dạng (normal/uniform, fan_in/fan_out) | mặc định PyTorch = Kaiming uniform a=√5 (`knowledge/architecture.md` §3.2) |
| G8 | Eq. (5) là cross-entropy **nhị phân**; Eq. (5) và (6) trộn softmax đa lớp với công thức BCE | CE đa lớp 16 lớp (softmax); dạng nhị phân là trường hợp C = 2 |

## 8. Vì sao không so số của bản dựng này với số của bài báo

Khác dữ liệu (VeReMi NextGen 16 lớp, đặc trưng bảng, thay vì ảnh flow), khác backbone
(DAGSNet 395 k tham số thay vì ResNet-50 25 M), khác phân hoạch (Dirichlet α = 0,5 thay vì IID),
khác số vòng (50 thay vì 5), khác mức cắt (chỉ 70 %). Các con số Accuracy/F1/TA/IA của bài báo
chỉ dùng để **so hình dạng** kết luận (cắt 70 % giữ hiệu năng? có tăng tốc không?), không để so
giá trị.
