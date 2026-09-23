# PerFed-SKD — trích xuất phương pháp và những chỗ bài báo để trống

Nguồn: Neha Singh, Jatin Rupchandani, Mainak Adhikari, *Personalized Federated Learning for
Heterogeneous Edge Device: Self-Knowledge Distillation Approach*, IEEE Trans. Consumer
Electronics. Bản đầy đủ nằm ở
[`../Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.md`](../Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.md).

File này **chỉ** ghi lại những gì bài báo thật sự nói, cộng với danh sách chỗ nó để trống.
Mọi lựa chọn để lấp chỗ trống nằm ở [`rebuild.md`](rebuild.md), không nằm ở đây.

---

## 1. Ký hiệu

| ký hiệu | nghĩa trong bài báo |
|---|---|
| M | tập edge device; m ∈ [M] là một device |
| D_m | dữ liệu riêng của device m; D = ∪ D_m |
| (k, l) | một mẫu: đầu vào k, nhãn l ∈ [1, C] |
| ω | model toàn cục ở server |
| ω_m^t | model cục bộ của device m ở vòng t |
| V_m | **model cá nhân hoá lịch sử** của device m — teacher trong SKD |
| S | tập con device được chọn trong một vòng |
| A | "Global model accuracy" |
| a, a_m | "Local model accuracy" |
| τ | ngưỡng accuracy |
| T | tổng số communication round |
| E | số epoch cục bộ |
| η | learning rate |
| λ | siêu tham số điều khiển SKD |

## 2. Bài toán gốc (Eq. 1)

$$\min\ F(\omega) = \sum_{m=1}^{M} \frac{|D_m|}{|D|} F_m(\omega)$$

Đây là mục tiêu FedAvg chuẩn, nêu ra để rồi **bác bỏ**: §III-B nói rằng dưới non-IID, model
toàn cục tốt hơn model cục bộ ở chỉ số trung bình nhưng tệ hơn ở một số device, nên mục tiêu
thật của PerFed-SKD là **M model cá nhân hoá**, không phải một ω tốt nhất. Output của
Algorithm 1 ghi rõ: "Local personalized models P_m, m ∈ [M]".

## 3. Hàm mục tiêu cục bộ (Eq. 2) — trái tim của phương pháp

$$\phi_m(\omega_m^t) = f_m(\omega_m^t) + \lambda\, L\big(x(V_m)\,\|\,x(\omega_m^t)\big)$$

$$\omega_m^t \leftarrow \omega_m^t - \eta\,\nabla \phi_m(\omega_m^t, V_m) \qquad (3)$$

Bài báo nói về từng thành phần:

* `f_m(ω_m^t)` — "cross entropy loss of m-th edge device". **Rõ ràng.**
* `L` — "the divergence function between current local prediction x(ω_m^t) and past
  personalized prediction x(V_m)". **Chỉ nói là "divergence function".**
* `λ` — "hyperparameter for controlling the process of SKD". **Không cho giá trị.**
* `V_m` — model cá nhân hoá của chính device đó, lưu ở vòng trước (Algorithm 2 dòng 12).
* Eq. (3) nói "Stochastic Gradient Descent". **Không nêu optimizer thật, không nêu η, không
  nêu lịch LR.**

Điểm quan trọng: `L` **một chiều** — teacher V_m truyền kiến thức cho student ω_m^t, không có
chiều ngược lại. §I: "the prior personalized model transmits historical personalized knowledge
to the current local model". Đây là khác biệt so với deep mutual learning.

## 4. Vòng lặp — bảy pha của §IV

a) server broadcast ω^t cho các device thuộc S;
b) các device train bằng SKD, khởi tạo ω_m^t từ tham số nhận được;
c) sau khi train, lưu ω_m^{t+1} thành teacher V_m cho vòng sau;
d) gửi ω_m^{t+1} về server;
e) server tổng hợp thành global model mới;
f) device selector chọn các device có accuracy dưới ngưỡng;
g) gửi model mới tổng hợp cho các device được chọn.

## 5. Algorithm 1 (server) và Algorithm 2 (device) — nguyên văn

```
Algorithm 1                                    Algorithm 2
1  Cloud Server Side                           1  Edge Device Side
2  Initialize model ω^0                        2  for each edge device m ∈ M do in parallel
3  for t = 1 to T do                           3    Receive ω^t from the cloud server
4    τ = A                                     4    if m ∈ S then
5    if a < τ then                             5      ω_m^t ← ω^t
6      Add device m to subset S                6    else
7      Send local model ω^t to all selected S  7      ω_m^t ← ω_m^t
8      for m = 1 to S do                       8    end if
9        ω_m^{t+1}, a_m ← Local update(m, ω^t) 9    for e = 1 to E do
10   ω^{t+1} ← Σ_{m∈M} ω^{t+1} / |S|          10     Compute local update using equation (3)
11   A ← Σ_{m∈M} a_m / |M|                    11     ω_m^t ← ω_m^t − η∇φ_m(ω_m^t)
                                              12     Save personalized model V_m ← ω_m^t
                                              13 Return ω_m^t, a
```

## 6. Cấu hình thực nghiệm bài báo công bố

| | |
|---|---|
| Phần cứng | Dell Vostro, Intel Core i7-12700 (CPU), Spyder + PyTorch |
| Quy mô | 20 → 80 edge device |
| Vòng toàn cục | **200** |
| Epoch cục bộ | **20** |
| Learning rate η | **0,01** |
| Lặp lại | 10 lần |
| Dữ liệu | MNIST (70.000 ảnh, 10 lớp), EMNIST (bài báo ghi 60.000 / 26 lớp ở Table II nhưng 814.255 / 47 lớp ở phần mô tả — **mâu thuẫn nội tại**) |
| Phân hoạch | Dirichlet α ∈ {0,001, 0,01, 0,1} |
| Metric | Accuracy, global loss, personalized local loss, user training time, server aggregation time |
| Baseline | FedAvg, FedProx, Fed-ensemble, PerFedAvg |

Kết quả headline: MNIST 20 device **95,69 ± 0,03**; MNIST 80 device 92,47 ± 0,06; EMNIST 20
device 78,63 ± 0,46; EMNIST 80 device 73,57 ± 0,24 (Dirichlet α = 0,001).

---

## 7. Mười chỗ bài báo để trống hoặc tự mâu thuẫn

Đánh số để `rebuild.md` §2 tham chiếu. Không mục nào ở đây được "suy ra" — chúng là chỗ trống
thật, và mọi cách lấp đều là lựa chọn của bản dựng.

| # | chỗ trống | bài báo nói gì | vì sao phải quyết |
|---|---|---|---|
| **G1** | **Định nghĩa τ** | Alg.1 dòng 4 `τ = A`, dòng 11 `A ← Σ_{m∈M} a_m / \|M\|` (trung bình accuracy các device). Nhưng §III-A mô tả Device Selector: "all local models' accuracy is compared with **global aggregated model accuracy**". | Hai định nghĩa cho hai thuật toán khác hẳn nhau. Chỉ định nghĩa thứ nhất được viết thành công thức. |
| **G2** | **Ai train, ai upload** | Alg.1 dòng 8–9 lặp `for m = 1 to S` (chỉ S train). §IV: "The selected edge devices perform the local training … whereas **the remaining edge devices train the model again with the existing model parameters**". Alg.2 dòng 2 lặp trên **mọi** m. Dòng 10 viết `Σ_{m∈M}` nhưng chia `\|S\|`. | Ba phát biểu, ba tập khác nhau. Số chia `\|S\|` chỉ nhất quán nếu tổng chạy trên S. |
| **G3** | **Dạng của L** | "divergence function". Không nói KL, không nói L2, không nói nhiệt độ. Có trích [16] (Hinton KD) và [18] (progressive self-KD) ở Related Work nhưng không nói dùng công thức nào. | Quyết định hàm loss. |
| **G4** | **λ** | "hyperparameter for controlling the process of SKD". Không có giá trị, không có bảng quét. | Quyết định trọng số giữa CE và KD. |
| **G5** | **Optimizer và LR** | Eq.(3) "Stochastic Gradient Descent"; §V-A "local learning rate η = 0,01" cho MNIST/EMNIST với 20 epoch cục bộ. Không có weight decay, momentum, clip, lịch LR. | 0,01 là của MNIST-MLP 20 epoch, không chuyển sang bài toán khác được. |
| **G6** | **V_m ở vòng 1** | Alg.2 dòng 12 lưu V_m sau khi train. Ở t = 1 chưa có V_m. | Cần định nghĩa teacher của vòng đầu. |
| **G7** | **S ở vòng 1** | Alg.1 dòng 4 cần A, mà A chỉ có sau dòng 11 của vòng trước. | Cần định nghĩa tập chọn của vòng đầu. |
| **G8** | **a_m đo trên tập nào** | "Local model accuracy". Không nói test cục bộ hay test chung. Table III gọi kết quả là "personalized average test accuracy". | Trên VeReMi không có test chia theo client. |
| **G9** | **Teacher chạy ở chế độ nào** | Không nhắc. | BatchNorm/Dropout khiến train() và eval() cho teacher khác nhau hoàn toàn. |
| **G10** | **Teacher model phía server** | §III-A: "The Teacher Model is responsible for training the dense network and transferring their knowledge to the student model through SKD. Initially, the global model is trained by the central server using the **predefined dataset**." Không có ở Algorithm 1 lẫn Algorithm 2, không có trong Eq. (2), và "predefined dataset" không được định nghĩa ở đâu. | Thành phần này xuất hiện trong mô tả kiến trúc nhưng không xuất hiện trong bất kỳ công thức hay dòng giả mã nào. |

## 8. Ba điều bài báo **không** yêu cầu, hay bị gán nhầm

1. **Không có kiến trúc cụ thể.** Bài báo không nêu mạng nào cho MNIST/EMNIST. Không có ràng
   buộc "phải là CNN 2 lớp" hay bất cứ gì. DAGSNet vì thế không mâu thuẫn với bài báo.
2. **Không có backbone trích xuất đặc trưng.** §III-A chỉ có Student Model Manager và Teacher
   Model; không có mô-đun đặc trưng tách rời. Không có gì phải "bỏ đi" — nó chưa từng có.
3. **Không có proximal term thật sự.** §III-A ghi Training Optimizer "A proximal term is
   incorporated as an objective function for local training" — nhưng Eq. (2) **không có** số
   hạng proximal, chỉ có CE + λ·L. Câu đó là mô tả FedProx bị sót lại, không phải phương pháp
   này. Không dựng proximal term.

## 9. Vì sao không so số của bản dựng này với số của bài báo

Bài báo đo trên MNIST/EMNIST (ảnh xám 28×28, 10 hoặc 26 lớp, 60–70 nghìn mẫu, Dirichlet
α = 0,001), 200 vòng × 20 epoch cục bộ, trên CPU. Bản dựng này đo trên VeReMi NextGen
(66 đặc trưng dạng bảng, 16 lớp, 43 triệu dòng train, Dirichlet α = 0,5), 50 vòng × 1 epoch,
trên 2×T4. Cái được kế thừa là **phương pháp** — Eq. (2), quy tắc chọn device, quy tắc tổng
hợp — không phải con số. Xem thêm `knowledge/architecture.md` §8 điểm 10.
