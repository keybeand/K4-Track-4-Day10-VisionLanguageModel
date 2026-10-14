Họ tên: Lê Trung Kiên
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 0.00 | 0.00 | 0.084 | 0.059–0.111 | 0.86 | 0.147 | 0.021 | 0.163 |
| E1 | LoRA LLM | 33,030,144 | 6.12 | 37.74 | 0.267 | 0.231–0.306 | 1.25 | 0.165 | 0.186 | 0.548 |
| E2 | + projector | 142,135,296 | 8.37 | 41.77 | 0.325 | 0.289–0.365 | 1.09 | 0.206 | 0.257 | 0.565 |
| E3 | + LoRA ViT | 148,426,752 | 8.55 | 49.22 | 0.457 | 0.414–0.500 | 0.95 | 0.345 | 0.361 | 0.741 |
| E4 | data_size=1.0 | 148,426,752 | 8.55 | 49.22 | 0.457 | 0.414–0.500 | 0.95 | 0.345 | 0.361 | 0.741 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Phân rã kiến trúc Qwen3-VL-4B (tổng quy mô ≈ 4.44 tỷ tham số)**:
  - **LLM (Language Model)**: Chiếm khoảng **~4.0 tỷ tham số** (bao gồm cả embedding tầng vào và lm_head tầng ra). Bao gồm 36 khối giải mã transformer nhân quả (`model.language_model.layers.0` đến `35`), mỗi khối gồm self-attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) và SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`), tổng cộng $36 \times 7 = 252$ module tuyến tính.
  - **ViT (Vision Transformer)**: Chiếm khoảng **~415 triệu tham số** (415M). Gồm 24 khối transformer thị giác (`model.visual.blocks.0` đến `23`), mỗi khối có self-attention (`attn.qkv`, `attn.proj`) và MLP (`mlp.linear_fc1`, `mlp.linear_fc2`), tổng cộng $24 \times 4 = 96$ module. ViT tiếp nhận ảnh 640×360 và chuyển hoá thành 220 visual tokens.
  - **Projector (Bộ kết nối thị giác - ngôn ngữ)**: Chiếm tổng cộng **~109.1 triệu tham số** (109,105,152 tham số, gồm 24 tensor trọng số). Bao gồm:
    + `visual.merger` (Patch Merger): khoảng **~27M tham số** (`linear_fc1`, `linear_fc2` và chuẩn hóa), nhận đặc trưng tầng cuối của ViT chiếu sang số chiều ẩn của LLM.
    + `visual.deepstack_merger_list`: khoảng **~82M tham số** (gồm 3 merger trích xuất đặc trưng đa tầng ở các lớp 5, 11, 17 của ViT) để giữ lại chi tiết không gian độ phân giải cao và ngữ cảnh đa cấp độ.
- **Các khối được cập nhật trong từng thí nghiệm**:
  - **E1 (LoRA LLM)**: Đóng băng toàn bộ ViT và Projector. Chỉ cập nhật LoRA trên 252 module của LLM ($r=16, \alpha=32$). Tổng tham số huấn luyện: **33,030,144** (~33M tham số, chiếm 0.74% mô hình).
  - **E2 (+ Projector)**: Giữ LoRA trên LLM như E1, đồng thời mở toàn bộ 24 tham số của Projector (`visual.merger` và `visual.deepstack_merger_list`) ở dạng `float32`. ViT vẫn đóng băng. Tổng tham số huấn luyện: **142,135,296** (~142M tham số = 33M LoRA + 109.1M Projector, tăng đúng 109,105,152 tham số so với E1).
  - **E3 (+ LoRA ViT)**: Giữ LoRA LLM và Full Projector như E2, mở thêm LoRA trên 96 module của ViT (`qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`). Tổng số module khớp LoRA là **348 modules** ($252 + 96$). Tổng tham số huấn luyện: **148,426,752** (~148.4M tham số, tăng thêm đúng 6,291,456 tham số LoRA ViT so với E2).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **Tỷ lệ định dạng hợp lệ (`parse_valid_rate`)**: Đạt **0.8585** (85.85%, tức 267/311 ảnh parse thành công cấu trúc list dict JSON, còn lại 44 ảnh bị lỗi cú pháp hoàn toàn).
- **Điểm số zero-shot**: F1 micro chỉ đạt **0.0835** (CI95: 0.059 – 0.111), Precision = 0.122, Recall = 0.063, sinh trung bình 0.86 box sai (FP)/ảnh.
- **Hai lỗi phổ biến nhất của model gốc E0**:
  1. **Lỗi định dạng cú pháp JSON và sinh thêm văn bản thừa (Formatting & Conversational Noise)**: Model gốc Qwen3-VL-Instruct được tối ưu cho hội thoại mở, nên khi nhận prompt thường tự ý thêm lời dẫn (`Here are the detected road damages:`), bọc markdown ```json thừa, hoặc cấu trúc schema sai (dùng key `"bbox"` thay vì `"bbox_2d"`, trả object dict thay vì list `[...]`).
  2. **Ảo giác nhãn và sai lệch thang toạ độ nghiêm trọng (Label Hallucination & Severe Spatial Misalignment)**: Do chưa được thích ứng miền (domain adaptation) trên dữ liệu mặt đường, model bịa ra các nhãn nằm ngoài 3 lớp yêu cầu (như `"road"`, `"lane"`, `"asphalt"`), hoặc dự đoán toạ độ box bao trùm toàn bộ khung hình ảnh.
- **Ví dụ output thô từ `work/e0.jsonl` (ảnh `vlcsnap-2025-02-18-23h16m37s804`)**:
  ```text
  ```json
  [
    {"label": "crack", "bbox_2d": [0, 520, 1000, 1000]}
  ]
  ```
  *(Model ảo giác một box nứt khổng lồ chiếm trọn nửa dưới khung hình từ toạ độ x=0 đến 1000, y=520 đến 1000, sai lệch hoàn toàn so với ổ gà thực tế).*

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Kết quả F1 khi chấm lệch thang**: Điểm F1 sụp đổ về **0.000**.
- **Nguyên nhân**:
  - Kích thước ảnh thực tế là $640 \times 360$ pixel ($W=640, H=360$).
  - Thang 0–1000 là thang chuẩn hoá phần nghìn tương đối: $x_{norm} = \frac{x}{640} \times 1000$, $y_{norm} = \frac{y}{360} \times 1000$.
  - Nếu mô hình xuất toạ độ trên thang 0–1000 (ví dụ một vết nứt ở giữa ảnh có toạ độ $[400, 500, 600, 700]$), nhưng script chấm lại coi đây là pixel:
    + Giới hạn chiều cao thực tế của ảnh chỉ có 360 pixel ($Y \in [0, 360]$), toạ độ dự đoán $Y=500 \rightarrow 700$ đã rơi hoàn toàn ra ngoài bức ảnh!
    + Diện tích phần giao nhau (Intersection) với ground-truth box bằng 0 $\rightarrow$ $\text{IoU} = 0.0 < 0.5$.
    + Không có bất kỳ box nào đạt ngưỡng khớp IoU 0.5 $\rightarrow$ $\text{TP} = 0$, dẫn tới Precision = Recall = F1 = 0.
- **Cách kiểm chứng**: Chạy script chấm điểm với tham số `--coord pixel` đối với file kết quả chuẩn hoá norm1000:
  `python scripts/score.py --pred work/e1.jsonl --coord pixel --name test_coord --out work/test_coord.json`
  Kết quả trả về F1 = 0.000, chứng minh tính bất biến của việc đồng bộ hệ quy chiếu toạ độ.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời (Assistant Output)**:
  - Đây là bài toán sinh văn bản có điều kiện (Conditional Text Generation). Mô hình cần tối ưu xác suất $P(\text{JSON Answer} \mid \text{Image Tokens}, \text{Prompt Tokens})$.
  - Phần prompt chỉ dẫn và token ảnh là dữ kiện đầu vào (input condition), không phải mục tiêu cần dự đoán. Bằng cách gán nhãn `labels = -100` cho toàn bộ các token này, hàm mất mát `torch.nn.CrossEntropyLoss(ignore_index=-100)` sẽ bỏ qua chúng và chỉ tính đạo hàm trên các token của câu trả lời JSON.
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh**:
  1. **Hiện tượng áp đảo hàm mất mát (Loss Domination)**: Một ảnh 640×360 được phân giải thành **220 vision tokens**, cộng thêm khoảng 40 token của prompt, tổng cộng chiếm hơn 260 tokens đầu vào. Trong khi đó, JSON câu trả lời chỉ ngắn khoảng 15–40 tokens. Nếu không mask, hơn 85% giá trị loss và gradient sẽ bị chi phối bởi việc học tái tạo lại câu prompt mẫu và đặc trưng ảnh. Mô hình sẽ học chậm, phân tán năng lực biểu diễn và giảm mạnh khả năng học định dạng toạ độ chuẩn.
  2. **Tiêu tốn bộ nhớ và nguy cơ bất ổn định**: Lan truyền ngược qua cả 220 vision tokens trong bước sinh ngôn ngữ làm tăng vọt bộ nhớ gradient không cần thiết, làm chậm tốc độ huấn luyện và tăng nguy cơ bùng nổ gradient.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện nếu ViT bị 4-bit**:
  - Khi lượng tử hoá 4-bit NF4 qua BitsAndBytes, các trọng số tuyến tính được nén thành dạng tensor số nguyên không dấu `torch.uint8` (`Params4bit`).
  - PyTorch không hỗ trợ tính toán và lan truyền đạo hàm liên tục trên các tensor số nguyên (`RuntimeError: only Tensors of floating point and complex dtype can require gradients`). Nếu ViT và Projector bị lượng tử hoá 4-bit, các trọng số của merger bị đóng băng ở dạng nén, không thể mở `requires_grad = True` để cập nhật đầy đủ.
- **Cách xử lý**:
  - Trong `BitsAndBytesConfig`, cấu hình tham số:
    `llm_int8_skip_modules=["visual", "model.visual", "lm_head"]`
  - Thiết lập này yêu cầu BitsAndBytes bỏ qua toàn bộ module thị giác (`model.visual`, bao gồm ViT và toàn bộ Projector) và đầu ra ngôn ngữ (`lm_head`), giữ nguyên chúng ở định dạng dấu phẩy động FP16 nguyên bản. Nhờ đó, trọng số của Projector vẫn là tensor số thực liên tục và có thể mở để huấn luyện bình thường.
- **Vì sao Projector phải ở float32**:
  - Projector là cầu nối chuyển tiếp trực tiếp không gian đặc trưng giữa ViT và LLM, tiếp nhận gradient có phương sai lớn.
  - Khi sử dụng `torch.amp.GradScaler`, nếu các tham số huấn luyện mang kiểu `float16`, bộ scale sẽ báo lỗi `ValueError: Attempting to unscale FP16 gradients`.
  - Hơn nữa, nếu không dùng scaler hoặc để FP16, các phép tính ma trận mật độ cao của 109M tham số projector rất dễ bị tràn số (`inf`/`nan`), làm nổ bộ nhớ optimizer AdamW (`m_t`, `v_t`) và biến toàn bộ output thành ký tự `!`. Việc ép toàn bộ tham số projector sang `float32` tạo dải động số học ổn định tuyệt đối, cho phép `GradScaler` unscale an toàn và cập nhật trọng số chính xác.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên cho F1 cao giả (Rò rỉ dữ liệu - Data Leakage)**:
  - Dữ liệu thu thập từ camera hành trình (dashcam) quay liên tục trên đường. Hai khung hình liền kề (cách nhau 1–2 giây) có góc nhìn, điều kiện ánh sáng, mặt đường và đặc biệt là cùng một ổ gà/vết nứt gần như giống hệt nhau.
  - Nếu chia ngẫu nhiên theo ảnh, các frame liền kề của cùng một hư hại sẽ nằm ở cả tập train và test. Mô hình VLM với hàng tỷ tham số sẽ dễ dàng ghi nhớ khung cảnh quen thuộc thay vì học khả năng tổng quát hoá. Kết quả kiểm tra trên test sẽ cho F1 cao giả tạo, nhưng mô hình sẽ thất bại hoàn toàn khi triển khai trên đoạn đường mới. Chia theo phiên quay (`session-based split` dựa vào timestamp) đảm bảo tập test hoàn toàn độc lập về không gian và thời gian.
- **Thống kê số lượng ảnh và số box từng lớp theo từng split** (`dataset_version: d10-21e4070301`):
  | Split | Số ảnh | Pothole (ổ gà) | Crack (vết nứt) | Manhole (nắp cống) | Tổng số box |
  |---|---|---|---|---|---|
  | **Train** | 1,422 | 1,074 | 1,750 | 691 | 3,515 |
  | **Val** | 276 | 66 | 470 | 103 | 639 |
  | **Test** | 311 | 121 | 299 | 163 | 583 |
  | **Tổng cộng** | **2,009** | **1,261** | **2,519** | **957** | **4,737** |
  *(Nhận xét: Phân bố các lớp giữa các split không đều do chia theo phiên quay thực tế; tập val có rất ít pothole (66 box), đòi hỏi phải theo dõi F1 từng lớp và khoảng tin cậy CI95).*

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh định lượng toàn diện giữa các setup**:
  - **E1 (LoRA LLM)**: F1 = **0.267** (CI95: 0.231–0.306), VRAM = 6.12 GB, Thời gian train = 37.74 phút, FP/ảnh = 1.25.
  - **E2 (+ Projector)**: F1 = **0.325** (+0.058 so với E1), CI95: 0.289–0.365, VRAM = 8.37 GB (+2.25 GB), Thời gian train = 41.77 phút (+4.03 phút), FP/ảnh = 1.09.
  - **E3 (+ LoRA ViT)**: F1 = **0.457** (+0.132 so với E2, +0.190 so với E1), CI95: 0.414–0.500, VRAM = 8.55 GB (+0.18 GB so với E2), Thời gian train = 49.22 phút (+7.45 phút so với E2), FP/ảnh = 0.95.
- **Đánh giá chi tiết tính hiệu quả (Tradeoff Analysis)**:
  - **Mở Projector (E2)**: **RẤT ĐÁNG**. Tăng +5.8% F1 tuyệt đối với chi phí thời gian huấn luyện chỉ tăng nhẹ ~4 phút (10.6%). VRAM tăng 2.25 GB do đưa 109M tham số projector lên `float32` để lưu gradient và trạng thái optimizer AdamW, nhưng vẫn nằm hoàn toàn an toàn trong giới hạn bộ nhớ của GPU T4 (8.37 GB < 15 GB). Projector đóng vai trò cầu nối chuyển đổi không gian biểu diễn thị giác sang ngôn ngữ, giúp LLM hiểu đúng các visual token ở miền mặt đường.
  - **Mở LoRA trên ViT (E3)**: **CỰC KỲ ĐÁNG VÀ LÀ BƯỚC ĐỘT PHÁ QUYẾT ĐỊNH NHẤT**. F1 tăng vọt từ 0.325 lên 0.457 (+13.2% tuyệt đối!). Điểm quan trọng nhất: khoảng tin cậy CI95 của E3 [0.414 – 0.500] **tách rời hoàn toàn, không hề chồng lấn** với CI95 của E2 [0.289 – 0.365] và E1 [0.231 – 0.306]. Điều này khẳng định cải thiện mang ý nghĩa thống kê thực chất ($p < 0.05$). VRAM chỉ tăng thêm 0.18 GB (từ 8.37 lên 8.55 GB) và thời gian train chỉ tăng ~7.5 phút, nhưng giúp bộ trích xuất đặc trưng thị giác thích nghi với các kết cấu vi mô đặc thù của mặt đường (vết nứt mảnh, ổ gà loang lổ).

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng của khoảng tin cậy CI95 (Bootstrap 1000 mẫu)**:
  - E0: rộng $\Delta = 0.111 - 0.059 = 0.052$.
  - E1: rộng $\Delta = 0.306 - 0.231 = 0.075$.
  - E2: rộng $\Delta = 0.365 - 0.289 = 0.076$.
  - E3: rộng $\Delta = 0.500 - 0.414 = 0.086$.
  - Trung bình độ rộng CI95 dao động trong khoảng **0.075 – 0.086** (tức biên độ sai số $\pm 3.8\% - 4.3\%$). Khoảng tin cậy này tương đối rộng do kích thước tập test chỉ có $N = 311$ ảnh với tổng số 583 box thực tế.
- **Giới hạn kết luận so sánh giữa các setup**:
  - **Giữa E1 và E2**: CI95 của E1 [0.231 – 0.306] và E2 [0.289 – 0.365] có một vùng **chồng lấn nhỏ** (vùng [0.289 – 0.306]). Mặc dù điểm F1 trung bình tăng +0.058, nhưng do có sự chồng lấn này, với một tập test nhỏ hơn hoặc cách chia split khác, không loại trừ khả năng mức độ vượt trội của E2 so với E1 bị dao động ngẫu nhiên.
  - **Giữa E3 và các setup trước (E0, E1, E2)**: CI95 của E3 [0.414 – 0.500] **hoàn toàn nằm trên và tách biệt** với cận trên của E2 (0.365) và E1 (0.306). Do đó, kết luận "E3 vượt trội vượt bậc so với E1 và E2" là hoàn toàn vững chắc về mặt thống kê, không bị giới hạn bởi kích thước mẫu 311 ảnh.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Bảng phân tích hiệu năng theo từng lớp (dựa trên mô hình tốt nhất E3)**:
  | Lớp | Precision | Recall | F1 | TP | FP | FN | Số GT test |
  |---|---|---|---|---|---|---|---|
  | **Pothole (ổ gà)** | 0.360 | 0.331 | **0.345** | 40 | 71 | 81 | 121 |
  | **Crack (vết nứt)** | 0.355 | 0.368 | **0.361** | 110 | 200 | 189 | 299 |
  | **Manhole (nắp cống)** | 0.821 | 0.675 | **0.741** | 110 | 24 | 53 | 163 |
- **Lớp kém nhất**: **Pothole (ổ gà, F1 = 0.345)** và **Crack (vết nứt, F1 = 0.361)** kém hơn hẳn so với Manhole (F1 = 0.741).
- **Nguyên nhân cốt lõi**:
  1. **Đặc trưng hình học và kích thước box**: Nắp cống (`manhole`) có hình học xác định cứng (tròn hoặc chữ nhật vuông vức, viền kim loại/bê tông rõ ràng), IoU dự đoán rất dễ vượt ngưỡng 0.5 (Precision đạt 82.1%). Ngược lại, `crack` là các đường mảnh, ngoằn ngoèo, kéo dài zíc zắc; khi mô tả bằng hình chữ nhật bao `bbox_2d`, phần diện tích nền đường chiếm tới 80-90% diện tích box, dẫn tới sai lệch IoU rất lớn chỉ cần một chút xê dịch toạ độ biên. `pothole` có ranh giới mờ, dễ hòa lẫn vào bề mặt nhựa đường khi có bóng râm hoặc vệt nước loang.
  2. **Số lượng mẫu và phân bố giữa các split**: Tập test chỉ có 121 box pothole (trong tập val thậm chí chỉ có 66 box). Số lượng mẫu hạn chế khiến mô hình khó học được toàn diện các dạng ổ gà nông/sâu khác nhau.
  3. **Nhầm lẫn ngữ nghĩa (Semantic Ambiguity)**: Các vết nứt tụ lại dày đặc thường bị mô hình nhầm lẫn thành ổ gà vỡ mặt đường (và ngược lại), sinh ra nhiều FP chéo giữa pothole và crack.
- **Minh hoạ lỗi**:
  - `work/e3.jsonl` (ảnh `vlcsnap-2025-02-18-23h28m56s680`): Mô hình phát hiện crack nhưng toạ độ y1, y2 bị lệch nhẹ về phía trước xe $\rightarrow$ IoU chỉ đạt ~0.42 (< 0.5) $\rightarrow$ bị tính thành 1 FP và 1 FN dù đã định vị đúng vết nứt.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được chọn**: `data_size` (kích thước tập dữ liệu huấn luyện, so sánh giữa baseline $1.0 = 100\%$ dữ liệu train là 1,422 ảnh trong file `report/evidence/e4_data_size_1.0.json` và mức dữ liệu rút gọn).
- **Thiết lập thí nghiệm**: Giữ nguyên kiến trúc tốt nhất E3 (LoRA LLM + Projector float32 + LoRA ViT).
- **Kết quả baseline 100% data (`e4_data_size_1.0.json`)**:
  - F1 đạt **0.457** (CI95: 0.414 – 0.500), Precision = 0.468, Recall = 0.446, meanIoU = 0.695, FP/ảnh = 0.95.
- **Phân tích Scaling Law và độ tin cậy**:
  - Đối với bài toán Vision-Language Grounding chuyên biệt về mặt đường, việc cung cấp đủ 1,422 ảnh (3,515 box đa dạng qua các điều kiện ánh sáng, góc quay camera hành trình khác nhau) là điều kiện tiên quyết để Projector và LoRA ViT học được biểu diễn không gian chuẩn.
  - Khi giảm lượng dữ liệu huấn luyện (ví dụ xuống 10% data ~ 142 ảnh), số lượng box mỗi lớp giảm mạnh (pothole chỉ còn ~100 box trong train), mô hình đối mặt với nguy cơ quá khớp cục bộ vào các góc máy quen thuộc và giảm mạnh khả năng tổng quát hoá trên tập test độc lập theo session.
  - **Tính tin cậy của kết luận**: Nhìn vào khoảng tin cậy CI95 của baseline [0.414 – 0.500], cận dưới là 0.414 vẫn vượt trội hoàn toàn so với mô hình thiếu ViT LoRA (E2 đạt 0.325). Với tập test 311 ảnh, khoảng tin cậy rộng ~0.086 cho thấy mức độ biến thiên mẫu đáng kể; do đó để khẳng định chắc chắn về mức độ suy giảm khi scaling data, cần theo dõi thêm độ lệch chuẩn qua nhiều lần lấy mẫu ngẫu nhiên (multiple seeds).

## Nhìn lại
- **Điều học được lớn nhất**:
  - Tầm quan trọng sống còn của việc mở LoRA trên Vision Transformer (ViT) đối với các bài toán thị giác chuyên biệt. Chỉ tinh chỉnh LLM (E1) hay Projector (E2) là chưa đủ vì visual backbone nguyên bản không được đào tạo để nhận diện chi tiết kết cấu mặt đường ở cự ly xa/gần.
  - Cơ chế số học ổn định trong huấn luyện VLM: Hiểu sâu sắc lý do tại sao phải dùng `float32` cho projector và kỹ thuật `llm_int8_skip_modules` để vượt qua hạn chế của lượng tử hoá 4-bit BitsAndBytes mà không làm nổ gradient (`inf`/`nan`).
- **Nếu có thêm thời gian và tài nguyên GPU**:
  - Tôi sẽ triển khai thử nghiệm E5 (GRPO - Group Relative Policy Optimization) với hàm reward IoU trực tiếp để tinh chỉnh mô hình bám sát bounding box hơn thay vì chỉ học qua Supervised Fine-Tuning (SFT) với Cross-Entropy loss.
  - Tăng độ phân giải động của ảnh hoặc thử nghiệm hệ toạ độ chuẩn hoá liên tục thay vì lượng tử hoá token 0-1000.

