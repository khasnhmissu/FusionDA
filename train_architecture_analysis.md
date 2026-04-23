# Phân tích Module Kiến trúc `train.py` (FusionDA)

File `train.py` đóng vai trò là entry point chính của toàn bộ hệ thống huấn luyện **Semi-Supervised Domain Adaptive Training (FDA)** cho mô hình nhận diện vật thể (YOLO26/YOLOv8). Cấu trúc của file được thiết kế phức tạp nhưng chặt chẽ, kết hợp giữa Object Detection cơ bản với các kỹ thuật Teacher-Student (Knowledge Distillation) và Adversarial Domain Adaptation (GRL).

Dưới đây là bản phân tích chi tiết từng khối cấu trúc, luồng dữ liệu, cập nhật thuật toán cũng như giải thích tường tận các tham số ở phần main.

---

## 1. Cấu trúc Các Component Cốt lõi (Module Architecture)

Hệ thống được thiết kế xoay quanh 4 thành phần chính:
1. **Student Model (`model_student`)**: Mô hình chính đang được cập nhật tham số thông qua gradient descent. Đóng vai trò là mô hình nhận diện cuối cùng sau khi train xong.
2. **Teacher Model (`model_teacher`, quản lý bởi `WeightEMA`)**: Phiên bản EMA (Exponential Moving Average) của Student model. Teacher không cập nhật bằng backpropagation mà update tịnh tiến theo Student. Nhiệm vụ của nó là sinh ra Pseudo-labels (nhãn giả) ổn định và trích xuất Features chuẩn trên ảnh **Target Fake** để hướng dẫn Student.
   - **`update_after_step=2000`**: EMA delay — Teacher không cập nhật trong 2000 step đầu để Student ổn định trên GT trước.
   - **`alpha_rampup_steps=5000`**: Cosine ramp-up — `α_eff` bắt đầu từ `1.0` (không cập nhật) rồi giảm dần về `α` target, tránh noise từ Student giai đoạn đầu "nhiễm độc" Teacher.
   - **`pause_updates(steps)`**: Cơ chế tạm dừng EMA nếu phát hiện Teacher thoái hóa (quality gating).
3. **Domain Discriminator & GRL (Gradient Reversal Layer)**: 
   - Có thể là dạng **Single-scale** (`DomainDiscriminator`): GAP + MLP (LayerNorm → FC → LeakyReLU → Dropout → FC → LeakyReLU → Dropout → FC → logit [B,1]). Hook ở layer backbone cuối cùng (SPPF/C2PSA).
   - Hoặc dạng **Multi-scale** (`MultiScaleFusedDiscriminator`): Trích xuất feature ở 3 level backbone (P3, P4, P5), mỗi scale qua GRL riêng (với alpha multiplier khác nhau: P3×1.5, P4×1.0, P5×0.5), rồi GAP → Bottleneck (asymmetric: 256/128/64), concat → ONE shared MLP (Spectral Normalization) → [B,1]. P3 chiếm 57% input của MLP, bias mạnh vào small objects. Dùng để đảo ngược gradient về backbone, ép mô hình học được đặc trưng không phân biệt Domain (Source vs Target).
4. **Feature Hooks**:
   - **`YOLOv8FeatureHook`** (single-scale): Cắm vào 1 layer chỉ định, trích xuất feature tensor khi forward pass chạy qua. Dùng cho GRL single-scale và Feature Distillation.
   - **`MultiScaleFeatureHook`** (multi-scale): Bọc nhiều `YOLOv8FeatureHook`, mỗi hook cắm vào 1 scale. Dùng cho Multi-scale GRL.
   - **Hooks cho Feature Distillation (`student_distill_hook`, `teacher_feature_hook`) tách biệt hoàn toàn với hooks cho GRL**: Student distill hook cắm trên Student ở backbone end, Teacher feature hook cắm trên Teacher ở backbone end. Hai hệ thống hook không can thiệp lẫn nhau.

---

## 2. Luồng Dữ liệu (Data Flow) và Các Bước Hoạt động

Thuật toán sử dụng DataLoader đặc biệt tên là `PairedMultiDomainDataset`, mỗi batch sinh ra 4 loại ảnh đồng thời: `Source Real`, `Source Fake`, `Target Real`, `Target Fake`. Dataset match ảnh theo **filename** (stem) để đảm bảo:
- Source Real ↔ Source Fake (cùng scene) → Consistency Loss.
- Target Real ↔ Target Fake (cùng scene) → Distillation.

### Trong mỗi Iteration (Vòng lặp bên trong Epoch):

**Bước 1: Forward Pass cho Dữ liệu Source (Học từ nhãn thật)**
- Đưa `imgs_source` qua Student → `pred_source`.
- Tính loss có giám sát `loss_source` (dựa trên GT thật) qua `FDALoss` (wrapper của `E2ELoss` cho YOLO26 hoặc `v8DetectionLoss` cho YOLOv8/11).
- Lấy `source_features` ra từ các Hooks (multi-scale hoặc single-scale tùy config) để dành cho GRL.
- Tương tự với `imgs_source_fake` → sinh ra `pred_source_fake` và tính `loss_source_fake` (dùng cùng batch label của Source Real — vì scene giống nhau). Thu thập `source_fake_features`.

**Bước 2: Forward Pass cho Dữ liệu Target (Học từ nhãn giả & GRL)**
- Đưa `imgs_target` (Target Real) qua Student → `pred_target`.
- Lấy `student_feat_for_mse` từ `student_distill_hook` (hook riêng, KHÔNG liên quan đến GRL hook). Đây là feature để so sánh với Teacher trong Feature KD.
- Lấy `target_features` từ GRL hooks (để dùng cho Domain Loss).

**Bước 3: Tính Domain Adversarial Loss (GRL)**
- *(Chỉ kích hoạt nếu `epoch >= grl_warmup`)*.
- **Adaptive GRL Weight Boost**: Dựa trên `prev_domain_acc` (accuracy từ iteration trước):
  - Nếu `prev_domain_acc > 0.75`: Discriminator đang thắng, boost `effective_grl_weight` lên tối đa `3x` theo công thức tuyến tính `boost = 1.0 + 2.0 * (acc - 0.75) / 0.25`.
- **Discriminator Throttling**: Nếu `domain_acc > 0.85`, DỪNG cập nhật optimizer cho Discriminator (chỉ backbone nhận GRL gradient). Discriminator chỉ được update lại khi acc giảm xuống.
- Multi-scale path: `ms_discriminator(ms_src_feats)` và `ms_discriminator(ms_tgt_feats)` → `compute_domain_loss()` × `effective_grl_weight`.
- Single-scale path: `domain_discriminator(source_features)` và `domain_discriminator(target_features)` → tương tự.

**Bước 4: Teacher sinh Pseudo-Labels và Feature trên Target Fake**
- *(Chỉ thực hiện nếu đã qua giai đoạn `burn_in_epochs`)*.
- Đưa `imgs_target_fake` qua **Teacher Model** (không dùng gradient — `torch.no_grad()`).
- Lấy đầu ra `pred_teacher`, xử lý:
  - Ưu tiên lấy `one2many` head (nếu model trả dict) vì `one2one` (end-to-end Hungarian) thường sparse và confidence thấp.
  - Kiểm tra/xử lý shape: permute nếu cần, apply sigmoid nếu score chưa normalize.
  - Kiểm tra NaN/Inf → fallback.
- **Adaptive Confidence Thresholding** (Curriculum Learning): Dùng `get_adaptive_conf_thres()` — hàm cosine schedule tăng dần từ `conf_thres` (min) lên `conf_thres_max` (max) sau burn-in. *Lưu ý: trong burn-in trả về `max_conf` trực tiếp*.
- **NMS**: `non_max_suppression()` với `max_det=20` (giới hạn) để ngăn box explosion.
- **Post-filter**:
  - Clamp coordinates về trong ảnh.
  - Loại bỏ box min < 5×5 pixel và box > 80% diện tích ảnh (phòng box explosion).
  - Lọc class theo `class_mapping`, remap COCO class ID → Dataset class ID.
- Lấy đặc trưng `teacher_feat_for_mse` ra từ `teacher_feature_hook` (hook cắm trên Teacher).

**Bước 5: Tính toán Các Thành phần Loss (Luồng Cập nhật Loss)**
Có 5 thành phần Loss định hướng quá trình học:

1. **Detection Losses (`loss_source`, `loss_source_fake`)**: Loss cơ bản YOLO tính bằng `FDALoss` — wrapper của `E2ELoss` (YOLO26, NMS-free) hoặc `v8DetectionLoss` (YOLOv8/11 có NMS). Chứa Box loss, CLS loss với trọng số `box_gain` và `cls_gain`.

2. **Domain Match Loss — GRL (`loss_domain`)**: 
   - Dùng `compute_domain_loss()` — BCE with **Label Smoothing** (`0.1` mặc định): source label = 0.9, target label = 0.1. KHÔNG dùng Focal loss (mặc định `use_focal=False`).
   - Loss được nhân với `effective_grl_weight` (có cơ chế adaptive boost).
   - GRL đảo ngược gradient: Discriminator cố phân biệt Source/Target, nhưng gradient truyền ngược về backbone bị đảo dấu (×`-alpha`) → ép backbone học feature domain-invariant.

3. **Pseudo-label Distillation Loss (`loss_distillation`)**:
   - `compute_loss.compute_distillation_loss(pred_target, pseudo_labels, img_shape)`:  
     Tính detection loss giữa prediction của Student trên `imgs_target` (Target Real!) so với pseudo-labels do Teacher sinh ra trên `imgs_target_fake`.
   - Nhân với `current_lambda` (có thể progressive hoặc cố định).

4. **Feature Knowledge Distillation — Feature Cosine (`loss_feature_mse`)**:
   - Ép tính tương đồng phân bố ở mức feature maps giữa `student_feat_for_mse` (Student chạy trên **Target Real**) và `teacher_feat_for_mse` (Teacher chạy trên **Target Fake**).
   - Sử dụng **Cosine Similarity** (`1.0 - cos_sim.mean()`) thay vì L2/MSE thuần, giá trị nằm trong [0, 2], an toàn. Thực hiện align spatial size bằng bilinear interpolation nếu shape khác nhau.
   - Chỉ kích hoạt khi `epoch >= burn_in_epochs`.
   - Nhân với `lambda_feature`.

5. **Consistency Loss (`loss_consistency`)**:
   - `loss_consistency = clamp(loss_source.detach() - loss_source_fake, -10, 10) ** 2`.
   - Ràng buộc mạng dự đoán ổn định giữa ảnh thực Source và ảnh Fake Source (chuyển đổi miền), chống chênh lệch. Lưu ý `loss_source` được **detach** trước khi tính diff (chỉ truyền gradient qua `loss_source_fake`).
   - `consistency_weight = 1.0` (hard-coded).

**Total Loss**:
```
Loss = L_source + L_source_fake 
     + λ_box × L_distill 
     + λ_feat × L_feat_mse 
     + 1.0 × L_consistency 
     + L_domain
```
Loss được clamp trong khoảng [0, 500] để tránh explosion.

**Bước 6: Cập nhật Gradient và Weight (Optimization Flow)**
- **AMP Mixed Precision**: `scaler.scale(loss).backward()`.
- **Gradient Unscale**: `scaler.unscale_(optimizer)` và `scaler.unscale_(grl_optimizer)` (nếu active).
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_` với `max_norm=gradient_clip` (default `2.0`) cho cả Student và Discriminator.
- **NaN/Inf Gradient Check**: Duyệt toàn bộ `named_parameters()` của Student. Nếu phát hiện NaN/Inf → zero_grad cả 2 optimizer, skip iteration.
- **Student Optimizer step**: `scaler.step(optimizer)`.
- **Discriminator Optimizer step**: `scaler.step(grl_optimizer)` — nhưng CHỈ khi `domain_acc <= 0.85` (Discriminator Throttling). Nếu acc > 0.85, D bị tạm dừng update.
- **Scaler Update**: `scaler.update()`.
- **Zero Grad**: Cả `optimizer` và `grl_optimizer`.
- **Cập nhật Teacher**: Gọi `teacher_ema.update(model_student)` SAU optimizer.step(). EMA cập nhật cả **parameters** lẫn **buffers** (BatchNorm running_mean, running_var) — KHÁC với nhiều implementation chỉ update parameters.
- `prev_domain_acc = domain_acc` để mang sang iteration tiếp theo cho GRL weight boost.
- Mỗi 100 iteration: `torch.cuda.empty_cache()`.

---

## 3. Các Logic Kiểm soát An toàn Đặc trưng

- **Self-Validation Gating (Màng lọc tín hiệu Teacher)**:
  - Cứ mỗi **5 epochs** (tại batch đầu tiên `i == 0`), cho Teacher inference trên ảnh **Source** (dùng cùng `imgs_source` hiện tại).
  - Lần đầu: lưu `source_loss_baseline`.
  - Nếu `teacher_source_loss > 2.0 * source_loss_baseline` (chất lượng tụt thảm hại), thuật toán gọi `teacher_ema.pause_updates(steps=500)` — **pause EMA update trong 500 steps** để Teacher không bị cập nhật theo Student có thể đang bị hỏng nát.

- **Curriculum Learning for Pseudo-box Target**:
  - Dùng `get_adaptive_conf_thres()` — cosine schedule:
    - Trong burn-in (`epoch < burn_in_epochs`): trả về `max_conf` (0.7) — threshold rất gắt gao.
    - Sau burn-in: bắt đầu từ `conf_thres` (0.1 — lỏng) rồi cosine tăng dần lên `conf_thres_max` (0.7).
    - **Lưu ý**: Hàm thực tế đi từ lỏng → chặt (conf_thres → conf_thres_max), NGƯỢC với mô tả phổ biến "từ chặt → lỏng". Lý do: Giai đoạn đầu Teacher mới bắt đầu sinh nhãn (chưa tốt lắm), dùng threshold thấp để lấy nhiều boxes hơn cho đa dạng, sau đó nâng threshold lên để chỉ giữ boxes chất lượng cao khi Teacher đã ổn định.
  - Vứt bỏ các hộp > 80% diện tích ảnh và < 5×5 pixel.
  - `max_det=20` trong NMS.

- **Discriminator Throttling**: Khi `domain_acc > 0.85`, Discriminator không được update (chỉ backbone nhận reversed gradient). Đợi domain_acc giảm xuống mới tiếp tục update D.

- **Validation & Checkpointing**:
  - Validation mỗi 10 epoch hoặc epoch cuối. Tạo bản sao val model riêng (khônsg dùng Student trực tiếp) để tránh ảnh hưởng BatchNorm.
  - Best model saved theo `mAP@50`.
  - Intermediate checkpoints tại các epoch chỉ định (default: 0, 50, 100, 150, epoch cuối) — lưu cả Student + Teacher state dict.

---

## 4. Giải nghĩa Param/Option Tùy chỉnh (Trong `parse_args`)

Các tuỳ chọn (`argparse` parameters) quyết định sâu sắc đến hình thái kiến trúc của pipeline. Dưới đây là ý nghĩa và sự khác biệt về vai trò của chúng:

### A. Teacher-Student & Pseudo-labeling
- `--teacher-alpha` (0.9999): Trọng số EMA của Teacher, càng lớn Teacher cập nhật càng chậm và bảo thủ.
- `--conf-thres` (0.1) & `--conf-thres-max` (0.7): Hệ thống dùng hàm `get_adaptive_conf_thres` — cosine schedule từ `conf_thres` tăng dần lên `conf_thres_max` sau burn-in. Trong burn-in, luôn dùng `conf_thres_max`.
- `--iou-thres` (0.45): IoU threshold cho NMS khi sinh pseudo-labels.
- `--lambda-weight` (0.1): Trọng số cho Pseudo-label Distillation Loss (bounding box KD).
- `--burn-in-epochs` (50): Thời gian "cấm vận" — trong X epoch đầu, Student chỉ được học từ GT Source (Teacher không sinh pseudo-labels, tránh Confirmation Bias giai đoạn đầu).
- `--freeze-teacher`: Nếu bật, Teacher giữ nguyên pretrained weight, không EMA update.
- `--use-progressive-lambda`: Bật progressive lambda — lambda tăng quadratic trong warmup rồi giữ nguyên.
- `--warmup-epochs` (10): Số epoch warmup cho progressive lambda.

### B. GRL (Gradient Reversal Layer) & Adversarial Training
- `--use-grl`: Bật Domain Adaptation bằng Adversarial Discriminator.
- `--use-multiscale-grl`: Sử dụng `MultiScaleFusedDiscriminator` — hook backbone tại P3/P4/P5, mỗi scale qua GRL riêng (alpha multiplier 1.5/1.0/0.5), qua bottleneck (256/128/64), concat → 1 MLP chung (Spectral Norm). P3 chiếm 57% input → bias mạnh cho small objects. ~15% compute thêm.
- `--grl-warmup` (20): Epoch trước khi GRL kích hoạt. Cho backbone kịp học Feature Map trước khi bắt đầu adversarial training.
- `--grl-max-alpha` (1.0): Alpha tối đa cho GRL schedule (DANN progressive: sigmoid schedule).
- `--grl-weight` (0.05): Trọng số tác động của GRL Loss vào Total Loss. Có cơ chế adaptive boost tối đa ×3 khi `prev_domain_acc > 0.75`.
- `--grl-hidden-dim` (512): Tham số hidden dim cho Discriminator MLP.
- `--grl-dropout` (0.1): Dropout cho Discriminator MLP.
- `--grl-lr` (0.00005): Learning rate riêng cho Discriminator (Adam optimizer).

### C. Feature Distillation
- `--use-feature-distill`: Bật Cosine Similarity Feature KD. Dùng feature Tensor ở layer backbone cuối: ép `Student(Target Real)` features "trông giống" `Teacher(Target Fake)` features. Hook hoàn toàn tách biệt với GRL hooks.
- `--lambda-feature` (0.05): Trọng số sức mạnh của Feature Cosine Loss.

### D. Kiến trúc Hàm Loss riêng dành cho Tiểu Đối tượng (Small Objects)
- `--use-small-object-loss`: Bật cơ chế thay IoU tiêu chuẩn bằng Inner-CIoU hoặc ScaleAware TAL. Tập trung tối ưu vào khu vực Box cực nhỏ.
- `--inner-ratio` (0.7): Kiểm soát tỉ lệ thu nhỏ auxiliary box cho Inner-CIoU.
- `--use-wise-iou`: Bật WiseIoU v3 reweighting — focus vào trường hợp khó (outlier) nhưng không làm bùng nổ gradients.
- `--box-gain` (7.5): Trọng số Box/IoU loss.
- `--cls-gain` (0.5): Trọng số Classification loss.
- `--gradient-clip` (2.0): Max norm cho gradient clipping (không có trong argparse, dùng `getattr` với default 2.0).

### E. Các tham số khác
- `--config`: YAML config file — nếu cung cấp, sẽ merge với CLI args.
- `--cfg`: Custom model YAML (ví dụ `yolov8-p2.yaml`).
- `--weights` (yolo26s.pt): Pretrained weights.
- `--data` (data_v8.yaml): Data config YAML.
- `--imgsz` (640): Image size.
- `--epochs` (200), `--batch` (16), `--device` ('0'), `--workers` (8).
- `--lr0` (0.0001): Initial learning rate. Scheduler: Cosine Annealing (`LambdaLR`).
- `--lrf` (0.01): Final learning rate ratio.
- `--enable-monitoring`: Bật DomainMonitor (UMAP/MMD visualization).
- `--amp`: Bật Automatic Mixed Precision.
- `--project` (runs/fda), `--name` (exp): Output directory.

---

## Tổng kết Kiến trúc (Summary)

`train.py` là một cơ chế đồng bộ hoành tráng hai vòng lặp lớn đan xen:
1. **Adversarial Flow (GRL)**: Student Backbone chiến đấu với Discriminator để nhòa ranh giới ảnh thật (Source/Target). Discriminator bị throttle khi quá mạnh (acc > 0.85), GRL weight tự boost khi discriminator đang thắng (acc > 0.75).
2. **Collaborative Flow (Distillation)**: Student Networks bám theo sự chỉ dẫn cẩn trọng, từ từ của Teacher Model trên ảnh chưa dán nhãn (Target). Cả Pseudo-label KD lẫn Feature Cosine KD đều có gating mechanism: burn-in delay, EMA ramp-up, quality validation pause.

Hội tụ lại, sự phức tạp của code thể hiện qua:
- Xử lý corner case triệt để: NaN/Inf gradient check, loss clamp [0, 500], teacher NaN detection, box explosion prevention.
- EMA update cả parameters lẫn buffers (BatchNorm), có cosine ramp-up và pause mechanism.
- Feature hooks đóng gói gọn gàng, tách biệt GRL và KD hooks, không can thiệp sâu phá vỡ API gốc của Ultralytics YOLO.
- AMP mixed precision xuyên suốt cả Student lẫn Discriminator.
- Validation tạo bản sao model riêng để không ảnh hưởng training state.
