# `train.py` — Giải thích chi tiết

Tài liệu này đi qua **toàn bộ** [train.py](../train.py) từ trên xuống dưới. Mục tiêu sau khi đọc:

- Hiểu **một iteration FDA** chạy gồm những gì.
- Biết **vì sao** lại có 4 forward, 4 loss, BN freeze, paired-augmentation, teacher gate, student gate, GRL boost/throttle, …
- Lần ngược được mỗi biến cục bộ về **nguồn gốc** và **vai trò**.

---

## 0. Khái niệm nền tảng (đọc trước nếu bạn mới)

Một số thuật ngữ sẽ xuất hiện liên tục. Tóm tắt nhanh để khỏi tra ngoài:

| Khái niệm | Giải thích ngắn |
|---|---|
| **Domain Adaptation (DA)** | Tập huấn (source) khác phân phối với tập triển khai (target). DA giúp model học được "đặc trưng bất biến" giữa hai miền. |
| **CycleGAN** | Mạng GAN không cần cặp ảnh; học hàm dịch style 2 chiều `source ↔ target`. Trong FDA, dùng để sinh `source_fake` (ảnh nguồn nhuộm style target) và `target_fake` (ảnh đích nhuộm style nguồn). |
| **Teacher–Student / EMA** | Một bản sao chậm của model (teacher) được tính bằng trung bình cộng có trọng số mũ (EMA) của student. Teacher mượt hơn → sinh pseudo-label ổn định hơn. |
| **Pseudo-label** | Nhãn tạo ra bằng cách cho teacher dự đoán trên dữ liệu **không nhãn** (target), rồi dùng làm GT cho student. |
| **Knowledge Distillation (KD)** | Học từ output của một model khác (teacher) thay vì từ nhãn người gán. Ở đây KD = detection loss với pseudo-label. |
| **GRL — Gradient Reversal Layer (DANN)** | Lớp đặc biệt: forward y = x, backward y' = −λx. Đặt giữa backbone và "domain discriminator". Discriminator học phân biệt source/target; backbone bị **đảo gradient** nên học để **lừa** discriminator → ép feature mất dấu vết miền. |
| **Burn-in** | Vài epoch đầu chỉ học từ GT (source), **chưa** dùng pseudo-label, để tránh confirmation bias khi teacher còn yếu. |
| **BN freezing** | Đóng băng `running_mean`/`running_var` của BatchNorm (cho BN về `eval()` ngay cả khi model đang `train()`). Bảo vệ thống kê BN COCO khỏi bị batch nhỏ làm hỏng. |
| **Paired augmentation** | Cùng một seed RNG cho cặp `(source_real, source_fake)` và `(target_real, target_fake)` khi augment (mosaic/flip/crop) → sau augment vẫn pixel-aligned. Điều kiện cần để pseudo-label sinh trên `target_fake` dùng được cho prediction trên `target_real`. |
| **AMP — Automatic Mixed Precision** | Tự động chuyển một phần tính toán sang FP16, kết hợp `GradScaler` để giữ chính xác số → nhanh hơn + ít VRAM. |
| **DFL — Distribution Focal Loss** | YOLOv8/26 không hồi quy thẳng tọa độ box mà dự đoán **phân phối rời rạc** trên một dải, rồi lấy kỳ vọng → ổn định hơn. `_get_decode_boxes` giải mã từ DFL về xyxy. |
| **one2one / one2many** | YOLO26 E2E có 2 head song song: `one2many` (TAL topk=10, conf cao, để train) và `one2one` (TAL topk=1, sparse, conf thấp, để deploy NMS-free). |
| **NMS (Non-Maximum Suppression)** | Loại bỏ các box trùng lặp. `agnostic=True` = bỏ qua class khi tính trùng. |

---

## 1. Pipeline ở mức cao

FDA = **Fusion Domain Adaptation** — semi-supervised domain-adaptive object detection cho YOLO26 / YOLOv8.

### Bốn miền dữ liệu

| Miền | Có nhãn? | Sinh ra từ đâu | Vai trò trong loss |
|---|---|---|---|
| `source_real` | ✓ | Ảnh nguồn gốc | Học detection (GT) |
| `source_fake` | ✓ | CycleGAN: source → style target | Học detection (GT), thúc feature **bất biến style** |
| `target_real` | ✗ | Ảnh đích gốc | Học qua pseudo-label + GRL |
| `target_fake` | ✗ | CycleGAN: target → style source | Teacher chạy trên đây để sinh pseudo-label |

### Bốn forward pass mỗi iteration

```
1. student(source_real)   → loss_source          (GT)
2. student(source_fake)   → loss_source_fake     (GT)
3. student(target_real)   → loss_distillation    (pseudo từ teacher)
4. teacher(target_fake)   → pseudo-labels        [no_grad]
```

### Năm thành phần loss

```
loss_source       — detection trên source_real (GT)
loss_source_fake  — detection trên source_fake (GT)            [BN frozen]
loss_distillation — detection trên target_real với pseudo-label [BN frozen]
loss_consistency  — 1 − cosine(feat_sf, feat_sr.detach())
loss_domain       — GRL adversarial giữa feat_sr và feat_tr
```

### Bản đồ gradient (chốt sớm để khỏi lăn tăn)

| Tensor | Nguồn | Vào loss | Có gradient vào student? |
|---|---|---|---|
| `pred_source` | Forward 1: `student(source_real)` | `loss_source` (GT) | ✓ bình thường |
| `feat_sr` | Forward 1: `student(source_real)` | GRL (đảo dấu) | ✓ qua GRL |
| `feat_sr` | Forward 1: `student(source_real)` | Consistency (anchor, `.detach()`) | ✗ |
| `pred_source_fake` | Forward 2: `student(source_fake)` | `loss_source_fake` (GT) | ✓ bình thường |
| `feat_sf` | Forward 2: `student(source_fake)` | Consistency (learner) | ✓ bình thường |
| `pred_target` | Forward 3: `student(target_real)` | `loss_distillation` (pseudo-label là target) | ✓ bình thường |
| `feat_tr` | Forward 3: `student(target_real)` | GRL (đảo dấu) | ✓ qua GRL (ngược chiều) |
| `pseudo_labels` | Forward 4: `teacher(target_fake)` [no_grad] | Distillation (làm **target**, không phải input) | ✗ |

> **Lưu ý Forward 3:** `pred_target` và `feat_tr` đến từ **cùng một forward pass**. Trong backward, cả `loss_distillation` lẫn `loss_domain` đều tạo gradient qua student backbone ở forward 3 — điều này hoàn toàn hợp lệ vì PyTorch cộng dồn gradient từ cùng một computational graph. Đây **không** phải double-gradient theo nghĩa xấu (một tensor vào hai loss): hai gradient đi qua hai con đường tính toán khác nhau (`pred_target` vs `feat_tr`) trong cùng một forward.

**Không có double-gradient trên cùng tensor**: mỗi tensor cụ thể chỉ vào đúng 1 loss có gradient. Teacher backbone hook chỉ được **drain**; feature teacher không tham gia loss nào.

---

## 2. Imports & hằng số

### Module imports chính

- `torch`, `torch.optim`, `torch.cuda.amp` — training + AMP.
- `ultralytics`:
  - `YOLO` — wrapper model + train/val.
  - `LOGGER`, `colorstr` — logging.
  - `xywh2xyxy`, `non_max_suppression` — utilities.
- `domain_adaptation` (project local):
  - `DomainDiscriminator` — MLP single-scale (1 feature map từ cuối backbone).
  - `YOLOv8FeatureHook` — hook lên một layer; lưu activation lần forward gần nhất.
  - `find_last_backbone_layer` — định vị layer index trên model YOLO.
  - `compute_domain_loss`, `get_grl_alpha`, `get_domain_accuracy` — utilities GRL.
- `fusion_da` (project local):
  - `FDALoss` — loss tổng (bọc box/cls/dfl của Ultralytics + distillation).
  - `WeightEMA` — teacher EMA của student.
  - `PairedMultiDomainDataset` — dataset 4 miền, đồng bộ RNG cho cặp.
- `utils.FDA_helpers` — `filter_pseudo_labels_by_uncertainty`, `get_progressive_lambda` (import nhưng hiện ít dùng).
- `utils.training_logger.TrainingLogger` — log step/epoch ra TensorBoard + file.
- `utils.domain_monitor.DomainMonitor` — vẽ UMAP feature của 3 miền cuối epoch.
- `cv2`, `numpy` — vẽ ảnh debug.

### Hằng số

```python
TEACHER_CONF_THRES = 0.5
```

Ngưỡng confidence **cố định** cho pseudo-label. Không dùng schedule động vì burn-in đã chặn pseudo-label noisy ở giai đoạn đầu — fixed threshold đơn giản và dễ đoán hơn.

---

## 3. Helpers cấp file

### `_freeze_bn(model)` / `_unfreeze_bn(model)`

Đặt mọi BatchNorm về `eval()` (đóng băng `running_mean`/`running_var`) hoặc `train()`.

**Tại sao cần?** Với batch=4 (hoặc 16), momentum=0.1 của BN khiến running stats bị **vài batch gần nhất** chi phối. Khi sang `eval()` (lúc validate), những running stats đã hỏng → mAP sụp trên **tất cả** miền. Giải pháp: cho running stats đứng yên ở stats COCO pretrain, chỉ cho conv/linear weights học. Lý do freeze cho **mọi** forward (kể cả `source_real`): batch nhỏ → ảnh từ bất kỳ miền nào cũng có thể làm running stats lệch.

### `validate_model(tag, state_dict, weights_path, data_yaml, split, device, dataset_label)`

Tạo một **YOLO instance mới** từ `weights_path`, nạp `state_dict` cần đánh giá vào, gọi `.val()` của Ultralytics.

Tham số:
- `tag` — nhãn người dùng đọc (`'STUDENT'`, `'TEACHER'`, `'STUDENT-INIT'`, `'TEACHER-GATE'`, …).
- `state_dict` — đang đánh giá weights của ai (cùng kiến trúc, khác trọng số).
- `weights_path` — file `.pt` để **dựng lại wrapper YOLO** (cấu trúc + names + stride).
- `data_yaml`, `split` — dataset nào, split nào.
- `dataset_label` — mô tả người đọc (vd `'target_real (test split)'`).

Output: in `mAP50`, `mAP50-95`, `P`, `R`. Nếu `mAP50 < 0.1` → in thêm per-class — bắt lỗi class-index lệch (mapping COCO ↔ project).

`finally`: xoá wrapper + `gc.collect()` + `empty_cache()` để không rò RAM/VRAM (mỗi lần gọi tạo cả model mới).

### `save_debug_image(img_tensor, pseudo_labels, save_path, names, conf_thres=0.5)`

Vẽ bbox lên ảnh để **kiểm tra bằng mắt** xem teacher đang detect cái gì.

Bước:
1. Tensor `[3, H, W]` float `[0,1]` → numpy `uint8` BGR.
2. Lặp qua từng box, lọc theo `conf_thres`, vẽ rectangle + label.
3. Ghi `Count: N | Conf >= 0.50` lên góc trên.
4. Lưu vào `save_path` (tự tạo thư mục).

Trả về số box thực sự vẽ (sau khi lọc).

---

## 4. `train(opt)` — hàm chính

Chia 4 phần:
- **4.1** Setup (device, models, hooks, GRL, optimizer, data, loss, logger).
- **4.2** State trước vòng lặp + baseline validation.
- **4.3** Vòng lặp epoch.
- **4.4** Cleanup.

---

### 4.1 Setup

#### 4.1.1 Device & save dir

```python
device   = torch.device(f'cuda:{opt.device}' if opt.device.isdigit() else opt.device)
save_dir = Path(opt.project) / opt.name
```

`opt.device` là string (`'0'`, `'cpu'`, …). `save_dir = runs/fda/<name>/`, có subdir `weights/`.

#### 4.1.2 Data config

`yaml.safe_load(opt.data)` → dict gồm `nc` (số class), `names` (list tên class), `path` (root dataset), và 4 key đường dẫn 4 miền.

#### 4.1.3 Student model

- Có `opt.cfg` (file `.yaml` mô tả kiến trúc) → tạo `YOLO(cfg)` rồi `.load(weights)` (ghép weights pretrain vào kiến trúc tuỳ biến, vd `yolov8-p2.yaml`).
- Không có → `YOLO(weights)` (kiến trúc lấy từ checkpoint).

`model_student = yolo_student.model.to(device)` — `nn.Module` thật. Đặt `requires_grad=True` cho **mọi** param (đề phòng checkpoint freeze sẵn).

#### 4.1.4 Teacher EMA

```python
teacher_ema = WeightEMA(
    model_student,
    alpha               = opt.teacher_alpha,        # 0.9999
    device              = device,
    update_after_step   = 500,
    alpha_rampup_steps  = 2000,
)
model_teacher = teacher_ema.module
```

`WeightEMA`:
- `.module` — teacher thực (`nn.Module`, cùng kiến trúc student).
- `.update(student)` — cập nhật `θ_t = α·θ_t + (1-α)·θ_s`. Gọi **sau** `optimizer.step()`.
- `update_after_step=500` — 500 step đầu teacher chưa update → tránh teacher bị kéo theo student còn non.
- `alpha_rampup_steps=2000` — α tăng dần từ ~0 đến `teacher_alpha` trong 2000 step đầu → teacher "đuổi" student nhanh lúc đầu rồi mới ổn định.

#### 4.1.5 Backbone hooks

```python
student_backbone_idx  = find_last_backbone_layer(model_student)
student_backbone_hook = YOLOv8FeatureHook(model_student, layer_idx=student_backbone_idx)
teacher_backbone_idx  = find_last_backbone_layer(model_teacher)
teacher_backbone_hook = YOLOv8FeatureHook(model_teacher, layer_idx=teacher_backbone_idx)
```

`YOLOv8FeatureHook` đăng ký `forward_hook` lên layer chỉ định. Mỗi lần forward, activation `[B, C, H, W]` được lưu. Gọi `.get_features()` để **lấy ra và xoá** (drain).

> **Quy tắc vàng:** sau mỗi forward, phải `get_features()` **trước** forward tiếp theo (vì hook bị overwrite). Teacher hook hiện chỉ được drain, không dùng vào loss.

#### 4.1.6 GRL discriminator (single-scale)

```python
if opt.use_grl:
    # Probe channel size bằng forward giả
    with torch.no_grad():
        test_img    = torch.zeros(1, 3, opt.imgsz, opt.imgsz, device=device)
        _           = model_student(test_img)
        test_feat   = student_backbone_hook.get_features()
        in_channels = test_feat.shape[1] if test_feat is not None else 256

    domain_discriminator = DomainDiscriminator(in_channels, opt.grl_hidden_dim, opt.grl_dropout).to(device)
    grl_optimizer = optim.Adam(domain_discriminator.parameters(), lr=opt.grl_lr)
```

**Tại sao probe?** Số channel của feature map tuỳ kiến trúc — phải biết trước khi khởi tạo MLP discriminator cho khớp.

#### 4.1.7 Checkpoint epochs

```python
checkpoint_epochs = set(getattr(opt, 'checkpoint_epochs', [10, 20, 30]))
checkpoint_epochs.add(opt.epochs - 1)
```

Tập epoch sẽ lưu weight (cả student + teacher) cho **phân tích chất lượng pseudo-label**. Luôn thêm epoch cuối.

#### 4.1.8 Optimizer + scheduler

```python
optimizer = optim.AdamW(model_student.parameters(), lr=opt.lr0, weight_decay=0.0005)

warmup_epochs_lr = 5
def lf(x):
    if x < warmup_epochs_lr:
        return (x + 1) / warmup_epochs_lr                                 # tuyến tính 0.2 → 1.0
    progress = (x - warmup_epochs_lr) / max(opt.epochs - warmup_epochs_lr, 1)
    return opt.lrf + (1.0 - opt.lrf) * (1 + math.cos(math.pi * progress)) / 2
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
```

- 5 epoch đầu: LR tăng tuyến tính.
- Sau đó: cosine decay từ `1.0·lr0` xuống `lrf·lr0`.
- **Lý do**: pure cosine cũ rớt LR xuống 1e-6 ở epoch ~19/100 → student ngừng học. Warmup + clamp dưới (`lrf`) khắc phục.

#### 4.1.9 Resolve data paths

```python
def get_paths(key, fallback=[]):
    paths = data_dict.get(key, fallback)
    if isinstance(paths, str): paths = [paths]
    return [str(root / p) for p in paths] if paths else []

source_real = get_paths('train_source_real', data_dict.get('train', []))
source_fake = get_paths('train_source_fake')
target_real = get_paths('train_target_real')
target_fake = get_paths('train_target_fake')
```

`train_source_real` fallback về `train` (tương thích YAML chuẩn YOLO).

#### 4.1.10 Dataset + DataLoader

```python
gs          = max(int(model_student.stride.max()), 32)   # grid size (stride lớn nhất)
default_cfg = get_cfg()                                  # hyperparams Ultralytics

paired_dataset = PairedMultiDomainDataset(
    source_real_path, source_fake_path, target_real_path, target_fake_path,
    imgsz=opt.imgsz, augment=True, hyp=default_cfg, data=data_dict, stride=gs,
)
paired_loader  = DataLoader(paired_dataset, batch_size=opt.batch, shuffle=True,
                            num_workers=opt.workers, pin_memory=True,
                            collate_fn=PairedMultiDomainDataset.collate_fn)
```

- `augment=True` **quan trọng**: trước đây là `False` → student overfit nặng (không có regularization). Bật mosaic/flip/scale nhưng **đồng bộ RNG** giữa cặp.
- `collate_fn` của class trả batch dạng `dict` 4 key (`source_real`, `source_fake`, `target_real`, `target_fake`), mỗi key là một batch YOLO chuẩn (`{'img': ..., 'cls': ..., 'bboxes': ...}`).
- `nb = len(paired_loader)` — số batch trong 1 epoch.

#### 4.1.11 Loss object

```python
class_mapping = getattr(opt, 'class_mapping', {0: 0, 1: 1, 2: 1})
compute_loss  = FDALoss(model_student, class_mapping, box_gain, cls_gain)
```

`class_mapping`: COCO class id → project class id (vd map COCO 1=bicycle và 2=car cùng vào class 1 của project). Dùng cho **filter pseudo-label** — chỉ giữ box thuộc class mình quan tâm.

`FDALoss` cung cấp:
- `__call__(pred, batch)` → detection loss (CIoU + BCE + DFL).
- `compute_distillation_loss(pred, pseudo_labels, img_hw)` → detection loss với target là pseudo-label.

#### 4.1.12 AMP

```python
use_amp = opt.amp and device.type != 'cpu'
scaler  = amp.GradScaler(enabled=use_amp)
```

Nếu CPU thì tắt AMP. `scaler` quản lý loss-scaling khi backward + step.

#### 4.1.13 Logger + DomainMonitor

- `TrainingLogger` — write TB nếu không phải `tuning_mode` (tránh phình disk khi sweep).
- `DomainMonitor` (chỉ bật khi `--enable-monitoring`):
  - Lập danh sách `umap_epochs` thưa: `[0, mid, last]` cho ≤50 epoch, hoặc 5 mốc cho >50.
  - Cuối mỗi epoch trong danh sách, vẽ UMAP các feature 3 miền (`sr`/`sf`/`tr`) — quan sát có "trộn" được không. Nếu các điểm cuộn vào nhau → DA đang ổn.

---

### 4.2 State trước vòng lặp

#### Banner pipeline

In ra mode hiện tại (`baseline` hay full FDA) và công thức loss.

#### Biến trạng thái

| Biến | Ý nghĩa |
|---|---|
| `best_fitness` | mAP@50 cao nhất của student trên target → chọn `best.pt` |
| `t0` | timestamp bắt đầu (đo tổng thời gian) |
| `global_step` | đếm step xuyên suốt training (cho TB) |
| `burn_in_epochs` | số epoch chỉ học GT, **chưa** dùng pseudo-label (mặc định 5 từ CLI, fallback 10) |
| `source_loss_baseline` | loss teacher trên `source_real` ở lần check đầu → gate phát hiện degrade |
| `teacher_map_baseline` | mAP@50 đầu tiên đo của teacher trên target → gate phát hiện degrade |
| `prev_domain_acc` | accuracy discriminator iteration trước → dùng adaptive GRL weight |
| `best_teacher_map50`, `best_teacher_state` | snapshot teacher tốt nhất → **restore** khi teacher collapse |
| `teacher_quality_ok` | `True` khi teacher đạt mAP@50 ≥ 0.30 → mới cho phép distillation |
| `best_student_map50`, `best_student_state` | đối xứng với teacher, để restore student khi nó collapse |

#### Baseline validation (epoch 0)

Trước khi train, validate `student-init` và `teacher-init` trên target test split (và source nếu `--eval-source`). Mục đích: phân biệt "model khởi tạo đã yếu" vs "training làm hỏng".

---

### 4.3 Vòng lặp `for epoch in range(opt.epochs)`

#### 4.3.1 Đầu mỗi epoch

```python
model_student.train()
_freeze_bn(model_student)    # BN frozen ngay cả khi train()
model_teacher.eval()

if opt.use_grl:
    if domain_discriminator is not None:
        domain_discriminator.train()
    current_grl_alpha = get_grl_alpha(epoch, opt.epochs, opt.grl_warmup, opt.grl_max_alpha)
```

`current_grl_alpha`: hệ số nhân vào gradient bị reverse trong GRL. Schedule kiểu DANN: `2/(1+exp(-γ·p)) - 1`, ramp từ 0 → `grl_max_alpha`.

**Lambda distillation schedule:**

```python
lambda_min = 0.02
if epoch < burn_in_epochs:
    current_lambda = 0.0
else:
    ramp_epochs = min(5, (opt.epochs - burn_in_epochs) // 3)
    progress    = min((epoch - burn_in_epochs) / max(ramp_epochs, 1), 1.0)
    current_lambda = lambda_min + (opt.lambda_weight - lambda_min) * (1 - cos(π·progress)) / 2
```

- Burn-in: `λ = 0` (không distillation).
- Sau burn-in: cosine ramp từ `lambda_min=0.02` đến `opt.lambda_weight` trong vài epoch.
- `lambda_min > 0` để distillation **active ngay khi qua burn-in** — tránh student lạc vào source-only trước khi DA kịp tác động.

#### 4.3.2 State trong epoch

```python
mloss        = torch.zeros(5, device=device)    # moving avg [box, cls, dfl, distill, domain]
paired_iter  = iter(paired_loader)
pbar         = tqdm(range(nb), desc=f'Epoch {epoch}/{opt.epochs-1}')
optimizer.zero_grad()
```

#### 4.3.3 Đọc batch & chuẩn hoá

`try/except StopIteration`: tự reset iterator phòng khi sampler hết sớm.

```python
imgs_source      = batch['source_real']['img'].to(device).float() / 255.0
imgs_source_fake = batch['source_fake']['img'].to(device).float() / 255.0
imgs_target      = batch['target_real']['img'].to(device).float() / 255.0
imgs_target_fake = batch['target_fake']['img'].to(device).float() / 255.0
batch_source     = batch['source_real']        # giữ lại GT của source_real

imgs_*.clamp(0, 1)                             # an toàn (tránh pixel < 0 hay > 1 sau augment)
```

#### 4.3.4 Nhánh BASELINE (`--baseline`)

```python
pred_source = model_student(imgs_source)
loss, _     = compute_loss(pred_source, batch_source)
_           = student_backbone_hook.get_features()   # drain (vẫn phải để hook không stale)
scaler.scale(loss).backward(); ...; optimizer.step(); ...
continue                                             # bỏ qua toàn bộ DA bên dưới
```

Mục đích: làm chuẩn so sánh. Nếu baseline đã cao → DA không cần thiết. Nếu baseline thấp → vấn đề ở detection loss/data, không phải DA.

#### 4.3.5 Forward 1 — `student(source_real)`

```python
with torch.amp.autocast('cuda', enabled=use_amp):
    pred_source = model_student(imgs_source)
    loss_source, loss_items_source = compute_loss(pred_source, batch_source)
    feat_sr = student_backbone_hook.get_features()        # [B, C, H, W]
```

- `loss_source` — detection loss có gradient bình thường vào toàn bộ student (head + neck + backbone).
- `loss_items_source` — tuple `(box, cls, dfl)` để log riêng.
- `feat_sr` dùng cho 2 việc: (a) consistency anchor (sẽ `.detach()`), (b) GRL.

#### 4.3.6 Forward 2 — `student(source_fake)`

```python
sf_base_weight = getattr(opt, 'source_fake_weight', 0.1)
if sf_base_weight > 0:
    pred_source_fake = model_student(imgs_source_fake)
    loss_source_fake, _ = compute_loss(pred_source_fake, batch['source_fake'])
    feat_sf = student_backbone_hook.get_features()
else:
    loss_source_fake = torch.tensor(0.0, ...); feat_sf = None
```

Hai chú ý quan trọng:

1. **BN vẫn frozen** (đã frozen ở đầu epoch). Ảnh CycleGAN nhiễu — không được làm bẩn running stats.
2. **Dùng `batch['source_fake']`, không phải `batch_source`.** Trước đây dùng nhầm `batch_source` → label từ draw augmentation khác với ảnh → box lệch pixel ↔ ảnh → loss explode 3-4× → student học chỗ sai → target mAP sụp. Sau khi có paired-aug, mỗi miền có label "đúng pixel" của riêng nó.

Nếu `source_fake_weight=0` → bỏ hẳn forward này (tiết kiệm compute).

#### 4.3.7 Forward 3 — `student(target_real)` (có điều kiện)

```python
need_target_forward = (
    epoch >= burn_in_epochs                        # distillation cần
    or (opt.use_grl and epoch >= opt.grl_warmup)   # GRL cần feat_tr
)
if need_target_forward:
    pred_target = model_student(imgs_target)
    feat_tr     = student_backbone_hook.get_features()
else:
    pred_target = None; feat_tr = None
```

Bỏ forward 3 khi không cần → **tiết kiệm ~25% compute** + tránh BN/gradient nhiễu.

#### 4.3.8 Forward 4 — `teacher(target_fake)`

Chia 3 phase: **gate** → **predict** → **filter**.

##### A. Burn-in branch

```python
if epoch < burn_in_epochs:
    pseudo_labels = [zeros(0, 6) for _ in range(B)]      # rỗng
    n_pseudo      = 0
```

Burn-in = không distillation. Student học pure GT cho ổn định.

##### B. Self-validation gate (mỗi 5 epoch, batch đầu)

Chạy thử teacher trên `imgs_source` (ảnh có GT) → so sánh với baseline:

```python
teacher_source_loss, _ = compute_loss(model_teacher(imgs_source), batch_source)
```

Đồng thời gọi `validate_model('TEACHER-GATE', ...)` đo `teacher_map50_gate` trên target test.

Hai bài test:

1. **Loss test:** `teacher_source_loss > 2 × source_loss_baseline` → degrade.
2. **mAP test:** `teacher_map50_gate < 0.5 × teacher_map_baseline` → degrade.

Best teacher snapshot:

```python
if teacher_map50_gate > best_teacher_map50:
    best_teacher_map50 = teacher_map50_gate
    best_teacher_state = clone(model_teacher.state_dict())
```

Quality gate (cho phép distillation hay không):

```python
if teacher_map50_gate >= 0.30:  teacher_quality_ok = True
else:                           teacher_quality_ok = False
```

Nếu `gate_pause` (lý do degrade):
- Có snapshot tốt (`best > 0.1`) → **restore** + `teacher_ema.pause_updates(1000)`.
- Không có snapshot tốt → `pause_updates(500)`.

> **Pause EMA = không cập nhật teacher trong N step** → cắt feedback loop (student xấu → teacher EMA theo → pseudo-label xấu → student càng xấu).

##### C. Forward thực + decode

```python
with torch.no_grad():
    pred_teacher = model_teacher(imgs_target_fake)
    _ = teacher_backbone_hook.get_features()    # drain hook (không dùng)
```

YOLO26 ở `eval()` trả `(y_postprocessed, raw_dict)`:
- `y_postprocessed` — output **one2one** (sparse, conf thấp ~0.3) cho NMS-free deploy.
- `raw_dict` chứa cả `one2many` và `one2one` ở dạng raw.

**Code dùng `one2many`** vì conf cao hơn → pseudo-label chất lượng tốt hơn:

```python
o2m_preds   = raw_preds['one2many']
detect_head = model_teacher.model[-1]
dbox        = detect_head._get_decode_boxes(o2m_preds)      # raw DFL → xyxy

# xyxy → xywh vì non_max_suppression nhận xywh:
x1, y1, x2, y2 = dbox.chunk(4, dim=1)
dbox = cat([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=1)

cls_sigmoid = o2m_preds['scores'].sigmoid()
pred_tensor = cat([dbox, cls_sigmoid], dim=1)               # [B, 4+nc, N]

pseudo_labels = non_max_suppression(
    pred_tensor, conf_thres=TEACHER_CONF_THRES, iou_thres=opt.iou_thres,
    max_det=30, classes=list(class_mapping.keys()), agnostic=True,
)
```

**`agnostic=True`** *rất quan trọng*: nếu teacher detect cùng một object với cả COCO class 1 và 2 (conf khác nhau), NMS class-aware giữ cả hai → sau `class_mapping` chúng map về **cùng** class project → **duplicate** pseudo-label.

Có nhánh fallback cho YOLOv8 (non-E2E) — logic tương tự nhưng đọc từ `pred_teacher[0]`.

##### D. Filter pseudo-label

Bốn lớp lọc trên mỗi ảnh:

1. **Clamp** toạ độ vào `[0, img_w/h]`.
2. Loại box "explode" (`box_area ≥ 0.8 · img_area`).
3. Loại box quá nhỏ (`box_w ≤ 5` hoặc `box_h ≤ 5`).
4. Chỉ giữ class thuộc `class_mapping.keys()`.

```python
n_pseudo = sum(len(p) for p in pseudo_labels)
```

Debug image mỗi 100 iter (vẽ pseudo-label lên `target_fake`; do paired-RNG, cùng tọa độ trên `target_real` cũng đúng pixel).

**Quality gate cuối:**

```python
if not teacher_quality_ok:
    pseudo_labels = [zeros(0, 6) for _ in range(B)]
    n_pseudo = 0
```

Pause distillation tới khi teacher đạt mAP@50 ≥ 0.30.

#### 4.3.9 Tính loss

##### Loss 1 — Distillation

```python
if pred_target is not None:
    loss_distillation = compute_loss.compute_distillation_loss(
        pred_target, pseudo_labels, (img_h, img_w))
    # Cap để không lấn át loss_source:
    if loss_distillation.item() > loss_source.item() * 1.5:
        loss_distillation *= loss_source.item() * 1.5 / loss_distillation.item()
else:
    loss_distillation = 0
```

Cap 1.5× để loss_distillation cao bất thường (vd nhiều pseudo-label sai) không kéo gradient ra khỏi GT signal.

##### Loss 2 — Consistency

```python
if not opt.no_consistency and feat_sr is not None and feat_sf is not None:
    sr_anchor = feat_sr.detach()                          # anchor: GT-supervised, frozen
    sf_learn  = feat_sf                                   # learner: gradient flows
    if shape mismatch:
        sr_anchor = interpolate(sr_anchor, sf_learn.shape[2:])
    loss_consistency = 1.0 - cosine_similarity(sf_learn, sr_anchor, dim=1).mean()
```

Buộc feature của `source_fake` giống `source_real` về cosine — học **bất biến với CycleGAN style change**.

##### Loss 3 — Domain adversarial GRL

```python
if opt.use_grl and epoch >= opt.grl_warmup:
    effective_grl_weight = opt.grl_weight
    if prev_domain_acc > 0.75:                            # discriminator quá giỏi → tăng weight
        boost = 1.0 + 2.0 * (prev_domain_acc - 0.75) / 0.25
        effective_grl_weight = min(opt.grl_weight * boost, opt.grl_weight * 3.0)

    if feat_sr is not None and feat_tr is not None:
        domain_pred_source = domain_discriminator(feat_sr, current_grl_alpha)
        domain_pred_target = domain_discriminator(feat_tr, current_grl_alpha)
        loss_domain        = compute_domain_loss(...) * effective_grl_weight
        domain_acc         = get_domain_accuracy(...)
```

`compute_domain_loss` là BCE: source label=0, target label=1. Discriminator học phân biệt; backbone (qua GRL reverse) học để discriminator **không** phân biệt được.

`prev_domain_acc` lưu lại cuối iteration → boost weight nếu discriminator đang thắng quá → ép backbone học mạnh hơn.

##### Total loss

```python
sf_base_weight = opt.source_fake_weight
progress       = epoch / max(opt.epochs - 1, 1)
sf_weight      = sf_base_weight * max(0.2, 1.0 - progress * 0.8)   # decay 1.0 → 0.2

loss = (loss_source
        + loss_source_fake  * sf_weight
        + loss_distillation * current_lambda
        + loss_consistency  * consistency_weight
        + loss_domain)                                 # đã nhân effective_grl_weight rồi
loss = clamp(loss, 0, 500)
```

`sf_weight` decay vì CycleGAN sớm hữu ích (đa dạng) nhưng càng về sau càng dễ học artefact.

NaN/Inf check → bỏ iteration.

#### 4.3.10 Backward + step + EMA

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
if grl_optimizer_active: scaler.unscale_(grl_optimizer)

clip_grad_norm_(model_student.parameters(), max_norm=opt.gradient_clip)   # mặc định 2.0
if opt.use_grl and domain_discriminator is not None:
    clip_grad_norm_(domain_discriminator.parameters(), 2.0)

# NaN gradient check — skip iter nếu có
for name, p in model_student.named_parameters():
    if grad is NaN/Inf: skip

scaler.step(optimizer)

# Discriminator throttling:
if grl_optimizer_active:
    if domain_acc > 0.85:
        pass                                  # discriminator quá khoẻ → đừng update
    else:
        scaler.step(grl_optimizer)

scaler.update()
optimizer.zero_grad(); grl_optimizer.zero_grad()
teacher_ema.update(model_student)             # EMA SAU optimizer.step
prev_domain_acc = domain_acc
```

**Hai cơ chế cân bằng GRL** (đối xứng):
- **Boost** GRL weight khi `prev_domain_acc > 0.75` (discriminator đang thắng).
- **Throttle** discriminator update khi `domain_acc > 0.85` (chặn không cho mạnh thêm).

`empty_cache()` mỗi 100 iter — chống fragmentation VRAM.

#### 4.3.11 Logging trong loop

`mloss` là moving average 5 thành phần `[box, cls, dfl, distill, domain]` để in pbar.

`logger.log_iteration(epoch, i, log_dict, extra=...)` ghi vào TB:
- 5 loss + tổng.
- LR, GRL alpha, conf threshold, λ, `n_pseudo`, `domain_acc`.

Cảnh báo (chỉ log ở `i==0`):
- `loss_source_fake / loss_source > 3.0` → CycleGAN artefact dominant; gợi ý đặt `--source-fake-weight 0`.

DomainMonitor thu feature `sr/sf/tr` (mỗi 50 iter, max 50 batch đầu mỗi epoch) → vẽ UMAP cuối epoch. Phải `.detach()` trước khi đưa vào.

#### 4.3.12 Cuối epoch

```python
scheduler.step()
domain_monitor.end_epoch(epoch)               # vẽ UMAP nếu epoch trong list
gc.collect(); torch.cuda.empty_cache()
```

##### Validation định kỳ (mỗi 5 epoch hoặc epoch cuối)

1. **Student trên target** (`opt.data` test split) → `current_map50`, `current_map`.
2. **Student trên source** (chỉ khi `--eval-source` hoặc baseline) → `source_map50`. Chẩn đoán:
   - source thấp + target thấp = training broken.
   - source cao + target thấp = domain gap (kỳ vọng).
3. **Teacher trên target** (trừ baseline) → `teacher_map50_val`, `teacher_map_val`.

Diagnostics:
- Cả hai miền < 0.1 → cảnh báo training instability (kiểm BN, source_fake weight).
- `target < 30% × source` → in "Large domain gap".

`logger.log_epoch(...)` ghi metrics (kể cả `student_target_minus_source`).

##### Best student

```python
if current_map50 > best_fitness:
    best_fitness = current_map50
    save({'epoch','model','optimizer','best_fitness'}, 'weights/best.pt')
```

##### Student gate restore

Đối xứng với teacher gate:
- Lưu `best_student_state` mỗi khi mAP cao kỷ lục.
- Nếu `current_map50 < 50% × best_student_map50` (và `best > 0.1`) → **load lại** `best_student_state` + reset `optimizer.state = defaultdict(dict)`.

> Adam reset phải dùng `defaultdict(dict)`, **không** phải `defaultdict()` rỗng — `Adam._init_group` sẽ KeyError ở step kế.

##### Save last + intermediate

```python
if not tuning_mode: save 'weights/last.pt'
if epoch in checkpoint_epochs:
    save {student, teacher} → 'weights/checkpoint_ep{epoch:03d}.pt'
```

Tuning mode bỏ `last.pt` để khỏi đầy disk khi sweep.

---

### 4.4 Cleanup

```python
student_backbone_hook.remove()
teacher_backbone_hook.remove()

logger.finalize()
domain_monitor.finalize()

LOGGER.info(f'{opt.epochs} epochs in {hours:.2f} h')
LOGGER.info(f'Best mAP@50: {best_fitness:.4f}')
return best_fitness                            # cho hyperparam tuning
```

`hook.remove()` quan trọng: nếu giữ hook thì model vẫn lưu activation cho mọi forward sau này — leak RAM.

---

## 5. `parse_args()` — bảng tham số

### Model & data

| Cờ | Default | Ý nghĩa |
|---|---|---|
| `--cfg` | `None` | File `.yaml` kiến trúc tuỳ biến (vd P2 cho small object) |
| `--weights` | `yolo26s.pt` | Checkpoint khởi tạo |
| `--data` | `configs/data/data.yaml` | Dataset YAML |
| `--imgsz` | `640` | Input size |

### Training

| Cờ | Default | Ý nghĩa |
|---|---|---|
| `--epochs` | `100` | Số epoch |
| `--batch` | `16` | Batch size mỗi miền |
| `--device` | `'0'` | GPU index hoặc `'cpu'` |
| `--workers` | `8` | DataLoader workers |
| `--lr0` | `1e-4` | Base LR |
| `--lrf` | `0.01` | Tỉ lệ LR cuối / LR đầu |

### Teacher–Student

| Cờ | Default | Ý nghĩa |
|---|---|---|
| `--teacher-alpha` | `0.9999` | α EMA của teacher |
| `--conf-thres` / `--conf-thres-max` | `0.4` / `0.7` | (legacy, code dùng `TEACHER_CONF_THRES=0.5` cố định) |
| `--iou-thres` | `0.45` | IoU threshold cho NMS pseudo-label |
| `--lambda-weight` | `0.2` | Trọng số đỉnh của distillation |

### Burn-in & schedule

| Cờ | Default | Ý nghĩa |
|---|---|---|
| `--use-progressive-lambda` | `False` | (legacy) |
| `--warmup-epochs` | `10` | (legacy) |
| `--burn-in-epochs` | `5` | Số epoch chỉ học GT, chưa distillation |

### GRL

| Cờ | Default | Ý nghĩa |
|---|---|---|
| `--use-grl` | `False` | Bật adversarial DANN |
| `--grl-warmup` | `5` | Epoch trước khi GRL + consistency active |
| `--grl-max-alpha` | `1.0` | α tối đa của GRL |
| `--grl-weight` | `0.05` | Trọng số loss_domain |
| `--grl-hidden-dim` | `512` | Hidden dim MLP discriminator |
| `--grl-dropout` | `0.1` | Dropout discriminator |
| `--grl-lr` | `5e-5` | LR của discriminator (Adam riêng) |

### Output / Misc

| Cờ | Default | Ý nghĩa |
|---|---|---|
| `--project` / `--name` | `runs/fda` / `exp` | Thư mục lưu |
| `--enable-monitoring` | `False` | Bật UMAP DomainMonitor |
| `--amp` | `False` | Bật mixed precision |
| `--baseline` | `False` | Source-only mode (no DA) |
| `--eval-source` | `False` | Validate cả source mỗi 5 epoch |

### Consistency / Source-fake / Detection gains

| Cờ | Default | Ý nghĩa |
|---|---|---|
| `--consistency-weight` | `0.5` | Trọng số loss_consistency |
| `--no-consistency` | `False` | Tắt consistency |
| `--source-fake-weight` | `0.1` | Trọng số loss_source_fake (0 = tắt) |
| `--box-gain` | `7.5` | Box/IoU loss weight |
| `--cls-gain` | `0.5` | Classification loss weight |

---

## 6. `__main__`

```python
args = parse_args()
if args.config:
    config = load_config(args.config)             # YAML
    config = merge_cli_args(config, args)         # CLI thắng YAML
    args   = config_to_namespace(config)

print(banner)                                     # hiển thị mode + GRL + teacher
train(args)
```

Kết hợp **YAML config + CLI**: YAML để cố định nhiều siêu tham số, CLI để override khi cần — tiện sweep.

---

## 7. Cheat sheet: thứ tự sự kiện trong 1 iteration

```
 1. Forward 1: student(source_real)        → loss_source         feat_sr
 2. Forward 2: student(source_fake)        → loss_source_fake    feat_sf      (BN frozen)
 3. Forward 3: student(target_real)        → pred_target         feat_tr      (BN frozen, có điều kiện)
 4. (gate, mỗi 5 epoch)  validate teacher
 5. Forward 4: teacher(target_fake)        → pseudo-labels       [no_grad, hook drained]
 6. Filter pseudo-labels                   (clamp / size / class)
 7. Quality gate → có thể zero-out pseudo-labels
 8. loss_distillation (cap 1.5× loss_source)
 9. loss_consistency
10. loss_domain (boost theo prev_domain_acc, throttle nếu acc > 0.85)
11. total loss = sum, clamp [0, 500], NaN check
12. backward
13. unscale → clip grad → NaN-grad check → optimizer.step
14. scaler.step(grl_optimizer) nếu acc < 0.85
15. scaler.update; zero_grad
16. teacher_ema.update(student)
17. log_iteration; pbar.set_postfix; (mỗi 50 iter) DomainMonitor collect
```

Cuối epoch: `scheduler.step` → `DomainMonitor.end_epoch` → (mỗi 5 epoch) validate student + teacher → save `best/last/intermediate` → student-gate restore nếu collapse.

---

## 8. Những điểm dễ nhầm

1. **BN frozen ngay cả ở `source_real`** — không chỉ cho fake/target. Lý do: batch nhỏ + momentum=0.1 phá running stats.
2. **`batch['source_fake']` chứ không phải `batch_source`** trong forward 2 — nếu nhầm thì box-image mismatch (label thuộc draw augmentation khác).
3. **NMS `agnostic=True`** cho pseudo-label — chống duplicate sau `class_mapping`.
4. **`one2many`** chứ không phải `one2one` để decode — conf cao hơn.
5. **`xyxy → xywh`** trước khi gọi `non_max_suppression` (NMS gọi `xywh2xyxy` bên trong; nếu đưa xyxy vào sẽ bị decode kép → box hỏng).
6. **`teacher_ema.update` đặt SAU `optimizer.step`** — EMA của weights **đã** update.
7. **`hook.get_features()` phải gọi NGAY sau forward** — hook bị overwrite ở forward kế. Teacher hook vẫn cần drain dù feature không dùng.
8. **`optimizer.state = defaultdict(dict)`** khi reset, không phải `defaultdict()` rỗng — Adam sẽ KeyError.
9. **`validate_model` tạo `YOLO(weights_path)` mới** mỗi lần — cần `gc.collect` + `empty_cache` trong `finally` để đỡ leak.
10. **GRL boost + throttle**: weight tự nhân lên nếu `prev_domain_acc > 0.75`, đồng thời discriminator bị skip update khi `domain_acc > 0.85` — hai cơ chế đối xứng giữ adversarial cân bằng.
11. **Burn-in default = 5 (CLI)** nhưng code có fallback `getattr(opt, 'burn_in_epochs', 10)` — khi nạp từ YAML thiếu key, sẽ là 10.
12. **`TEACHER_CONF_THRES = 0.5`** hardcoded trong file — các cờ `--conf-thres` chỉ tương thích, không thực sự dùng cho pseudo-label.
