# Thêm một phép tấn công vào AdverTest

> Mục tiêu của tài liệu này: mỗi người trong nhóm tự thêm **một attack** mà
> không bao giờ phải sửa file của người khác — nên gần như không có merge conflict.

---

## 1. Nguyên tắc: một attack = một file

Không có "danh sách attack" ở đâu cả. Plugin tự đăng ký bằng decorator trong
chính file của nó, còn `src/core/registry.py` sẽ import mọi module trong
`src/attacks/**` khi khởi động (`discover()`). Vì vậy:

| Bạn được sửa | Bạn **không** sửa |
|---|---|
| `src/attacks/<nhóm>/<attack_của_bạn>.py` | `src/attacks/base.py`, `src/attacks/__init__.py` |
| `tests/test_attacks/test_<attack_của_bạn>.py` | `src/core/*`, `src/pipeline/*`, `src/evaluation/*` |
| Thêm `docs/` riêng nếu cần giải thích công thức | `tests/test_attacks/test_contract.py` |

Chỉ còn **một** điểm có thể trùng: hai người chọn cùng `name`. Trường hợp đó
registry báo lỗi `RegistryConflictError` ngay khi import (CI chặn), không im lặng
ghi đè. Trước khi bắt đầu, chạy `make catalog` để xem tên nào đã có.

Nếu bạn thấy mình *cần* sửa file core để attack chạy được → hợp đồng plugin đang
thiếu thứ gì đó. Mở issue/PR riêng cho phần core, đừng nhét vào PR attack.

---

## 2. Nhóm nào, thư mục nào

| Nhóm (plan §2) | Thư mục | Ví dụ đã có |
|---|---|---|
| A — Common corruptions | `src/attacks/corruption/` | `gaussian_noise.py` |
| B — Thời tiết vật lý (depth-aware) | `src/attacks/weather/` | *(trống)* |
| C — Che khuất & lỗi cảm biến | `src/attacks/occlusion/` | *(trống)* |
| D — Adversarial white-box | `src/attacks/adversarial/` | `fgsm.py` |
| E — Adversarial patch | `src/attacks/patch/` | *(trống)* |
| F — Black-box & transfer | `src/attacks/blackbox/` | *(trống)* |

`group` khai báo trong class (`"A"`…`"F"`) là thứ hệ thống dùng để nhóm báo cáo;
thư mục chỉ để người đọc dễ tìm. Giữ hai thứ khớp nhau.

---

## 3. Năm bước

```bash
# 1) Nhánh riêng cho attack của bạn
git checkout -b feat/attack-motion-blur

# 2) Copy template (file bắt đầu bằng "_" không bị auto-discovery nạp)
cp src/attacks/_template.py src/attacks/corruption/motion_blur.py
```

3. Sửa phần khai báo — đây là "tờ khai" của plugin (plan §2):

```python
class MotionBlurParams(AttackParams):
    kernel_per_severity: tuple[int, ...] = (3, 5, 9, 13, 17)


@ATTACKS.register
class MotionBlur(BaseAttack):
    """Motion blur do rung/di chuyển camera."""      # dòng này lên UI

    name: ClassVar[str] = "motion_blur"              # snake_case, duy nhất
    group: ClassVar[AttackGroup] = "A"
    cost_class: ClassVar[CostClass] = "cheap"        # cheap | medium | expensive
    owner: ClassVar[str] = "ten-cua-ban"             # thay cho bảng phân công
    reference: ClassVar[str] = "Hendrycks & Dietterich, ICLR 2019 (arXiv:1903.12261)"
    params_model: ClassVar[type[AttackParams]] = MotionBlurParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        kernel = int(self.level(severity, self.params.kernel_per_severity))
        return sample.with_image(box_blur(sample.image, kernel // 2))
```

4. Kiểm tra — test hợp đồng đã tự bao phủ file mới của bạn:

```bash
uv run pytest tests/test_attacks -q             # gồm cả contract test
uv run python -m src.cli attacks                # attack đã xuất hiện chưa
uv run python -m src.cli run --attacks motion_blur --severities 1,3,5 --limit 4
```

5. Thêm test riêng cho phần chỉ bạn biết (công thức, biên, tham số) vào
   `tests/test_attacks/test_motion_blur.py` — xem `test_gaussian_noise.py` làm mẫu.

---

## 4. Hợp đồng bắt buộc

`BaseAttack.run()` (không override) tự lo phần chung; `apply()` của bạn phải tuân:

| Điều | Vì sao | Ai kiểm |
|---|---|---|
| `severity = 0` là no-op | sanity check #1 của plan §3: `ε = 0` ⇒ `AP = AP_clean` | contract test |
| Severity càng cao, nhiễu càng mạnh | sanity check #2: `AP(c,s)` phải giảm đơn điệu | contract test |
| Không sửa `sample`, trả về `sample.with_image(...)` | mọi cell chạy song song trên cùng dataset | contract test |
| Không đổi `sample.boxes` | ground truth là bất biến, nếu không mọi chỉ số vô nghĩa | contract test |
| Chỉ dùng `ctx.rng`, không dùng `np.random` | run phải tái lập được từ `seed` | contract test |
| Ảnh giữ shape, float32, giá trị `[0,1]` | `run()` tự clip + validate, nhưng đừng trả về NaN | `validate_image` |
| Mọi con số tuỳ chỉnh nằm trong `params_model` | UI đọc `params_schema`, cache key băm theo params | contract test |

Severity mặc định là 1..5 (`severity_levels`). Muốn 3 mức thì đặt
`severity_levels = 3`, hệ thống tự hiểu.

---

## 5. Attack cần model (nhóm D, E, F)

Khai báo thêm hai cờ:

```python
needs_model: ClassVar[bool] = True        # apply() dùng ctx.model
needs_gradients: ClassVar[bool] = True    # cần input_gradient (chỉ nhóm D, E)
```

Trong `apply()`:

```python
model = ctx.require_model(self.name)      # lỗi rõ ràng nếu thiếu
gradient = model.input_gradient(sample)   # numpy, cùng shape với ảnh
prediction = model.predict([sample])[0]   # black-box: chỉ cần đầu ra
```

Hai điều quan trọng:

* **Không import torch trong file attack.** Adapter giữ framework, attack chỉ
  thấy numpy. Nhờ vậy catalog vẫn import được trong CI không GPU, và cùng một
  attack chạy được cho YOLO (torch), MMDet, hay ONNX.
* `loss_for_attack` được định nghĩa để **càng lớn = model càng sai**, nên attack
  luôn *đi lên* theo gradient (ascent). Untargeted dùng dấu `+`, targeted đổi
  hàm loss chứ không đổi dấu bước đi.

Nếu adapter báo `supports_gradients = False`, runner **bỏ qua** attack của bạn và
ghi lý do vào `report.skipped` — không có cell nào biến mất im lặng.

### Tham số mạnh yếu khác nhau giữa model tham chiếu và model thật

`blob_detector` là detector ngưỡng, biên quyết định rộng hơn CNN thật nhiều bậc.
Hệ quả: `ε = 8/255` (đúng theo plan §2, đủ để đánh gục YOLO) gần như không làm
giảm AP của nó. Đó **không** phải lỗi của attack.

Quy tắc: giữ mặc định đúng theo paper/plan, còn khi thử trên model tham chiếu thì
truyền tham số mạnh hơn từ ngoài, không sửa mặc định:

```bash
uv run python -m src.cli run --attacks fgsm \
  --params '{"fgsm": {"epsilon_per_severity": [0.02, 0.04, 0.08, 0.16, 0.32]}}'
```

Ngược lại, corruption (nhóm A/B/C) tác động lên toàn ảnh nên tham số theo paper
thường đã đủ mạnh với cả hai loại model.

### Adapter có gradient

Muốn nhóm D chạy trên model thật thì adapter phải hiện thực `loss_for_attack` và
`input_gradient` (xem `src/adapters/_template.py`). Đó là việc của người sở hữu
adapter, không phải người viết attack.

---

## 6. Chọn `cost_class` cho đúng

`cost_class` là thứ hệ thống dùng để ước tính GPU **trước khi** chạy (plan §5) và
để xếp thứ tự quét hai tầng. Ước lượng theo số forward/backward pass mỗi ảnh:

| `cost_class` | Ý nghĩa | Ví dụ |
|---|---|---|
| `cheap` | không gọi model, thuần xử lý ảnh | corruption, thời tiết, che khuất |
| `medium` | vài lần forward/backward mỗi ảnh | FGSM, PGD ít bước |
| `expensive` | hàng trăm vòng lặp mỗi ảnh | C&W, patch + EOT, Square Attack |

Khai báo sai làm ước tính chi phí sai → người khác bấm Run và bị vượt ngân sách.

---

## 7. Checklist trước khi mở PR

- [ ] `name` chưa ai dùng (`make catalog`), snake_case, đúng tên trong paper
- [ ] `owner` là bạn, `reference` trỏ tới paper/thư viện thật
- [ ] `uv run pytest tests/test_attacks -q` xanh (gồm contract test)
- [ ] `uv run ruff check src/ tests/` xanh
- [ ] Có test riêng cho công thức/biên của attack
- [ ] `uv run python -m src.cli run --attacks <ten>` cho thấy `D%` tăng theo severity
- [ ] PR chỉ chạm file của bạn (`git diff --name-only main` để kiểm tra)

---

## 8. Slot còn trống

Cứ nhận một dòng, ghi tên vào `owner` trong file của bạn (không cần bảng phân
công tập trung — `make catalog` in ra ai đang giữ gì).

| Nhóm | Slot | Ghi chú |
|---|---|---|
| A | 18 corruption còn lại của ImageNet-C | giữ đúng tên ImageNet-C để so sánh mPC/rPC với paper |
| B | rain, snow, fog LiDAR, snow LiDAR | dùng `sample.depth`; công thức ở plan §2 nhóm B |
| C | random erasing, occlusion theo box GT, camera dropout, LiDAR beam/sector drop, frame freeze | attack đa cảm biến đặt `modality = "multi"` |
| D | Không còn slot trong phạm vi generator hiện tại | FGSM, PGD, MI-FGSM, C&W, TOG, DAG và SAM2-PGD đã có plugin |
| E | Không còn slot trong phạm vi generator hiện tại | DPatch và Thys patch đã có train-artifact-apply, EOT, TV/NPS |
| F | Square Attack, transfer matrix, random-noise baseline cùng `ε` | baseline này là điều kiện tin cậy của nhóm D (plan §11) |

Ngoài attack, các slot khác cùng cơ chế plugin: adapter model
(`src/adapters/`), dataset (`src/datasets/`), và chỉ số đánh giá
(`src/evaluation/`, danh sách trong docstring của package).
