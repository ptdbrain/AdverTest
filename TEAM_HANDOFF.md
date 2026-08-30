# TEAM HANDOFF — feat/3d-auth-settings-naming

> Handoff cho teammates trước khi review/merge. Nhánh đã push lên origin.

**Nhánh:** `feat/3d-auth-settings-naming`
**Mục tiêu:** đa ngôn ngữ (vi/en) toàn UI + làm gọn trang Settings + component so sánh phòng thủ.
**Cổng chất lượng đã chạy:** `npm run build` ✅ pass (TypeScript + đủ 5 route prerender). Lint/test frontend: `npm run lint` + `npm run test -- --run`.

---

## 1. Chạy và dùng

```bash
cd frontend
npm install     # nếu chưa
npm run dev     # http://localhost:3000
```

Build kiểm chứng: `npm run build`. Cổng QA frontend giống `.github/workflows/ci.yml` (lint + test + build).

## 2. i18n — cách teammate thêm/hiển thị text mới

Trước hết **KHÔNG viết chuỗi cứng**; dùng hook trong component client:

```js
import { useLanguage } from "@/context/LanguageContext";
const { t } = useLanguage();
<button>{t("common.save")}</button>
```

- **Thêm key mới:** mở `src/lib/i18n.js`, thêm vào **cả** `vi:` (≈ dòng 13) lẫn `en:` (≈ dòng 330), dùng **cùng một key**. Prefix theo module: `common.*`, `settings.*`, `auth.*`, …
- **Ngoài component / trong hook:** `translate(lang, key, params)` là hàm **thuần (không phụ thuộc React)** — xem ví dụ trong `src/hooks/useAdverTest.js`.
- **Ngôn ngữ lưu ở `localStorage.advertest_lang`** (đúng pattern theme/auth); toggle ở `Settings → Giao Diện & Ngôn Ngữ`.
- Provider `LanguageProvider` đã bọc trong `AppProviders.jsx` và setup trong `app/layout.js`.

## 3. File mới trong lượng đổi này

| File | Vai trò |
|---|---|
| `frontend/src/lib/i18n.js` | Từ điển phẳng vi/en + `translate()` (không framework) |
| `frontend/src/context/LanguageContext.jsx` | Provider + hook `useLanguage()` |
| `frontend/src/components/DefenceVisualComparison.jsx` | So sánh trực quan phòng thủ |

## 4. ⚠️ Điểm cần teammate lưu ý khi merge / coi là "deploy-ready"

### CI chỉ chạy với `main`
`.github/workflows/ci.yml` chỉ trigger trên push `main`/`develop` và **PR vào `main`**.
→ **PR vào `feature/platform-deploy` sẽ KHÔNG chạy CI.** Muốn gate: merge qua `main` trước, hoặc tự chạy lint/test/build.

### Xung đột merge dự kiến (đã xác minh bằng `git merge-tree`)
Khi đưa vào `feature/platform-deploy`, 4 file xung đột nội dung:
- `frontend/src/hooks/useAdverTest.js`
- `src/api/dependencies.py`
- `src/visualization/bev3d.py`
- `tests/conftest.py`

(ngoài ra `pyproject.toml`, `src/api/helpers.py` auto-merge nhưng vẫn nên xem lại — 2 nhánh chạm nhau ở 25 file.)

### Bảo mật — chưa sửa, cần XÁC NHẬN trước khi gọi là deploy-ready
1. **JWT secret hardcode** `src/auth/security.py:14` — không có env override → ai đọc mã nguồn cũng ký được token `ADMIN`. Cần nạp từ env (settings) ở sản xuất.
2. **Google SSO fallback** `src/auth/service.py:103-114` — khi `oauth2.googleapis.com/tokeninfo` fail, giải mã JWT client-cấp mà **không kiểm tra chữ ký/issuer/audience** → nguy cơ login giả. Cần bỏ fallback không xác thực chữ ký.
3. **Default admin password** `AdminPassword123!` trong `ensure_default_accounts` — nếu demo thì note rõ; deploy thật phải đổi/khóa.
4. **W&B "mã hóa an toàn"** — UI hiện ghi "API key được lưu mã hóa an toàn" (`frontend/src/lib/i18n.js:92`). Hãy **xác minh back-end thật sự mã hóa key**; đừng hứa tính năng chưa có.

## 5. Xem toàn bộ lượng đổi

```bash
git log --oneline origin/feat/3d-auth-settings-naming --not origin/main
git diff --stat origin/main...origin/feat/3d-auth-settings-naming
```