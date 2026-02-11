# GitHub Actions Secrets Setup (Urdu Guide)

یہ مختصر گائیڈ آپ کو `autopost-ci` workflow چلانے کے لیے required secrets set کرنے میں مدد دیتی ہے۔

## 1) Secrets page کھولیں
1. اپنی repository کھولیں۔
2. اوپر سے **Settings** پر جائیں۔
3. بائیں مینو میں: **Secrets and variables → Actions**۔
4. **New repository secret** پر کلک کریں۔

## 2) Required secrets add کریں
یہ 4 secrets لازمی ہیں:

- `WP_URL` (مثال: `https://example.com`)
- `WP_USER` (WordPress username)
- `WP_APP_PASSWORD` (WordPress Application Password)
- `OPENAI_API_KEY` (`sk-...`)

ہر secret کے لیے:
- **Name** میں exact نام لکھیں (uppercase/underscore کے ساتھ)
- **Secret** میں value paste کریں
- **Add secret** پر کلک کریں

## 3) Optional secrets (اگر چاہیں)
- `OPENAI_MODEL` (مثال: `gpt-4o-mini`)
- `PEXELS_API_KEY`

## 4) Workflow manually run کریں
1. **Actions** tab کھولیں۔
2. `autopost-ci` workflow منتخب کریں۔
3. **Run workflow** بٹن دبائیں۔

## 5) Expected behavior
- اگر required secrets موجود ہوں تو `python autopost.py` چلے گا۔
- اگر secrets missing ہوں تو workflow fail نہیں ہوگا؛ optional run step skip/message دے گا۔

## 6) WordPress Application Password بنانے کا طریقہ
1. WordPress admin میں login کریں۔
2. **Users → Profile** پر جائیں۔
3. نیچے **Application Passwords** سیکشن میں جائیں۔
4. نام دیں (مثلاً `github-actions`) اور **Add New Application Password** دبائیں۔
5. generated password copy کرکے `WP_APP_PASSWORD` secret میں save کریں۔

> نوٹ: Secret names میں اسپیس/چھوٹے حروف نہ رکھیں، exact key names ہی استعمال کریں۔


## 7) `autoscript` naam se secrets (aapki request ke mutabiq)
Agar aap WP_* ki jagah `AUTOSCRIPT_*` naming use karna chahte hain, ab workflow support karta hai:

- `AUTOSCRIPT_WP_URL`
- `AUTOSCRIPT_WP_USER`
- `AUTOSCRIPT_WP_APP_PASSWORD`
- `AUTOSCRIPT_OPENAI_API_KEY`

Optional:
- `AUTOSCRIPT_OPENAI_MODEL`
- `AUTOSCRIPT_PEXELS_API_KEY`

> Workflow pehle `AUTOSCRIPT_*` values use karega; agar missing ہوں تو `WP_*` fallback use hoga.
