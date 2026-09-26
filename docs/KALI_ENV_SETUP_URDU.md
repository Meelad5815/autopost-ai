# Kali Linux Environment (Autopost Project)

یہ فائل Kali Linux کلاؤڈ انوائرمنٹ کو جلدی سیٹ اپ کرنے کے لیے ہے۔

## کیا شامل کیا گیا ہے

- `docker-compose.kali.yml` (Kali container service)
- `scripts_kali_setup.sh` (one-command bootstrap script)
- UI میں `Kali Env Status` بٹن (admin کے لیے)
- API endpoint: `GET /ui/kali-env/status`

## استعمال

### 1) Kali container اسٹارٹ کریں

```bash
docker compose -f docker-compose.kali.yml up -d kali
```

### 2) Tools install کریں

```bash
./scripts_kali_setup.sh
```

### 3) Container shell میں جائیں

```bash
docker exec -it autopost-kali bash
```

## UI سے status چیک

Dashboard میں **Kali Env Status** دبائیں:

- `docker=yes/no`
- `compose=yes/no`
- `Kali: running/stopped`

## نوٹ

- یہ setup Docker-based Kali فراہم کرتا ہے، host OS replace نہیں کرتا۔
- اگر Docker daemon بند ہو تو status "stopped" یا "unavailable" آ سکتا ہے۔
