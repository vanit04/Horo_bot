# โหราจารย์ ๓ ภพ — Streamlit + Google Gemini 2.5 (Streamlit Secrets)

## สรุป
แอป Streamlit ที่ใช้ Google Gemini (model: gemini-2.5-flash) เพื่อสร้างคำทำนายและตอบคำถามเกี่ยวกับดวง

## ตั้งค่า (Streamlit Cloud)
1. สร้าง GitHub repo แล้วอัปโหลดไฟล์ใน repository นี้
2. ใน Streamlit Cloud → Settings → Secrets เพิ่ม:

```toml
GEMINI_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY"
```

3. คลิก Rerun

## รันแบบ local (เพื่อทดสอบ)
- สร้างไฟล์ `secrets.toml` ใน `.streamlit/` หรือสร้าง `config.py` ชั่วคราว
- ติดตั้ง dependency: `pip install -r requirements.txt`
- รัน: `streamlit run app.py`

## หมายเหตุ
- API key ของ Google Gemini หาได้จาก Google Cloud / AI Studio
- แอปนี้มีการตรวจจับข้อผิดพลาดและแสดง JSON สำหรับ debug หากเกิดปัญหา
