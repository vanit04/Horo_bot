# 🔮 โหราจารย์ ๓ ภพ (Horacharn Sam Phop) — Streamlit + OpenRouter

## 🧠 Concept
โหราจารย์ดิจิทัลที่ผสานศาสตร์ไทย จีน และสากล พร้อมพยากรณ์ผ่าน AI (OpenRouter)

## 🚀 การติดตั้งและใช้งาน
1. Clone หรือดาวน์โหลด repo
2. สร้างไฟล์ `config.py` และใส่ API key:
   ```python
   OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
   ```
3. ติดตั้ง dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. รันแอป:
   ```bash
   streamlit run app.py
   ```

## ☁️ Deploy บน Streamlit Cloud
1. อัปโหลด repo ไปยัง GitHub
2. เข้า [https://share.streamlit.io](https://share.streamlit.io)
3. เพิ่ม Secret:
   ```toml
   OPENROUTER_API_KEY="sk-xxxxxxx"
   ```
4. กด Deploy แล้วใช้งานได้ทันที

## 🧭 หมายเหตุ
- โมเดล: gpt-4o-mini (ผ่าน OpenRouter)
- แนะนำให้ใช้ Referer เป็นลิงก์ GitHub ของคุณเอง
