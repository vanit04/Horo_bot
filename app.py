import streamlit as st
import requests
import json
from datetime import datetime
import config

st.set_page_config(page_title="โหราจารย์ ๓ ภพ", page_icon="🔮", layout="centered")

st.title("🔮 โหราจารย์ ๓ ภพ (AI Horoscope)")
st.markdown("ผสานศาสตร์ไทย สากล และจีน เพื่อสร้างคำทำนายเฉพาะคุณ 🌕")

mode = st.radio("เลือกโหมดที่ต้องการ:", ["ทำนายดวงส่วนตัว", "ถามคำถาม"])

if mode == "ทำนายดวงส่วนตัว":
    timeframe = st.selectbox("ช่วงเวลาที่ต้องการทำนาย:", ["วันนี้", "สัปดาห์นี้", "เดือนนี้", "ปีนี้"])
    name = st.text_input("ชื่อ:")
    dob = st.text_input("วันเกิด (เช่น 14 กุมภาพันธ์ 2530):")
    tob = st.text_input("เวลาเกิด (หรือพิมพ์ว่า 'ไม่ทราบ')")

    if st.button("✨ สร้างคำทำนาย"):
        user_query = f"ชื่อ: {name}, วันเกิด: {dob}, เวลาเกิด: {tob}, ช่วงเวลา: {timeframe}"
        prompt = f'''
คุณคือ 'โหราจารย์ ๓ ภพ' นักพยากรณ์ผู้เชี่ยวชาญไทย-จีน-สากล
จงสร้างคำทำนายแบบละเอียดให้กับข้อมูลต่อไปนี้:

{user_query}

รูปแบบคำตอบ:
🌟 สิ่งดีที่จะเกิดขึ้น:
⚠️ ข้อควรระวัง:
🍀 โชคลาภและแนวทางเสริมดวง:
💡 ปัญหาและแนวทางแก้ไข:
'''
        with st.spinner("กำลังประมวลผลคำทำนาย..."):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/your-username/",
                    "X-Title": "Horacharn Sam Phop (Streamlit Edition)"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "คุณคือโหราจารย์ผู้ใช้ศาสตร์ 3 วัฒนธรรม"},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            result = response.json()
            st.write(result["choices"][0]["message"]["content"])

else:
    question = st.text_area("พิมพ์คำถามเกี่ยวกับดวงของคุณ:")
    if st.button("🔍 ถามโหราจารย์"):
        prompt = f"คุณคือโหราจารย์ ๓ ภพ จงตอบคำถามนี้อย่างละเอียดและให้กำลังใจ: {question}"
        with st.spinner("กำลังค้นหาคำตอบ..."):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/your-username/",
                    "X-Title": "Horacharn Sam Phop (Streamlit Edition)"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "คุณคือโหราจารย์ผู้ใช้ศาสตร์ 3 วัฒนธรรม"},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            result = response.json()
            st.write(result["choices"][0]["message"]["content"])
