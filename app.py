import streamlit as st
import requests
import json
from datetime import datetime
import config

st.set_page_config(page_title="โหราจารย์ ๓ ภพ", page_icon="🔮", layout="centered")

st.title("🔮 โหราจารย์ ๓ ภพ (AI Horoscope)")
st.markdown("ผสานศาสตร์ไทย สากล และจีน เพื่อสร้างคำทำนายเฉพาะคุณ 🌕")

def call_openrouter(prompt: str):
    try:
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
            },
            timeout=60
        )

        # ตรวจสอบรหัสสถานะ HTTP
        if response.status_code != 200:
            st.error(f"❌ API ตอบกลับด้วยรหัสสถานะ {response.status_code}")
            st.text(response.text)
            return "⚠️ ระบบไม่สามารถเชื่อมต่อ AI ได้ในขณะนี้ กรุณาลองใหม่ภายหลังครับ"

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            msg = result["error"].get("message", "ไม่ทราบสาเหตุ")
            st.error(f"⚠️ ข้อผิดพลาดจาก API: {msg}")
            return "ขออภัยครับ ระบบ AI มีปัญหาในการประมวลผล กรุณาลองใหม่อีกครั้งครับ 🙏"
        else:
            st.warning("⚠️ รูปแบบข้อมูลตอบกลับไม่เป็นไปตามที่คาดไว้")
            st.json(result)
            return "ระบบไม่พบข้อมูลคำตอบจาก AI กรุณาลองใหม่อีกครั้งครับ 🙏"

    except requests.exceptions.RequestException as e:
        st.error(f"🚨 ปัญหาการเชื่อมต่อ: {e}")
        return "ระบบขัดข้องชั่วคราว กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ตหรือลองใหม่ภายหลังครับ"
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดไม่คาดคิด: {e}")
        return "เกิดข้อผิดพลาดภายในระบบ กรุณาลองใหม่ครับ"

# ส่วน UI หลัก
mode = st.radio("เลือกโหมดที่ต้องการ:", ["ทำนายดวงส่วนตัว", "ถามคำถาม"])

if mode == "ทำนายดวงส่วนตัว":
    timeframe = st.selectbox("ช่วงเวลาที่ต้องการทำนาย:", ["วันนี้", "สัปดาห์นี้", "เดือนนี้", "ปีนี้"])
    name = st.text_input("ชื่อ:")
    dob = st.text_input("วันเกิด (เช่น 14 กุมภาพันธ์ 2530):")
    tob = st.text_input("เวลาเกิด (หรือพิมพ์ว่า 'ไม่ทราบ')")

    if st.button("✨ สร้างคำทำนาย"):
        if not name or not dob:
            st.warning("กรุณากรอกข้อมูลชื่อและวันเกิดให้ครบก่อนครับ 🙏")
        else:
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
            with st.spinner("🔮 กำลังสังเคราะห์คำทำนายจากศาสตร์ทั้ง 3 โปรดรอสักครู่..."):
                answer = call_openrouter(prompt)
                st.success("✨ คำทำนายของคุณ:")
                st.markdown(answer)

else:
    question = st.text_area("พิมพ์คำถามเกี่ยวกับดวงของคุณ:")
    if st.button("🔍 ถามโหราจารย์"):
        if not question.strip():
            st.warning("กรุณาพิมพ์คำถามก่อนครับ 🙏")
        else:
            prompt = f"คุณคือโหราจารย์ ๓ ภพ จงตอบคำถามนี้อย่างละเอียดและให้กำลังใจ: {question}"
            with st.spinner("🔮 กำลังค้นหาคำตอบจากศาสตร์ทั้ง 3..."):
                answer = call_openrouter(prompt)
                st.success("💫 คำตอบจากโหราจารย์ ๓ ภพ:")
                st.markdown(answer)
