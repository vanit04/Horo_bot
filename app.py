import streamlit as st
import httpx
import json
from datetime import datetime
import time

st.set_page_config(page_title="โหราจารย์ ๓ ภพ (Gemini)", page_icon="🔮", layout="centered")
st.title("🔮 โหราจารย์ ๓ ภพ (Streamlit + Gemini 2.5)")
st.markdown("ผสานศาสตร์ไทย สากล และจีน — สร้างคำทำนายด้วย Google Gemini 2.5 Flash")

# Load API key from Streamlit secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error(
        "❌ ไม่พบค่า `GEMINI_API_KEY` ใน Streamlit Secrets.\n"
        "ให้ไปที่ App → Settings → Secrets แล้วเพิ่ม:\n\n"
        "GEMINI_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY"\n\n"
        "แล้วกด Rerun"
    )
    st.stop()

# Config: model
MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"

# Utility: call Gemini with retries and robust parsing
async def _call_gemini_async(payload, max_retries=3):
    timeout = httpx.Timeout(60.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        delay = 2
        for attempt in range(max_retries):
            try:
                r = await client.post(GEMINI_URL, json=payload)
                # Try to parse JSON even on non-200 to show details
                try:
                    parsed = r.json()
                except Exception:
                    parsed = {"raw_text": r.text, "status_code": r.status_code}

                if r.status_code == 200:
                    # expected shape: { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
                    return {"ok": True, "resp": parsed}
                else:
                    # expose API error message
                    return {"ok": False, "resp": parsed, "status": r.status_code}

            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return {"ok": False, "error": str(e)}
            except Exception as e:
                return {"ok": False, "error": str(e)}

def call_gemini(prompt, system_prompt):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        # enable Google search tool only if available in your Gemini settings
        "tools": [{"google_search": {}}]
    }
    # Call async from sync Streamlit
    import asyncio
    result = asyncio.run(_call_gemini_async(payload))

    if not result.get("ok"):
        # show diagnostic info in UI
        info = result.get("resp") or {"error": result.get("error")}
        return False, info

    resp = result["resp"]
    # Parse the typical response
    try:
        if "candidates" in resp and resp["candidates"]:
            part = resp["candidates"][0]["content"]["parts"][0].get("text")
            return True, part
        else:
            return False, resp
    except Exception as e:
        return False, {"error": f"parsing_error: {e}", "raw": resp}

# System prompts (Thai)
PREDICTION_SYSTEM_PROMPT = """คุณคือ "โหราจารย์ ๓ ภพ" (Hōracharn Sam Phop) — นักพยากรณ์ผู้เชี่ยวชาญการสังเคราะห์ศาสตร์ 3 วัฒนธรรม (ไทย, สากล, จีน).
ให้ใช้ภาษาที่เมตตา เข้าใจง่าย และให้กำลังใจ"""

QUESTION_SYSTEM_PROMPT = """คุณคือ "โหราจารย์ ๓ ภพ" — ตอบคำถามเกี่ยวกับโหราศาสตร์จากหลักฐาน/แหล่งข้อมูลที่เชื่อถือได้
ให้คำตอบกระชับ ชัดเจน และมีแนวทางปฏิบัติ"""

# UI: choose mode
mode = st.radio("โหมด:", ["🔮 ทำนายดวงส่วนตัว", "❓ ถามคำถามเกี่ยวกับดวง"])

if mode.startswith("🔮"):
    st.subheader("ข้อมูลสำหรับทำนาย")
    timeframe = st.selectbox("ช่วงเวลาที่ต้องการทำนาย:", ["วันนี้", "สัปดาห์นี้", "เดือนนี้", "ปีนี้"])
    name = st.text_input("ชื่อ:")
    dob = st.text_input("วัน/เดือน/ปีเกิด (เช่น 14 กุมภาพันธ์ 2530):")
    tob = st.text_input("เวลาเกิด (หรือพิมพ์ว่า 'ไม่ทราบ'):", value="ไม่ทราบ")

    if st.button("✨ สร้างคำทำนาย"):
        if not name.strip() or not dob.strip():
            st.warning("กรุณากรอกชื่อและวันเกิดให้ครบก่อนครับ")
        else:
            user_query = f"ชื่อ: {name}\nวันเกิด: {dob}\nเวลาเกิด: {tob}\nช่วงเวลา: {timeframe}"
            prompt = f"""สร้างคำทำนายโดยสังเคราะห์จากศาสตร์ไทย สากล และจีน ให้แบ่งคำตอบเป็น 4 ส่วน:\n\n🌟 สิ่งดีที่จะเกิดขึ้น:\n⚠️ ข้อควรระวัง:\n🍀 โชคลาภและแนวทางเสริมดวง:\n💡 ปัญหาและแนวทางแก้ไข:\n\nข้อมูลผู้ใช้:\n{user_query}"""
            with st.spinner("กำลังติดต่อ Gemini... โปรดรอสักครู่"):
                ok, resp = call_gemini(prompt, PREDICTION_SYSTEM_PROMPT)
                if ok:
                    st.success("✨ คำทำนาย:")
                    st.markdown(resp)
                else:
                    st.error("ไม่สามารถรับคำตอบจาก Gemini ได้")
                    st.write("รายละเอียดสำหรับ debug:")
                    st.json(resp)

else:
    st.subheader("ถามคำถามเกี่ยวกับดวง")
    question = st.text_area("พิมพ์คำถามของคุณ:")
    if st.button("🔍 ถามโหราจารย์"):
        if not question.strip():
            st.warning("กรุณาพิมพ์คำถามก่อน")
        else:
            prompt = f"{question}\n\nจงตอบโดยอิงหลักโหราศาสตร์ และให้คำแนะนำที่เป็นประโยชน์"
            with st.spinner("กำลังติดต่อ Gemini... โปรดรอสักครู่"):
                ok, resp = call_gemini(prompt, QUESTION_SYSTEM_PROMPT)
                if ok:
                    st.success("💫 คำตอบ:")
                    st.markdown(resp)
                else:
                    st.error("ไม่สามารถรับคำตอบจาก Gemini ได้")
                    st.write("รายละเอียดสำหรับ debug:")
                    st.json(resp)
