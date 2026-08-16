import os
import re
import json
import requests
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Chèn CSS ẩn biểu tượng edit / toolbar của Streamlit
st.markdown(
    """
    <style>
    /* Ẩn Header chính của Streamlit */
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0%;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------------------------------------------------
# 1. CẤU HÌNH HỆ THỐNG & GEMINI API
# ---------------------------------------------------------
load_dotenv()
st.set_page_config(
    page_title="QuangAnh Investment Agent", 
    page_icon="📈", 
    layout="wide"
)

gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key) if gemini_key else None

REMINDERS_FILE = "reminders.json"
CHAT_SESSIONS_FILE = "chat_sessions.json"

# --- Hàm làm sạch văn bản, chống dính chữ ---
def clean_markdown_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\*{3,}', '**', text)
    text = re.sub(r'(?<=[a-zA-Zà-ỹÀ-Ỹ])([.,:;!?])(?=[a-zA-Zà-ỹÀ-Ỹ])', r'\1 ', text)
    text = re.sub(r'([a-zA-Z0-9à-ỹÀ-Ỹ])(\*\*)', r'\1 \2', text)
    text = re.sub(r'(\*\*)([a-zA-Z0-9à-ỹÀ-Ỹ])', r'\1 \2', text)
    return text

# --- Hàm quản lý dữ liệu JSON ---
def load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- Hàm lấy giá Crypto trực tiếp từ OKX (Giảm TTL xuống 5 giây) ---
@st.cache_data(ttl=5)
def get_crypto_prices():
    prices = {"BTC": "N/A", "ETH": "N/A", "PI": "N/A"}
    try:
        url_btc = "https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT"
        url_eth = "https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT"
        url_pi  = "https://www.okx.com/api/v5/market/ticker?instId=PI-USDT"
                
        res_btc = requests.get(url_btc, timeout=5).json()
        res_eth = requests.get(url_eth, timeout=5).json()
        res_pi  = requests.get(url_pi, timeout=5).json()
        
        if res_btc.get("code") == "0" and res_btc.get("data"):
            prices["BTC"] = f"${float(res_btc['data'][0]['last']):,.2f}"
            
        if res_eth.get("code") == "0" and res_eth.get("data"):
            prices["ETH"] = f"${float(res_eth['data'][0]['last']):,.2f}"
            
        if res_pi.get("code") == "0" and res_pi.get("data"):
            prices["PI"] = f"${float(res_pi['data'][0]['last']):,.5f}"    
    except Exception:
        pass
    return prices

# ---------------------------------------------------------
# 2. THANH SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("📌 Investment Agent")

# 1. Bảng giá thị trường (Tự động cập nhật)
st.sidebar.subheader("📊 Giá Thị Trường (OKX - Realtime)")
prices = get_crypto_prices()

# Nút cập nhật thủ công tức thì + hiển thị trạng thái
col_p1, col_p2 = st.sidebar.columns([3, 1])
with col_p1:
    st.caption("🔄 Tự động làm mới mỗi 10s")
with col_p2:
    if st.button("🔄", help="Bấm để cập nhật giá ngay lập tức"):
        st.cache_data.clear()
        st.rerun()

st.sidebar.write(f"• **Bitcoin (BTC):** {prices['BTC']}")
st.sidebar.write(f"• **Ethereum (ETH):** {prices['ETH']}")
st.sidebar.write(f"• **Pi Network (PI):** {prices['PI']}")

st.sidebar.markdown("---")

# 2. Lời nhắc việc
reminders = load_json(REMINDERS_FILE)
uncompleted_count = len([x for x in reminders if x.get("status") != "hoàn thành"])

with st.sidebar.expander(f"📝 Lời Nhắc Việc ({uncompleted_count} chưa xong)", expanded=False):
    new_task = st.text_input("➕ Thêm việc mới:", key="sb_new_task")
    if st.button("Lưu Lời Nhắc", use_container_width=True) and new_task.strip():
        reminders.append({"task": new_task.strip(), "status": "chưa xong"})
        save_json(REMINDERS_FILE, reminders)
        st.rerun()

    st.markdown("---")
    if reminders:
        updated_reminders = []
        for idx, item in enumerate(reminders):
            col_check, col_del = st.columns([4, 1])
            with col_check:
                checked = st.checkbox(
                    item["task"], 
                    value=(item.get("status") == "hoàn thành"), 
                    key=f"sb_task_{idx}"
                )
                item["status"] = "hoàn thành" if checked else "chưa xong"
            with col_del:
                if st.button("🗑️", key=f"sb_del_{idx}"):
                    continue
            updated_reminders.append(item)

        if len(updated_reminders) != len(reminders) or reminders != updated_reminders:
            save_json(REMINDERS_FILE, updated_reminders)
            st.rerun()

        if st.button("🗑️ Xóa Tất Cả Lời Nhắc", use_container_width=True):
            save_json(REMINDERS_FILE, [])
            st.rerun()
    else:
        st.info("Chưa có lời nhắc nào.")

st.sidebar.markdown("---")

# 3. Lịch sử Chat
st.sidebar.subheader("🕒 Lịch Sử Chat")
all_sessions = load_json(CHAT_SESSIONS_FILE)

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = all_sessions[0]["id"] if all_sessions else None

if st.sidebar.button("➕ Cuộc trò chuyện mới", use_container_width=True):
    st.session_state.current_session_id = None
    st.rerun()

for s in all_sessions[:100]:
    is_active = (s["id"] == st.session_state.current_session_id)
    label = f"{'💬 ' if is_active else '📄 '}{s['title']}"
    if st.sidebar.button(label, key=f"session_btn_{s['id']}", use_container_width=True):
        st.session_state.current_session_id = s["id"]
        st.rerun()

st.sidebar.caption("✅ Đã kết nối Gemini API" if client else "⚠️ Chưa nhận GEMINI_API_KEY")

# ---------------------------------------------------------
# 3. KHUNG CHÁT AI CHÍNH
# ---------------------------------------------------------
st.subheader("💬 Trợ Lý AI Phân Tích")

current_messages = []
active_session_index = None

if st.session_state.current_session_id is not None:
    for idx, s in enumerate(all_sessions):
        if s["id"] == st.session_state.current_session_id:
            current_messages = s["messages"]
            active_session_index = idx
            break

for msg in current_messages:
    with st.chat_message(msg["role"]):
        st.markdown(clean_markdown_text(msg["content"]))

prompt = st.chat_input("Hỏi AI về BTC, ETH (OKX), Pi Network, tài chính...")

if prompt:
    if active_session_index is None:
        import time
        new_id = int(time.time() * 1000)
        title = prompt[:30] + ("..." if len(prompt) > 30 else "")
        new_session = {"id": new_id, "title": title, "messages": []}
        all_sessions.insert(0, new_session)
        active_session_index = 0
        st.session_state.current_session_id = new_id

    all_sessions[active_session_index]["messages"].append({"role": "user", "content": prompt})
    save_json(CHAT_SESSIONS_FILE, all_sessions)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    if not client:
        st.error("Chưa cài đặt GEMINI_API_KEY trong file .env!")
    else:
        system_instruction_text = f"""
        Bạn là QuangAnh Investment Agent - cố vấn tài chính chuyên nghiệp.
        Dữ liệu giá thị trường cập nhật trực tiếp từ OKX:
        - Bitcoin (BTC): {prices['BTC']}
        - Ethereum (ETH): {prices['ETH']}
        - Pi Network (PI): {prices['PI']}

        Yêu cầu trình bày:
        - Trả lời bằng tiếng Việt, phân tích ngắn gọn, sắc bén và logic.
        - LUÔN để khoảng trắng rõ ràng sau dấu câu, trước và sau từ in đậm/in nghiêng (**từ**). Không để chữ bị dính liền vào nhau.
        - Bám sát vào dữ liệu giá thực tế khi phân tích.
        """

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_text = ""

            try:
                session_msgs = all_sessions[active_session_index]["messages"]
                formatted_history = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])]
                    ) for m in session_msgs[:-1]
                ]

                chat = client.chats.create(
                    
                    model="gemini-3.7-flash",
                   
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction_text,
                        temperature=0.7,
                    ),
                    history=formatted_history
                )

                response_stream = chat.send_message_stream(prompt)
                for chunk in response_stream:
                    if chunk.text:
                        full_text += chunk.text
                        clean_text = clean_markdown_text(full_text)
                        message_placeholder.markdown(clean_text + "▌")
                
                final_clean_text = clean_markdown_text(full_text)
                message_placeholder.markdown(final_clean_text)

                all_sessions[active_session_index]["messages"].append({"role": "assistant", "content": final_clean_text})
                save_json(CHAT_SESSIONS_FILE, all_sessions)

            except Exception as e:
                import traceback
                st.error(f"❌ Lỗi Gemini API: {str(e)}")
                st.code(traceback.format_exc())

# ---------------------------------------------------------
# 4. TỰ ĐỘNG LÀM MỚI GIÁ MỖI 10 GIÂY (NẾU KHÔNG ĐANG CHÁT)
# ---------------------------------------------------------
if not prompt:
    import time
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()
