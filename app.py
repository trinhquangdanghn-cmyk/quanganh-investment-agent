import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import streamlit as st
import yfinance as yf

# -----------------------------------------------------------------------------
# 0. Hàm lấy dữ liệu Vĩ mô (DXY, US10Y)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache 1 tiếng để tránh bị YFinance block IP
def get_macro_data():
    try:
        # Ticker chính & dự phòng cho DXY
        dxy_tickers = ["DX-Y.NYB", "DX=F"]
        dxy_data = None
        for ticker in dxy_tickers:
            df = yf.Ticker(ticker).history(period="7d")
            if not df.empty:
                dxy_data = df
                break
        
        # Ticker cho US10Y
        us10y_data = yf.Ticker("^TNX").history(period="7d")

        # Xử lý DXY
        if dxy_data is not None and not dxy_data.empty:
            dxy_latest = round(dxy_data['Close'].iloc[-1], 2)
            if len(dxy_data) >= 2:
                dxy_change = round(((dxy_data['Close'].iloc[-1] - dxy_data['Close'].iloc[-2]) / dxy_data['Close'].iloc[-2]) * 100, 2)
                dxy_status = "TĂNG 📈" if dxy_change > 0 else "GIẢM 📉"
            else:
                dxy_change, dxy_status = 0.0, "N/A"
        else:
            dxy_latest, dxy_change, dxy_status = "103.50", 0.0, "N/A" # Giá trị mặc định nếu YFinance lỗi

        # Xử lý US10Y
        if not us10y_data.empty:
            us10y_latest = f"{round(us10y_data['Close'].iloc[-1], 2)}%"
        else:
            us10y_latest = "3.90%" # Giá trị mặc định

        return {
            "DXY": dxy_latest,
            "DXY_Change_%": dxy_change,
            "DXY_Status": dxy_status,
            "US10Y": us10y_latest
        }
    except Exception as e:
        return {"DXY": "103.50", "US10Y": "3.90%", "DXY_Change_%": 0, "DXY_Status": "N/A", "Error": str(e)}
# -----------------------------------------------------------------------------
# 1. Cấu hình giao diện Streamlit (CHỈ GỌI 1 LẦN DUY NHẤT)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QuangAnh Crypto Investment Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Agent - Phân Tích & Theo Dõi Thị Trường Crypto")
st.caption(
    "Hệ thống phân tích định lượng, dòng tiền cá voi & AI Cố vấn Open Mainnet"
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
CHAT_SESSIONS_FILE = "chat_sessions.json"

# -----------------------------------------------------------------------------
# 2. System Prompt nghiêm ngặt
# -----------------------------------------------------------------------------
STRICT_SYSTEM_PROMPT = """
Bạn là QuangAnh Investment Agent - Cố vấn tài chính định lượng cao cấp.

[QUY TẮC BẮT BUỘC 100% VỀ PI NETWORK (NẾU ĐỒNG COIN ĐANG CHỌN LÀ PI)]:
1. Pi Network (PI) ĐÃ CHÍNH THỨC RA MẮT OPEN MAINNET THỰC TẾ và giao dịch trực tiếp trên OKX (PI/USDT).
2. TUYỆT ĐỐI KHÔNG sử dụng các từ: "IOU", "Futures", "Hợp đồng tương lai", "Chưa niêm yết".
3. Mọi phân tích kỹ thuật và dòng tiền của PI dựa hoàn toàn trên dữ liệu Mainnet thực tế từ OKX.

[QUY TẮC PHÂN TÍCH GIÁ THỰC TẾ & SO SÁNH TƯƠNG QUAN]:
1. COIN TRỌNG TÂM: Mọi phân tích kỹ thuật, chỉ báo (RSI, ATR) và mốc giao dịch (Entry, SL, TP) PHẢI lấy coin đang chọn trong Context làm trọng tâm chính.
2. SO SÁNH LÍNH HOẠT: Khi người dùng yêu cầu so sánh hoặc đối chiếu (ví dụ: so sánh với BTC, ETH hay DXY), bạn ĐƯỢC PHÉP sử dụng dữ liệu vĩ mô và chỉ số các coin khác có trong Context để phân tích tương quan dòng tiền.
3. MINH BẠCH DỮ LIỆU: TUYỆT ĐỐI KHÔNG tự bịa giá. Nếu giá hoặc chỉ báo của coin trả về là "N/A", phải thông báo rõ ràng dữ liệu real-time chưa sẵn sàng.

"""

macro_info = get_macro_data()

# --- HIỂN THỊ CHỈ SỐ VĨ MÔ TRÊN SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Chỉ Số Vĩ Mô & Lãi Suất")
col_m1, col_m2 = st.sidebar.columns(2)
col_m1.metric("DXY Index", f"{macro_info.get('DXY')}", f"{macro_info.get('DXY_Change_%')}%")
col_m2.metric("Trái Phiếu US10Y", f"{macro_info.get('US10Y')}")
st.sidebar.caption("🏛️ **Lãi suất FED:** `5.25% - 5.50%`")

# -----------------------------------------------------------------------------
# 3. Quản lý Lịch sử Chat (JSON)
# -----------------------------------------------------------------------------
def load_chat_sessions():
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_sessions(sessions):
    try:
        with open(CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Lỗi lưu file lịch sử: {e}")

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()

if not st.session_state.chat_sessions:
    st.session_state.chat_sessions = [
        {"title": "Cuộc trò chuyện mới", "messages": []}
    ]
    save_chat_sessions(st.session_state.chat_sessions)

if "active_index" not in st.session_state:
    st.session_state.active_index = 0

# -----------------------------------------------------------------------------
# 4. API Fetching linh hoạt cho BẤT KỲ đồng coin nào
# -----------------------------------------------------------------------------
@st.cache_data(ttl=10)
def get_single_coin_price(coin_symbol):
    clean_coin = coin_symbol.upper().replace("-USDT", "").replace("USDT", "").strip()
    inst_id = f"{clean_coin}-USDT"
    
    # 1. Fetch OKX Spot
    url_okx = f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}"
    try:
        res = requests.get(url_okx, timeout=3).json()
        if res.get("code") == "0" and res.get("data"):
            return float(res["data"][0]["last"])
    except Exception:
        pass

    # 2. Fallback Binance Spot
    try:
        res_b = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={clean_coin}USDT", timeout=3).json()
        if "price" in res_b:
            return float(res_b["price"])
    except Exception:
        pass

    return "N/A"

@st.cache_data(ttl=300)
def get_fear_and_greed_index():
    try:
        res = requests.get("https://api.alternative.me/fng/", timeout=3).json()
        if res.get("data"):
            val = res["data"][0]["value"]
            classification = res["data"][0]["value_classification"]
            return f"{val}/100 ({classification})"
    except Exception:
        pass
    return "N/A"

@st.cache_data(ttl=60)
def get_okx_candlesticks(coin_symbol, bar="1D", limit=100):
    clean_coin = coin_symbol.upper().replace("-USDT", "").replace("USDT", "").strip()
    inst_id = f"{clean_coin}-USDT"
    url = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("code") == "0" and data.get("data"):
            raw_candles = data["data"]
            df = pd.DataFrame(
                raw_candles,
                columns=[
                    "timestamp", "open", "high", "low", "close", "vol",
                    "volCcy", "volCcyQuote", "confirm",
                ],
            )
            for col in ["open", "high", "low", "close", "vol"]:
                df[col] = df[col].astype(float)
            df = df.iloc[::-1].reset_index(drop=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60)
def get_okx_taker_volume(coin="BTC"):
    clean_coin = coin.upper().replace("-USDT", "").replace("USDT", "").strip()
    url = f"https://www.okx.com/api/v5/rubik/stat/taker-volume?ccy={clean_coin}&contractType=SWAP"
    try:
        res = requests.get(url, timeout=3).json()
        if res.get("code") == "0" and res.get("data"):
            latest = res["data"][0]
            buy_vol = float(latest[1])
            sell_vol = float(latest[2])
            ratio = round(buy_vol / sell_vol, 2) if sell_vol > 0 else 1.0

            if ratio > 1.2:
                flow_status = f"🟢 MUA CHỦ ĐỘNG MẠNH (Ratio: {ratio})"
            elif ratio < 0.8:
                flow_status = f"🔴 BÁN CHỦ ĐỘNG MẠNH (Ratio: {ratio})"
            else:
                flow_status = f"⚖️ CÂN BẰNG MUA/BÁN (Ratio: {ratio})"

            return {
                "taker_ratio": ratio,
                "buy_vol": round(buy_vol, 2),
                "sell_vol": round(sell_vol, 2),
                "flow_status": flow_status,
            }
    except Exception:
        pass
    return {"flow_status": "N/A", "taker_ratio": 1.0}

@st.cache_data(ttl=300)
def get_okx_derivatives_data(coin="BTC"):
    clean_coin = coin.upper().replace("-USDT", "").replace("USDT", "").strip()
    inst_id = f"{clean_coin}-USDT-SWAP"
    url_funding = f"https://www.okx.com/api/v5/public/funding-rate?instId={inst_id}"
    url_oi = f"https://www.okx.com/api/v5/market/open-interest?instId={inst_id}"

    funding_rate = "N/A"
    open_interest = "N/A"

    try:
        res_f = requests.get(url_funding, timeout=3).json()
        if res_f.get("code") == "0" and res_f.get("data"):
            funding_val = float(res_f["data"][0]["fundingRate"]) * 100
            funding_rate = f"{funding_val:+.4f}%"

        res_oi = requests.get(url_oi, timeout=3).json()
        if res_oi.get("code") == "0" and res_oi.get("data"):
            oi_val = float(res_oi["data"][0]["oiCcy"])
            open_interest = f"${oi_val / 1e6:,.2f}M USDT"
    except Exception:
        pass

    return {"funding_rate": funding_rate, "open_interest": open_interest}

# -----------------------------------------------------------------------------
# 5. Thuật toán Chỉ báo Định lượng
# -----------------------------------------------------------------------------
def calculate_quant_indicators(df):
    if df.empty or len(df) < 30:
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    close_prev = close.shift(1)
    high_prev = high.shift(1)
    low_prev = low.shift(1)

    current_price = close.iloc[-1]

    ma20 = close_prev.rolling(window=20).mean().iloc[-1]
    ma50 = close_prev.rolling(window=50).mean().iloc[-1]

    if current_price > ma20 and ma20 > ma50:
        trend_score = 1.0
        trend_simple = "Tăng"
    elif current_price < ma20 and ma20 < ma50:
        trend_score = -1.0
        trend_simple = "Giảm"
    else:
        trend_score = 0.0
        trend_simple = "Đi ngang"

    rsi_series = ta.rsi(close_prev, length=14)
    rsi = rsi_series.iloc[-1] if rsi_series is not None else 50

    support_30 = low_prev.tail(30).min()
    resistance_30 = high_prev.tail(30).max()
    tr = np.maximum(
        high_prev - low_prev,
        np.maximum(
            abs(high_prev - close_prev.shift(1)),
            abs(low_prev - close_prev.shift(1)),
        ),
    )
    atr14 = tr.rolling(window=14).mean().iloc[-1]

    return {
        "rsi": round(rsi, 2),
        "rsi_status": (
            "Quá mua (>70)"
            if rsi > 70
            else ("Quá bán (<30)" if rsi < 30 else "Trung tính")
        ),
        "trend_score": trend_score,
        "trend_simple": trend_simple,
        "support": round(support_30, 4),
        "resistance": round(resistance_30, 4),
        "atr": round(atr14, 4),
        "stop_loss_atr": round(current_price - (2 * atr14), 4),
        "tp1": round(current_price + (2 * atr14), 4),
        "tp2": round(current_price + (4 * atr14), 4),
    }

def calculate_volume_metrics(df_1d):
    if df_1d.empty or len(df_1d) < 60:
        return {}

    vol_usd = df_1d["vol"] * df_1d["close"]
    vol_usd_prev = vol_usd.shift(1)

    current_vol_usd = vol_usd.iloc[-1]
    mean_20 = vol_usd_prev.tail(20).mean()
    std_20 = vol_usd_prev.tail(20).std()

    z_score = (current_vol_usd - mean_20) / (std_20 + 1e-10)

    obv = (np.sign(df_1d["close"].diff()) * df_1d["vol"]).fillna(0).cumsum()
    obv_trend = (
        "Dòng tiền vào (OBV Tăng)"
        if obv.iloc[-1] > obv.tail(30).mean()
        else "Dòng tiền rút (OBV Giảm)"
    )

    if z_score >= 2.0:
        vol_status = f"🔥 THỊ TRƯỜNG MẠNH (Z-Score: +{z_score:.2f}σ)"
    elif z_score <= -1.5:
        vol_status = f"❄️ CẠN KIỆT THANH KHOẢN (Z-Score: {z_score:.2f}σ)"
    else:
        vol_status = f"⚖️ BÌNH THƯỜNG (Z-Score: {z_score:.2f}σ)"

    return {
        "vol_usd_mil": round(current_vol_usd / 1e6, 2),
        "z_score": round(z_score, 2),
        "vol_status": vol_status,
        "obv_trend": obv_trend,
    }

# -----------------------------------------------------------------------------
# 6. Sidebar: Quản lý Chat & TRA CỨU DYNAMIC COIN
# -----------------------------------------------------------------------------
st.sidebar.title("📈 QuangAnh Investment Agent")
fng_index = get_fear_and_greed_index()
st.sidebar.caption(f"😱 Fear & Greed Index: **{fng_index}**")

# --- Quản lý Chat ---
st.sidebar.subheader("💬 Quản lý Lịch sử Chat")
if st.sidebar.button("➕ Tạo cuộc trò chuyện mới"):
    st.session_state.chat_sessions.append(
        {
            "title": f"Trò chuyện {len(st.session_state.chat_sessions)+1}",
            "messages": [],
        }
    )
    st.session_state.active_index = len(st.session_state.chat_sessions) - 1
    save_chat_sessions(st.session_state.chat_sessions)
    st.rerun()

session_titles = [
    f"{i+1}. {s.get('title', 'Trò chuyện')}"
    for i, s in enumerate(st.session_state.chat_sessions)
]
selected_session = st.sidebar.selectbox(
    "Chọn cuộc trò chuyện:",
    options=range(len(session_titles)),
    format_func=lambda x: session_titles[x],
    index=min(st.session_state.active_index, len(session_titles) - 1),
)
st.session_state.active_index = selected_session

col_del1, col_del2 = st.sidebar.columns(2)
with col_del1:
    if st.button("🗑️ Xóa hội thoại"):
        if len(st.session_state.chat_sessions) > 1:
            st.session_state.chat_sessions.pop(st.session_state.active_index)
            st.session_state.active_index = max(0, st.session_state.active_index - 1)
        else:
            st.session_state.chat_sessions = [{"title": "Cuộc trò chuyện mới", "messages": []}]
            st.session_state.active_index = 0
        save_chat_sessions(st.session_state.chat_sessions)
        st.rerun()

with col_del2:
    if st.button("⚠️ Xóa tất cả"):
        st.session_state.chat_sessions = [{"title": "Cuộc trò chuyện mới", "messages": []}]
        st.session_state.active_index = 0
        save_chat_sessions(st.session_state.chat_sessions)
        st.rerun()

st.sidebar.markdown("---")

# --- CHỌN COIN TỰ DO TRÊN SIDEBAR ---
col_title, col_btn = st.sidebar.columns([3, 1])
with col_title:
    st.subheader("🔍 Tra Cứu Coin Tự Do")
with col_btn:
    if st.button("🔄"):
        st.cache_data.clear()
        st.rerun()

selected_coin = st.sidebar.selectbox(
    "Chọn hoặc gõ tên đồng Coin:",
    options=["BTC", "ETH", "SOL", "PI", "BNB", "XRP", "DOGE", "ADA", "NEAR", "SUI", "LINK", "AVAX"],
    index=0
)

# Fetch dữ liệu động
coin_price = get_single_coin_price(selected_coin)
price_display = f"${coin_price:,.4f}" if isinstance(coin_price, (int, float)) else "N/A"

tf_data = {}
for tf in ["1D", "4H", "1H"]:
    limit_val = 365 if tf == "1D" else 100
    df = get_okx_candlesticks(selected_coin, bar=tf, limit=limit_val)
    tf_data[tf] = calculate_quant_indicators(df)

df_1d = get_okx_candlesticks(selected_coin, bar="1D", limit=100)
vol_metrics = calculate_volume_metrics(df_1d)
taker_info = get_okx_taker_volume(selected_coin)
deriv_info = get_okx_derivatives_data(selected_coin)

# Hiển thị thông tin coin trên Sidebar
st.sidebar.markdown(f"### 📌 **{selected_coin.upper()}**: `{price_display}`")

if coin_price != "N/A":
    st.sidebar.caption(f"• **RSI (14) Khung 1D:** {tf_data['1D'].get('rsi', 'N/A')} ({tf_data['1D'].get('rsi_status', '')})")
    st.sidebar.caption(f"• **Xu hướng 1D:** {tf_data['1D'].get('trend_simple', 'N/A')}")
    st.sidebar.caption(f"• **Hỗ trợ / Kháng cự:** `${tf_data['1D'].get('support', 'N/A')}` / `${tf_data['1D'].get('resistance', 'N/A')}`")
    st.sidebar.caption(f"• **Taker Flow:** {taker_info.get('flow_status')}")
    st.sidebar.caption(f"• **Funding Rate:** `{deriv_info.get('funding_rate')}`")
    st.sidebar.caption(f"• **Volume Status:** {vol_metrics.get('vol_status', 'N/A')}")
else:
    st.sidebar.warning(f"Chưa lấy được dữ liệu real-time cho {selected_coin.upper()}. Vui lòng kiểm tra lại Ticker.")

# -----------------------------------------------------------------------------
# 7. Main Area: Chatbot AI Agent
# -----------------------------------------------------------------------------
current_session = st.session_state.chat_sessions[st.session_state.active_index]
messages = current_session["messages"]

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi chiến lược đa khung thời gian & dòng tiền cá voi..."):
    if not messages:
        current_session["title"] = prompt[:20] + ("..." if len(prompt) > 20 else "")

    messages.append({"role": "user", "content": prompt})
    save_chat_sessions(st.session_state.chat_sessions)

    with st.chat_message("user"):
        st.markdown(prompt)

    # ĐÓNG GÓI CHÍNH XÁC DỮ LIỆU REAL-TIME CỦA COIN ĐANG CHỌN CHO AGENT
    system_instruction_text = f"""
{STRICT_SYSTEM_PROMPT}

[DỮ LIỆU VĨ MÔ & LÃI SUẤT REAL-TIME]
- Lãi suất điều hành FED: 5.25% - 5.50%
- Chỉ số DXY (Sức mạnh USD): {macro_info.get('DXY')} ({macro_info.get('DXY_Status')} {macro_info.get('DXY_Change_%')}%)
- Lợi suất trái phiếu Mỹ 10 năm (US10Y): {macro_info.get('US10Y')}
- Fear & Greed Index: {fng_index}

[DỮ LIỆU CHÍNH XÁC REAL-TIME DÀNH RIÊNG CHO COIN: {selected_coin.upper()}]
- Giá Spot Thực Tế: {price_display}
- Khung 1D: Xu hướng = {tf_data['1D'].get('trend_simple', 'N/A')} | RSI(14) = {tf_data['1D'].get('rsi', 'N/A')} ({tf_data['1D'].get('rsi_status', '')})
- Vùng Hỗ trợ 30 ngày: ${tf_data['1D'].get('support', 'N/A')}
- Vùng Kháng cự 30 ngày: ${tf_data['1D'].get('resistance', 'N/A')}
- Mức Dừng lỗ gợi ý (ATR 2x): ${tf_data['1D'].get('stop_loss_atr', 'N/A')}
- Mục tiêu Chốt lời TP1: ${tf_data['1D'].get('tp1', 'N/A')} | TP2: ${tf_data['1D'].get('tp2', 'N/A')}
- Lực Mua/Bán chủ động (Taker Flow): {taker_info.get('flow_status')}
- Phái sinh: Funding Rate = {deriv_info.get('funding_rate')} | Open Interest = {deriv_info.get('open_interest')}
- Biến động Volume USD: {vol_metrics.get('vol_status', 'N/A')} | OBV: {vol_metrics.get('obv_trend', 'N/A')}

Nhiệm vụ của Cố vấn:
1. Đánh giá bối cảnh Vĩ mô (DXY, US10Y) tác động thế nào tới {selected_coin.upper()}.
2. Phân tích CHI TIẾT DỰA TRÊN ĐÚNG MỨC GIÁ THỰC TẾ {price_display} CỦA {selected_coin.upper()} NÊU TRÊN.
3. Thiết lập Kế hoạch Giao dịch cụ thể: Mức giá Entry (xung quanh {price_display}), Điểm Cắt lỗ (Stop Loss) và Chốt lời (Take Profit) dựa đúng trên các mốc Hỗ trợ/Kháng cự/ATR đã cung cấp.
"""

    formatted_history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        formatted_history.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )

    client = genai.Client(api_key=api_key)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            chat = client.chats.create(
                model="gemini-3.6-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction_text, temperature=0.3
                ),
                history=formatted_history,
            )
            response_stream = chat.send_message_stream(prompt)
            for chunk in response_stream:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")

            for forbidden_word in ["IOU", "iou", "Futures"]:
                full_response = full_response.replace(forbidden_word, "Spot Mainnet")

            message_placeholder.markdown(full_response)
            messages.append({"role": "assistant", "content": full_response})
            save_chat_sessions(st.session_state.chat_sessions)

        except Exception as e:
            message_placeholder.markdown(f"❌ Có lỗi xảy ra: {e}")
