import json
import os

# import ccxt
from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import streamlit as st

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
# 2. System Prompt nghiêm ngặt (Khóa quan điểm IOU của Pi Network)
# -----------------------------------------------------------------------------
STRICT_SYSTEM_PROMPT = """
Bạn là QuangAnh Investment Agent - Cố vấn tài chính định lượng cao cấp.

[QUY TẮC BẮT BUỘC 100% VỀ PI NETWORK]:
1. Pi Network (PI) ĐÃ CHÍNH THỨC RA MẮT OPEN MAINNET THỰC TẾ và giao dịch trực tiếp trên OKX (PI/USDT).
2. TUYỆT ĐỐI KHÔNG sử dụng các từ: "IOU", "Futures", "Hợp đồng tương lai", "Chưa niêm yết".
3. Mọi phân tích kỹ thuật và dòng tiền của PI dựa hoàn toàn trên dữ liệu Mainnet thực tế từ OKX.
"""


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
# 4. Fetch Data APIs (OKX Mainnet, Alternative.me & Dòng tiền Nâng cao)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=5)
def get_realtime_prices():
    symbols = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "PI": "PI-USDT"}
    prices = {}
    for coin, inst_id in symbols.items():
        url = f"https://www.okx.com/api/v5/market/ticker?instId={inst_id}"
        try:
            res = requests.get(url, timeout=3).json()
            if res.get("code") == "0" and res.get("data"):
                prices[coin] = float(res["data"][0]["last"])
            else:
                prices[coin] = "N/A"
        except Exception:
            prices[coin] = "N/A"
    return prices


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
def get_okx_candlesticks(inst_id="BTC-USDT", bar="1D", limit=100):
    url = f"https://www.okx.com/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("code") == "0" and data.get("data"):
            raw_candles = data["data"]
            df = pd.DataFrame(
                raw_candles,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "volCcy",
                    "volCcyQuote",
                    "confirm",
                ],
            )
            for col in ["open", "high", "low", "close", "vol"]:
                df[col] = df[col].astype(float)
            df = df.iloc[::-1].reset_index(drop=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()


# --- BỔ SUNG: API DÒNG TIỀN NÂNG CAO (TAKER FLOW & PHÁI SINH) ---
@st.cache_data(ttl=60)
def get_okx_taker_volume(coin="BTC"):
    """Lấy tỷ lệ lực Mua/Bán chủ động (Taker Buy/Sell Ratio)"""
    url = f"https://www.okx.com/api/v5/rubik/stat/taker-volume?ccy={coin}&contractType=SWAP"
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
def get_okx_derivatives_data(inst_id="BTC-USDT-SWAP"):
    """Lấy Funding Rate & Open Interest (Hợp đồng mở)"""
    url_funding = (
        f"https://www.okx.com/api/v5/public/funding-rate?instId={inst_id}"
    )
    url_oi = f"https://www.okx.com/api/v5/market/open-interest?instId={inst_id}"

    funding_rate = "N/A"
    open_interest = "N/A"

    try:
        # Funding Rate
        res_f = requests.get(url_funding, timeout=3).json()
        if res_f.get("code") == "0" and res_f.get("data"):
            funding_val = float(res_f["data"][0]["fundingRate"]) * 100
            funding_rate = f"{funding_val:+.4f}%"

        # Open Interest
        res_oi = requests.get(url_oi, timeout=3).json()
        if res_oi.get("code") == "0" and res_oi.get("data"):
            oi_val = float(res_oi["data"][0]["oiCcy"])
            open_interest = f"${oi_val / 1e6:,.2f}M USDT"
    except Exception:
        pass

    return {"funding_rate": funding_rate, "open_interest": open_interest}


# -----------------------------------------------------------------------------
# 5. Thuật toán Chỉ báo Định lượng (Anti Look-Ahead & Z-Score Volume)
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

    # Moving Averages
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

    # RSI
    rsi_series = ta.rsi(close_prev, length=14)
    rsi = rsi_series.iloc[-1] if rsi_series is not None else 50

    # ATR & Levels
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
        "support": round(support_30, 2),
        "resistance": round(resistance_30, 2),
        "atr": round(atr14, 2),
        "stop_loss_atr": round(current_price - (2 * atr14), 2),
        "tp1": round(current_price + (2 * atr14), 2),
        "tp2": round(current_price + (4 * atr14), 2),
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
    vol_percentile = (vol_usd_prev < current_vol_usd).mean() * 100

    obv = (np.sign(df_1d["close"].diff()) * df_1d["vol"]).fillna(0).cumsum()
    obv_trend = (
        "Dòng tiền vào (OBV Tăng)"
        if obv.iloc[-1] > obv.tail(30).mean()
        else "Dòng tiền rút (OBV Giảm)"
    )

    if z_score >= 2.0:
        vol_status = f"🔥 BÙNG NỔ CÁ VOI (Z-Score: +{z_score:.1f}σ)"
    elif z_score <= -1.5:
        vol_status = f"❄️ CẠN KIỆT THANH KHOẢN (Z-Score: {z_score:.1f}σ)"
    else:
        vol_status = f"⚖️ Bình thường (Z-Score: {z_score:.1f}σ)"

    return {
        "vol_usd_mil": round(current_vol_usd / 1e6, 2),
        "z_score": round(z_score, 2),
        "vol_percentile": round(vol_percentile, 1),
        "vol_status": vol_status,
        "obv_trend": obv_trend,
    }


# -----------------------------------------------------------------------------
# 6. Xử lý Dữ liệu Đa Khung Thời Gian & Dòng Tiền Thật
# -----------------------------------------------------------------------------
realtime_prices = get_realtime_prices()
fng_index = get_fear_and_greed_index()
symbols = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "PI": "PI-USDT"}
timeframes = ["1D", "4H", "1H"]
market_data = {}

for name, inst_id in symbols.items():
    tf_data = {}
    vol_metrics = {}

    for tf in timeframes:
        limit_val = 365 if tf == "1D" else 100
        df = get_okx_candlesticks(inst_id, bar=tf, limit=limit_val)
        tf_data[tf] = calculate_quant_indicators(df)
        if tf == "1D":
            vol_metrics = calculate_volume_metrics(df)

    s_1d = tf_data["1D"].get("trend_score", 0)
    s_4h = tf_data["4H"].get("trend_score", 0)
    s_1h = tf_data["1H"].get("trend_score", 0)

    mtf_score = (s_1d * 0.5) + (s_4h * 0.3) + (s_1h * 0.2)

    if mtf_score >= 0.8:
        mtf_status = "🔥 TĂNG MẠNH (Đồng thuận cao)"
    elif 0.2 < mtf_score < 0.8:
        mtf_status = "📈 TĂNG YẾU / Điều chỉnh ngắn"
    elif mtf_score <= -0.8:
        mtf_status = "📉 GIẢM MẠNH (Đồng thuận cao)"
    elif -0.8 < mtf_score < -0.2:
        mtf_status = "📉 GIẢM YẾU / Hồi phục ngắn"
    else:
        mtf_status = "🔄 XUNG ĐỘT KHUNG (Sideway / Rủi ro)"

    # LẤY DỮ LIỆU DÒNG TIỀN NÂNG CAO
    taker_flow = get_okx_taker_volume(name)
    swap_id = f"{name}-USDT-SWAP"
    deriv_data = get_okx_derivatives_data(swap_id)

    market_data[name] = {
        "price": realtime_prices.get(name, "N/A"),
        "mtf_score": round(mtf_score, 2),
        "mtf_status": mtf_status,
        "vol_metrics": vol_metrics,
        "taker_flow": taker_flow,  # Mới
        "deriv_data": deriv_data,  # Mới
        "tf_1d": tf_data["1D"],
        "tf_4h": tf_data["4H"],
        "tf_1h": tf_data["1H"],
    }

# -----------------------------------------------------------------------------
# 7. Sidebar: Quản lý Chat & Hiển thị Chỉ báo Thu Gọn (Scrollable)
# -----------------------------------------------------------------------------
st.sidebar.title("📈 QuangAnh Investment Agent")
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
            st.session_state.active_index = max(
                0, st.session_state.active_index - 1
            )
        else:
            st.session_state.chat_sessions = [
                {"title": "Cuộc trò chuyện mới", "messages": []}
            ]
            st.session_state.active_index = 0
        save_chat_sessions(st.session_state.chat_sessions)
        st.rerun()

with col_del2:
    if st.button("⚠️ Xóa tất cả"):
        st.session_state.chat_sessions = [
            {"title": "Cuộc trò chuyện mới", "messages": []}
        ]
        st.session_state.active_index = 0
        save_chat_sessions(st.session_state.chat_sessions)
        st.rerun()

st.sidebar.markdown("---")

# --- Hiển thị Chỉ Báo Crypto Thu Gọn Trong Expander & Scroll ---
col_title, col_btn = st.sidebar.columns([3, 1])
with col_title:
    st.subheader("🔍 Chỉ Báo Real-time")
with col_btn:
    if st.button("🔄"):
        st.cache_data.clear()
        st.rerun()

for coin, data in market_data.items():
    price_display = (
        f"${data['price']:,.4f}"
        if isinstance(data["price"], (int, float))
        else data["price"]
    )
    vol_info = data["vol_metrics"]
    taker_info = data["taker_flow"]
    deriv_info = data["deriv_data"]

    # Dùng Expander thu gọn không gian
    with st.sidebar.expander(f"📌 **{coin}**: {price_display}", expanded=False):
        with st.container(height=260):
            st.markdown(f"**Giá Spot OKX:** `{price_display}`")
            st.caption(
                f"• **Trạng thái MTF:** {data['mtf_status']} (Score: {data['mtf_score']})"
            )

            st.markdown("##### 🌊 Dòng Tiền Chủ Động & Phái Sinh")
            st.caption(f"• **Taker Flow:** {taker_info.get('flow_status')}")
            st.caption(
                f"• **Funding Rate:** `{deriv_info.get('funding_rate')}`"
            )
            st.caption(
                f"• **Hợp đồng mở (OI):** `{deriv_info.get('open_interest')}`"
            )

            st.markdown("##### 📈 Kỹ thuật 1D")
            st.caption(
                f"• **RSI (14):** {data['tf_1d'].get('rsi', 'N/A')} ({data['tf_1d'].get('rsi_status', '')})"
            )
            st.caption(
                f"• **HT/KC:** `${data['tf_1d'].get('support', 'N/A')}` / `${data['tf_1d'].get('resistance', 'N/A')}`"
            )

            st.markdown("##### 📊 Volume USD")
            st.caption(
                f"• **Volume Status:** {vol_info.get('vol_status', 'N/A')}"
            )
            st.caption(f"• **Dòng tiền OBV:** {vol_info.get('obv_trend', 'N/A')}")

# -----------------------------------------------------------------------------
# 8. Main Area: Chatbot AI Agent
# -----------------------------------------------------------------------------
current_session = st.session_state.chat_sessions[
    st.session_state.active_index
]
messages = current_session["messages"]

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(
    "Hỏi chiến lược đa khung thời gian & dòng tiền cá voi..."
):
    if not messages:
        current_session["title"] = prompt[:20] + (
            "..." if len(prompt) > 20 else ""
        )

    messages.append({"role": "user", "content": prompt})
    save_chat_sessions(st.session_state.chat_sessions)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Ghép Strict System Prompt + Ground Truth Data Đã Nâng Cấp Dòng Tiền
    system_instruction_text = f"""
{STRICT_SYSTEM_PROMPT}

TÂM LÝ THỊ TRƯỜNG & DÒNG TIỀN:
- Fear & Greed Index: {fng_index}

DỮ LIỆU THỰC TẾ ĐỊNH LƯỢNG (OKX MAINNET):
1. BTC:
   - Giá: ${market_data['BTC']['price']}
   - Lực Mua/Bán chủ động (Taker Flow): {market_data['BTC']['taker_flow']['flow_status']}
   - Phái sinh: Funding Rate = {market_data['BTC']['deriv_data']['funding_rate']} | Open Interest = {market_data['BTC']['deriv_data']['open_interest']}
   - MTF Status: {market_data['BTC']['mtf_status']} (Score: {market_data['BTC']['mtf_score']})
   - Volume USD Z-Score: {market_data['BTC']['vol_metrics'].get('vol_status')} | OBV: {market_data['BTC']['vol_metrics'].get('obv_trend')}
   - Khung 1D: Trend = {market_data['BTC']['tf_1d'].get('trend_simple')} | RSI = {market_data['BTC']['tf_1d'].get('rsi')}

2. ETH:
   - Giá: ${market_data['ETH']['price']}
   - Lực Mua/Bán chủ động (Taker Flow): {market_data['ETH']['taker_flow']['flow_status']}
   - Phái sinh: Funding Rate = {market_data['ETH']['deriv_data']['funding_rate']} | Open Interest = {market_data['ETH']['deriv_data']['open_interest']}
   - MTF Status: {market_data['ETH']['mtf_status']} (Score: {market_data['ETH']['mtf_score']})
   - Volume USD Z-Score: {market_data['ETH']['vol_metrics'].get('vol_status')} | OBV: {market_data['ETH']['vol_metrics'].get('obv_trend')}
   - Khung 1D: Trend = {market_data['ETH']['tf_1d'].get('trend_simple')} | RSI = {market_data['ETH']['tf_1d'].get('rsi')}

3. PI NETWORK (PI MAINNET OKX):
   - Giá Spot: ${market_data['PI']['price']}
   - MTF Status: {market_data['PI']['mtf_status']} (Score: {market_data['PI']['mtf_score']})
   - Volume USD Z-Score: {market_data['PI']['vol_metrics'].get('vol_status')} | OBV: {market_data['PI']['vol_metrics'].get('obv_trend')}
   - Khung 1D: Trend = {market_data['PI']['tf_1d'].get('trend_simple')} | RSI = {market_data['PI']['tf_1d'].get('rsi')}

Nhiệm vụ:
1. Đưa ra nhận định dựa trên sự kết hợp giữa MTF Score, Taker Flow (Lực mua/bán chủ động thực tế) và Funding Rate/OI.
2. Thiết lập Kế hoạch Giao dịch (Entry, Stop Loss theo ATR, Take Profit).
"""

    formatted_history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        formatted_history.append(
            types.Content(
                role=role, parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    client = genai.Client(api_key=api_key)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            chat = client.chats.create(
                model="gemini-3.5-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction_text, temperature=0.3
                ),
                history=formatted_history,
            )
            response_stream = chat.send_message_stream(prompt)
            for chunk in response_stream:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")

            # Lọc sạch từ ngữ IOU nếu Gemini cố vi phạm
            for forbidden_word in ["IOU", "iou", "Futures"]:
                full_response = full_response.replace(
                    forbidden_word, "Spot Mainnet"
                )

            message_placeholder.markdown(full_response)
            messages.append({"role": "assistant", "content": full_response})
            save_chat_sessions(st.session_state.chat_sessions)

        except Exception as e:
            message_placeholder.markdown(f"❌ Có lỗi xảy ra: {e}")
