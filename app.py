import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from bs4 import BeautifulSoup
import datetime
import plotly.graph_objects as go
import os

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 디자인
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Logic Pro", page_icon="💎", layout="wide")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    div.stMetric {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    .risk-badge {
        background-color: #ffebee;
        color: #c62828;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        border: 1px solid #ef9a9a;
    }
</style>
""", unsafe_allow_html=True)

st.title("💎 Quant Logic : Pro Dashboard")
st.markdown("##### AI 기반 주식 매매 전략 시스템 (Auto-Save Supported)")

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"🕒 데이터 기준: {now} (약 20분 지연 실시간)")

# -----------------------------------------------------------------------------
# [신규 기능] 포트폴리오 저장 및 불러오기 함수
# -----------------------------------------------------------------------------
PORTFOLIO_FILE = "my_portfolio.csv"

def load_portfolio():
    """CSV 파일에서 저장된 종목 리스트를 불러옵니다."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            df = pd.read_csv(PORTFOLIO_FILE)
            return df['stock_name'].tolist()
        except:
            return []
    return []

def save_portfolio(stock_list):
    """종목 리스트를 CSV 파일로 저장합니다."""
    df = pd.DataFrame({'stock_name': stock_list})
    df.to_csv(PORTFOLIO_FILE, index=False)

# 세션 초기화 (파일에서 불러오기)
if 'my_portfolio' not in st.session_state:
    st.session_state['my_portfolio'] = load_portfolio() # 저장된 파일 로드

if 'market_results' not in st.session_state: st.session_state['market_results'] = []
if 'analysis_cache' not in st.session_state: st.session_state['analysis_cache'] = []

# -----------------------------------------------------------------------------
# 2. 데이터 및 분석 엔진
# -----------------------------------------------------------------------------
@st.cache_data
def get_stock_listing(market):
    try:
        df = fdr.StockListing(market)
        df = df[~df['Name'].str.contains('우|스팩|ETN|ETF|홀딩스')]
        return df
    except: return pd.DataFrame()

def get_stock_data(code, days=400):
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        df = fdr.DataReader(code, start_date, end_date)
        if df is None or df.empty: return None
        return df
    except: return None

def check_risk_status(code):
    try:
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        html = response.text
        risks = []
        if "alt=\"관리종목\"" in html: risks.append("관리종목")
        if "alt=\"거래정지\"" in html: risks.append("거래정지")
        if "alt=\"투자경고\"" in html: risks.append("투자경고")
        if "alt=\"투자주의\"" in html: risks.append("투자주의")
        if "alt=\"환기종목\"" in html: risks.append("환기종목")
        return risks
    except: return []

def get_sentiment(code):
    try:
        url = f"https://finance.naver.com/item/news_news.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        titles = soup.select('.title')
        score = 0
        headline = "-"
        good = ['체결', '수주', '돌파', '역대', '최대', '급등', '강세', '성장', '기대', '매수', '호재', '실적', '흑자', '공급']
        bad = ['하락', '약세', '적자', '우려', '매도', '불확실', '급락', '손실', '악재', '지연', '감소']
        if titles: headline = titles[0].get_text().strip()
        for t in titles[:5]:
            txt = t.get_text().strip()
            for w in good:
                if w in txt: score += 10
            for w in bad:
                if w in txt: score -= 10
        return score, headline
    except: return 0, "뉴스 없음"

def analyze_logic(code, name):
    risk_labels = check_risk_status(code)
    df = get_stock_data(code)
    if df is None: return None
    
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Range'] = (df['High'] - df['Low']).shift(1)
    df['Target_Price'] = df['Open'] + (df['Range'] * 0.5)
    df['VBO_Signal'] = np.where(df['Close'] > df['Target_Price'], 1, 0)
    rng = df['High'] - df['Low']
    df['Noise'] = np.where(rng > 0, 1 - (np.abs(df['Close']-df['Open']) / rng), 0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    cols = ['Close', 'RSI', 'VBO_Signal', 'Noise', 'MA5', 'MA20', 'MA60']
    df_clean = df.dropna(subset=cols).copy()
    if len(df_clean) < 10: return None
    
    X = df_clean[cols].iloc[:-1]
    y = df_clean['Target'].iloc[:-1]
    last_row = df_clean[cols].iloc[[-1]]
    
    try:
        model = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)
        model.fit(X, y)
        score = model.predict_proba(last_row)[0][1] * 100
        
        last_close = df['Close'].iloc[-1]
        volatility = (df['High'] - df['Low']).rolling(5).mean().iloc[-1]
        target_price = last_close + (volatility * 2.0)
        stop_loss = last_close - (volatility * 1.5)
        sent, head = get_sentiment(code)
        final = round((score * 0.7) + (sent + 50) * 0.3, 1)
        
        return {
            'code': code, 'name': name, 'price': int(last_close),
            'final_score': final, 'target_price': int(target_price),
            'stop_loss': int(stop_loss), 'headline': head, 'sentiment': sent,
            'last_data': df, 'risks': risk_labels
        }
    except: return None

def create_chart(item):
    df_chart = item['last_data'][-60:]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'))
    fig.add_hline(y=item['target_price'], line_dash="dash", line_color="#00C853", annotation_text="Target", annotation_position="top right")
    fig.add_hline(y=item['stop_loss'], line_dash="dash", line_color="#D50000", annotation_text="Stop Loss", annotation_position="bottom right")
    fig.update_layout(title=f"<b>{item['name']}</b> 전략 차트", height=350, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# -----------------------------------------------------------------------------
# 3. 사이드바 UI
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ 설정 패널")
market_type = st.sidebar.selectbox("시장 선택", ["KOSPI", "KOSDAQ"])
top_n = st.sidebar.slider("스캔 범위 (Top N)", 10, 50, 20)

st.sidebar.markdown("---")
st.sidebar.header("💼 내 포트폴리오")
all_stocks = get_stock_listing(market_type)

if not all_stocks.empty:
    selected_stock = st.sidebar.selectbox("종목 검색", ["종목 선택..."] + all_stocks['Name'].tolist())
    
    if st.sidebar.button("➕ 포트폴리오에 추가", type="primary", use_container_width=True):
        if selected_stock != "종목 선택...":
            if selected_stock not in st.session_state.my_portfolio:
                st.session_state.my_portfolio.append(selected_stock)
                save_portfolio(st.session_state.my_portfolio) # [저장] 파일에 쓰기
                st.rerun()

    st.sidebar.markdown("---")
    
    if st.session_state.my_portfolio:
        st.sidebar.caption(f"관리 종목: {len(st.session_state.my_portfolio)}개")
        for stock in st.session_state.my_portfolio:
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            col1.markdown(f":pushpin: **{stock}**")
            if col2.button("✖", key=f"del_{stock}"):
                st.session_state.my_portfolio.remove(stock)
                save_portfolio(st.session_state.my_portfolio) # [저장] 삭제 반영
                st.rerun()
        
        if st.sidebar.button("전체 초기화"):
            st.session_state.my_portfolio = []
            save_portfolio([]) # [저장] 초기화 반영
            st.rerun()
    else:
        st.sidebar.info("종목을 추가해주세요.")
else:
    st.sidebar.error("데이터 로딩 실패")

# -----------------------------------------------------------------------------
# 4. 메인 컨텐츠 UI (공통 렌더링 함수)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 내 종목 진단", "🚀 시장 전체 추천"])

def render_stock_card(item, key_prefix):
    with st.container():
        c_head, c_score = st.columns([3, 1])
        
        risk_tags = ""
        if item['risks']:
            for r in item['risks']: risk_tags += f" <span class='risk-badge'>⚠️ {r}</span>"
        
        c_head.subheader(f"📈 {item['name']}")
        if risk_tags: c_head.markdown(risk_tags, unsafe_allow_html=True)
        
        score_val = item['final_score']
        score_str = "{:.1f}".format(score_val)
        score_color = "green" if score_val >= 70 else "orange" if score_val >= 50 else "red"
        
        c_score.markdown(f"### <span style='color:{score_color}'>{score_str}점</span>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("현재가", f"{item['price']:,}원")
        c2.metric("목표가 (Target)", f"{item['target_price']:,}원", delta=f"{item['target_price']-item['price']:,}")
        c3.metric("손절가 (Stop Loss)", f"{item['stop_loss']:,}원", delta=f"{item['stop_loss']-item['price']:,}", delta_color="inverse")
        
        with st.expander("📊 상세 차트 및 뉴스 보기", expanded=False):
            st.plotly_chart(create_chart(item), use_container_width=True, key=f"chart_{key_prefix}")
            st.info(f"📰 **최신 뉴스**: {item['headline']} (감성점수: {item['sentiment']})")
            if item['risks']:
                st.error(f"🚫 **투자 주의**: 현재 **{', '.join(item['risks'])}** 상태입니다.")
        st.divider()

# [Tab 1] 내 보유 종목
with tab1:
    if not st.session_state.my_portfolio:
        st.info("👈 왼쪽 사이드바에서 관심 종목을 추가해주세요. (자동 저장됨)")
    else:
        if st.button("🔄 내 종목 진단 시작", use_container_width=True, type="primary"):
            with st.spinner('AI 분석 및 리스크 스캔 중...'):
                res_list = []
                for s_name in st.session_state.my_portfolio:
                    try:
                        code = all_stocks[all_stocks['Name'] == s_name]['Code'].values[0]
                        r = analyze_logic(code, s_name)
                        if r: res_list.append(r)
                    except: continue
                st.session_state['analysis_cache'] = res_list
        
        if st.session_state['analysis_cache']:
            for i, item in enumerate(st.session_state['analysis_cache']):
                render_stock_card(item, f"my_{i}")

# [Tab 2] 시장 추천
with tab2:
    if st.button("🚀 시장 전체 스캔 시작 (Top Picks)", use_container_width=True, type="primary"):
        st.info(f"KOSPI/KOSDAQ Top {top_n} 종목 정밀 분석 중...")
        bar = st.progress(0)
        target_df = all_stocks.head(top_n)
        m_res = []
        total = len(target_df)
        for idx, (i, row) in enumerate(target_df.iterrows()):
            bar.progress(min((idx+1)/total, 1.0))
            risks = check_risk_status(row['Code'])
            if risks: continue # 위험 종목 패스
            r = analyze_logic(row['Code'], row['Name'])
            if r: m_res.append(r)
        bar.empty()
        st.session_state['market_results'] = sorted(m_res, key=lambda x: x['final_score'], reverse=True)
        st.rerun()

    if st.session_state['market_results']:
        best = st.session_state['market_results'][0]
        st.success(f"🏆 현재 시장 1위 Pick: **{best['name']}** ({best['final_score']:.1f}점)")
        st.markdown("---")
        for i, item in enumerate(st.session_state['market_results']):
            render_stock_card(item, f"market_{i}")
    else:
        st.write("위의 '스캔 시작' 버튼을 눌러주세요.")