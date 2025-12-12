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
# 1. 페이지 설정 및 디자인 (모바일 최적화)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Logic Pro", page_icon="💎", layout="wide")

st.markdown("""
<style>
    /* 전체 배경 및 폰트 설정 */
    .main { background-color: #ffffff; }
    
    /* 카드 스타일 */
    div.stMetric {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* 탭 디자인 */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 10px; 
        flex-wrap: wrap; /* 모바일에서 탭 줄바꿈 허용 */
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        background-color: #f1f3f5;
        border: none;
        margin-bottom: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
    }
    
    /* 리스크 뱃지 */
    .risk-badge {
        background-color: #ffebee;
        color: #c62828;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: bold;
        border: 1px solid #ef9a9a;
        margin-left: 5px;
        white-space: nowrap; /* 줄바꿈 방지 */
    }
    
    /* 모바일용 버튼 크기 조정 */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 45px; /* 터치하기 좋은 높이 */
    }
</style>
""", unsafe_allow_html=True)

st.title("💎 Quant Logic Mobile")
st.caption(f"기준: {datetime.datetime.now().strftime('%m-%d %H:%M')} (20분 지연)")

# -----------------------------------------------------------------------------
# 2. 포트폴리오 저장 관리 (자동 저장)
# -----------------------------------------------------------------------------
PORTFOLIO_FILE = "my_portfolio.csv"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try: return pd.read_csv(PORTFOLIO_FILE)['stock_name'].tolist()
        except: return []
    return []

def save_portfolio(stock_list):
    pd.DataFrame({'stock_name': stock_list}).to_csv(PORTFOLIO_FILE, index=False)

if 'my_portfolio' not in st.session_state: st.session_state['my_portfolio'] = load_portfolio()
if 'market_results' not in st.session_state: st.session_state['market_results'] = []
if 'analysis_cache' not in st.session_state: st.session_state['analysis_cache'] = []

# -----------------------------------------------------------------------------
# 3. 데이터 및 분석 엔진
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
        response = requests.get(url, headers=headers, timeout=2)
        html = response.text
        risks = []
        if "alt=\"관리종목\"" in html: risks.append("관리")
        if "alt=\"거래정지\"" in html: risks.append("정지")
        if "alt=\"투자경고\"" in html: risks.append("경고")
        if "alt=\"투자주의\"" in html: risks.append("주의")
        return risks
    except: return []

def get_sentiment(code):
    try:
        url = f"https://finance.naver.com/item/news_news.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=2)
        soup = BeautifulSoup(response.text, 'html.parser')
        titles = soup.select('.title')
        score = 0
        headline = "-"
        good = ['체결', '수주', '돌파', '역대', '최대', '급등', '강세', '성장', '기대', '매수', '호재', '실적', '흑자']
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
    rs = (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
          df['Close'].diff().clip(upper=0).abs().rolling(14).mean())
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Target_Price'] = df['Open'] + ((df['High'] - df['Low']).shift(1) * 0.5)
    df['VBO_Signal'] = np.where(df['Close'] > df['Target_Price'], 1, 0)
    df['Noise'] = 1 - (np.abs(df['Close']-df['Open']) / (df['High']-df['Low']))
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    cols = ['Close', 'RSI', 'VBO_Signal', 'Noise', 'MA5', 'MA20', 'MA60']
    df_clean = df.dropna(subset=cols).copy()
    if len(df_clean) < 10: return None
    
    X = df_clean[cols].iloc[:-1]
    y = df_clean['Target'].iloc[:-1]
    last_row = df_clean[cols].iloc[[-1]]
    
    try:
        model = xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)
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
    fig.add_hline(y=item['target_price'], line_dash="dash", line_color="#00C853", annotation_text="Target")
    fig.add_hline(y=item['stop_loss'], line_dash="dash", line_color="#D50000", annotation_text="Cut")
    fig.update_layout(
        title=dict(text=f"<b>{item['name']}</b>", font=dict(size=15)),
        height=300, # 모바일 맞춤 높이
        xaxis_rangeslider_visible=False, 
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------------------------------------------------------
# 4. 사이드바 (설정용으로만 사용)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 설정 (Settings)")
market_type = st.sidebar.selectbox("시장 (Market)", ["KOSPI", "KOSDAQ"])
top_n = st.sidebar.slider("스캔 개수", 10, 50, 20)
st.sidebar.info("💡 종목 추가는 메인 화면(내 포트폴리오 탭)에서 할 수 있습니다.")
all_stocks = get_stock_listing(market_type)

# -----------------------------------------------------------------------------
# 5. 메인 UI (모바일 최적화)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["💼 내 포트폴리오", "🚀 시장 추천"])

# [Tab 1] 내 포트폴리오 관리 (메인 화면으로 이동!)
with tab1:
    # 1. 종목 관리 섹션 (접이식)
    with st.expander("➕ 종목 추가 및 관리 (여기를 누르세요)", expanded=not bool(st.session_state.my_portfolio)):
        if not all_stocks.empty:
            col_sel, col_add = st.columns([3, 1])
            with col_sel:
                selected_stock = st.selectbox("종목 검색", ["선택..."] + all_stocks['Name'].tolist(), label_visibility="collapsed")
            with col_add:
                if st.button("추가"):
                    if selected_stock != "선택..." and selected_stock not in st.session_state.my_portfolio:
                        st.session_state.my_portfolio.append(selected_stock)
                        save_portfolio(st.session_state.my_portfolio)
                        st.rerun()
            
            # 현재 종목 리스트 (삭제 버튼)
            if st.session_state.my_portfolio:
                st.write("📋 보유 종목 목록:")
                cols = st.columns(3) # 3열로 배치
                for i, stock in enumerate(st.session_state.my_portfolio):
                    if cols[i % 3].button(f"🗑️ {stock}", key=f"del_{stock}", help="삭제"):
                        st.session_state.my_portfolio.remove(stock)
                        save_portfolio(st.session_state.my_portfolio)
                        st.rerun()
        else:
            st.error("데이터 로딩 중...")

    # 2. 진단 실행 버튼
    if st.session_state.my_portfolio:
        if st.button("🔄 내 종목 진단 실행", type="primary"):
            with st.spinner('분석 중...'):
                res_list = []
                for s_name in st.session_state.my_portfolio:
                    try:
                        code = all_stocks[all_stocks['Name'] == s_name]['Code'].values[0]
                        r = analyze_logic(code, s_name)
                        if r: res_list.append(r)
                    except: continue
                st.session_state['analysis_cache'] = res_list
    else:
        st.info("위의 메뉴를 열어 종목을 추가해주세요.")

    # 3. 분석 결과 카드 뷰
    if st.session_state['analysis_cache']:
        for item in st.session_state['analysis_cache']:
            with st.container():
                # 헤더
                c_head, c_score = st.columns([2.5, 1])
                risk_tags = "".join([f" <span class='risk-badge'>⚠️{r}</span>" for r in item['risks']])
                c_head.markdown(f"**{item['name']}** {risk_tags}", unsafe_allow_html=True)
                
                score = item['final_score']
                color = "green" if score >= 70 else "orange" if score >= 50 else "red"
                c_score.markdown(f"<span style='color:{color}; font-size:1.2em; font-weight:bold'>{score:.1f}점</span>", unsafe_allow_html=True)
                
                # 핵심 데이터 (2열 배치로 모바일 공간 확보)
                c1, c2 = st.columns(2)
                c1.metric("현재가", f"{item['price']:,}")
                c1.caption(f"뉴스: {item['headline'][:15]}...") # 뉴스 제목 길이 제한
                c2.metric("목표가", f"{item['target_price']:,}", delta=f"{item['target_price']-item['price']:,}")
                c2.metric("손절가", f"{item['stop_loss']:,}", delta=f"{item['stop_loss']-item['price']:,}", delta_color="inverse")
                
                # 차트 (버튼 없이 바로 보여주기 or 확장)
                with st.expander("차트 보기"):
                    st.plotly_chart(create_chart(item), use_container_width=True, key=f"chart_{item['code']}")
                    if item['risks']: st.error(f"주의: {', '.join(item['risks'])}")
                st.divider()

# [Tab 2] 시장 추천
with tab2:
    if st.button("🚀 Top Picks 스캔", type="primary"):
        st.info(f"상위 {top_n}개 종목 분석 중...")
        bar = st.progress(0)
        target_df = all_stocks.head(top_n)
        m_res = []
        total = len(target_df)
        for idx, (i, row) in enumerate(target_df.iterrows()):
            bar.progress(min((idx+1)/total, 1.0))
            if check_risk_status(row['Code']): continue
            r = analyze_logic(row['Code'], row['Name'])
            if r: m_res.append(r)
        bar.empty()
        st.session_state['market_results'] = sorted(m_res, key=lambda x: x['final_score'], reverse=True)
        st.rerun()

    if st.session_state['market_results']:
        best = st.session_state['market_results'][0]
        st.success(f"🏆 1위: **{best['name']}** ({best['final_score']:.1f}점)")
        st.divider()
        for i, item in enumerate(st.session_state['market_results']):
            # 탭1과 동일한 카드 레이아웃 적용
            with st.container():
                c_head, c_score = st.columns([2.5, 1])
                c_head.markdown(f"**{item['name']}**", unsafe_allow_html=True)
                score = item['final_score']
                color = "green" if score >= 70 else "orange" if score >= 50 else "red"
                c_score.markdown(f"<span style='color:{color}; font-weight:bold'>{score:.1f}</span>", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("현재가", f"{item['price']:,}")
                c2.metric("목표가", f"{item['target_price']:,}")
                
                with st.expander("상세 보기"):
                    st.plotly_chart(create_chart(item), use_container_width=True, key=f"m_chart_{i}")
                    st.info(item['headline'])
                st.divider()
