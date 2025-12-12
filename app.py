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
import yfinance as yf
import xml.etree.ElementTree as ET

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 CSS (속도 대폭 감속)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Logic V17.2", page_icon="🐢", layout="wide")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    div.stMetric {
        background-color: #f8f9fa; border: 1px solid #e9ecef;
        padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: auto; padding: 10px 20px; border-radius: 15px;
        background-color: #f1f3f5; border: none; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2196F3 !important; color: white !important;
    }
    
    /* 뉴스 티커 디자인 */
    .ticker-wrap {
        width: 100%; overflow: hidden; background-color: #263238; color: #ffffff;
        padding: 12px 0; margin-bottom: 20px; border-radius: 8px; white-space: nowrap;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-family: 'Pretendard', sans-serif;
    }
    
    /* [핵심 수정] 속도를 200초로 설정 (아주 천천히) */
    .ticker { display: inline-block; animation: ticker 200s linear infinite; } 
    
    .ticker-wrap:hover .ticker { animation-play-state: paused; } /* 마우스 올리면 멈춤 */
    
    @keyframes ticker {
        0% { transform: translate3d(100%, 0, 0); }
        100% { transform: translate3d(-100%, 0, 0); }
    }
    
    .ticker-item { 
        display: inline-block; padding: 0 4rem; font-size: 1.1rem; /* 간격도 넓힘 */
        border-right: 1px solid #546e7a;
    }
    
    .ticker-item a {
        color: #ffffff !important;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .ticker-item a:hover {
        color: #80cbc4 !important;
        text-decoration: underline;
    }
    
    .risk-badge { background-color: #ffebee; color: #c62828; padding: 3px 8px; border-radius: 5px; font-size: 0.8em; font-weight: bold; }
    .opinion-badge { padding: 4px 10px; border-radius: 6px; font-weight: bold; color: white; font-size: 0.9em; }
    .stButton button { width: 100%; border-radius: 8px; height: 45px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 강력한 뉴스 엔진 (MK & Google RSS)
# -----------------------------------------------------------------------------
def make_ticker_html(news_data):
    content = ""
    for news in news_data:
        if news.get('link'):
            content += f'<div class="ticker-item"><a href="{news["link"]}" target="_blank">📰 {news["title"]}</a></div>'
        else:
            content += f'<div class="ticker-item">{news["title"]}</div>'
    return f'<div class="ticker-wrap"><div class="ticker">{content}</div></div>'

@st.cache_data(ttl=300) 
def fetch_fast_news():
    news_data = []
    # 1. MK RSS
    try:
        url = "https://www.mk.co.kr/rss/30000001/" 
        resp = requests.get(url, timeout=1.5)
        root = ET.fromstring(resp.content)
        for item in root.findall('./channel/item')[:15]:
            title = item.find('title').text
            link = item.find('link').text
            news_data.append({'title': title, 'link': link})
        if news_data: return news_data
    except: pass

    # 2. Google RSS
    try:
        url = "https://news.google.com/rss/topics/CAAqIggKIhxDQkFTRHdvSkwyMHZNR2RtY0hNREVnSmxiaWdBUAE?hl=ko&gl=KR&ceid=KR%3Ako"
        resp = requests.get(url, timeout=1.5)
        root = ET.fromstring(resp.content)
        for item in root.findall('./channel/item')[:15]:
            title = item.find('title').text
            link = item.find('link').text
            clean_title = title.split(' - ')[0]
            news_data.append({'title': clean_title, 'link': link})
        if news_data: return news_data
    except: pass
    
    return []

DEFAULT_NEWS = [
    {'title': "⏳ 실시간 시장 뉴스를 불러오고 있습니다...", 'link': ''},
    {'title': "💡 투자 팁: 공포에 사서 환희에 팔아라", 'link': ''},
    {'title': "⚠️ 분할 매수는 리스크를 줄이는 최고의 습관입니다", 'link': ''}
]

# -----------------------------------------------------------------------------
# 3. 데이터 엔진 & 포트폴리오
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

def get_fallback_stocks():
    data = {'Name': ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '삼성바이오로직스', '현대차', '기아', '셀트리온', 'POSCO홀딩스', 'NAVER', '카카오', '삼성물산', 'KB금융', '신한지주', 'LG전자', '한화오션', '두산에너빌리티', '에코프로비엠', '에코프로', 'HLB', '알테오젠'],
            'Code': ['005930', '000660', '373220', '207940', '005380', '000270', '068270', '005490', '035420', '035720', '028260', '105560', '055550', '066570', '042660', '034020', '247540', '086520', '028300', '196170']}
    return pd.DataFrame(data)

@st.cache_data
def get_stock_listing():
    try:
        df = fdr.StockListing("KOSPI")
        df = pd.concat([df, fdr.StockListing("KOSDAQ")])
        df = df[~df['Name'].str.contains('우|스팩|ETN|ETF|홀딩스')]
        return df[['Code', 'Name', 'Market']]
    except: return get_fallback_stocks()

all_stocks = get_stock_listing()

def get_stock_data_hybrid(code, days=400):
    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    try:
        df = fdr.DataReader(code, start_date)
        if df is not None and not df.empty: return df
    except: pass
    try:
        for suffix in ['.KS', '.KQ']:
            df = yf.download(f"{code}{suffix}", start=start_date, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                return df
    except: pass
    return None

def check_risk_status(code):
    try:
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=2)
        if "alt=\"관리종목\"" in response.text: return ["관리"]
        return []
    except: return []

def get_sentiment(code):
    try:
        url = f"https://finance.naver.com/item/news_news.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=2)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.select_one('.title')
        return (10, title.get_text().strip()) if title else (0, "-")
    except: return 0, "-"

def get_opinion(score):
    if score >= 75: return "🔥 강력 매수", "#e53935"
    elif score >= 60: return "📈 매수 추천", "#fb8c00"
    elif score >= 40: return "✋ 관망 유지", "#757575"
    else: return "📉 매도 우위", "#1e88e5"

def analyze_logic(code, name):
    df = get_stock_data_hybrid(code)
    if df is None or len(df) < 60: return None
    try:
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['Target_Price'] = df['Open'] + ((df['High'] - df['Low']).shift(1) * 0.5)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        cols = ['Close', 'RSI', 'MA5', 'MA20', 'MA60']
        df_clean = df.dropna(subset=cols).copy()
        
        X = df_clean[cols].iloc[:-1]
        y = df_clean['Target'].iloc[:-1]
        last_row = df_clean[cols].iloc[[-1]]
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)
        model.fit(X, y)
        score = model.predict_proba(last_row)[0][1] * 100
        
        last_close = float(df['Close'].iloc[-1])
        volatility = (df['High'] - df['Low']).rolling(5).mean().iloc[-1]
        target_price = last_close + (volatility * 2.0)
        stop_loss = last_close - (volatility * 1.5)
        sent_score, headline = get_sentiment(code)
        
        final_score = float(round((score * 0.7) + (sent_score + 50) * 0.3, 1))
        opinion, css = get_opinion(final_score)
        
        risk_labels = check_risk_status(code)
        
        return {
            'code': code, 'name': name, 'price': int(last_close),
            'final_score': final_score, 'target_price': int(target_price),
            'stop_loss': int(stop_loss), 'headline': headline,
            'last_data': df, 'risks': risk_labels,
            'opinion': opinion, 'opinion_css': css
        }
    except: return None

def create_chart(item):
    df_chart = item['last_data'][-60:]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'))
    fig.add_hline(y=item['target_price'], line_dash="dash", line_color="#4CAF50", annotation_text="목표가")
    fig.add_hline(y=item['stop_loss'], line_dash="dash", line_color="#F44336", annotation_text="손절가")
    fig.update_layout(height=300, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig

@st.cache_data(ttl=600)
def get_market_ranking(type='rise'):
    try:
        base_url = "https://finance.naver.com/sise/"
        if type == 'rise': url = f"{base_url}sise_rise.naver"
        elif type == 'fall': url = f"{base_url}sise_fall.naver"
        elif type == 'admin': url = f"{base_url}sise_adm.naver"
        
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        dfs = pd.read_html(resp.text, encoding='euc-kr')
        df = dfs[1].dropna(subset=['종목명'])
        return df[['종목명', '현재가', '등락률']].head(10)
    except: return pd.DataFrame()

# -----------------------------------------------------------------------------
# 4. UI 구성
# -----------------------------------------------------------------------------
st.title("💎 Quant Logic : Pro Station")
st.caption(f"시스템 상태: 정상 | 기준: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

# [뉴스 엔진] 1. 빈 공간 생성 -> 2. 기본 멘트 표시 -> 3. 진짜 뉴스 로딩
news_placeholder = st.empty()
news_placeholder.markdown(make_ticker_html(DEFAULT_NEWS), unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💼 내 포트폴리오", "🌏 시장 현황", "🚀 AI 추천"])

with tab1:
    with st.expander("➕ 종목 추가/관리", expanded=not bool(st.session_state.my_portfolio)):
        col1, col2 = st.columns([3, 1])
        with col1:
            sel_stock = st.selectbox("종목 검색", ["선택..."] + all_stocks['Name'].tolist(), label_visibility="collapsed")
        with col2:
            if st.button("추가", use_container_width=True):
                if sel_stock != "선택..." and sel_stock not in st.session_state.my_portfolio:
                    st.session_state.my_portfolio.append(sel_stock)
                    save_portfolio(st.session_state.my_portfolio)
                    st.rerun()
        if st.session_state.my_portfolio:
            cols = st.columns(4)
            for i, stock in enumerate(st.session_state.my_portfolio):
                if cols[i % 4].button(f"🗑️ {stock}", key=f"del_{stock}"):
                    st.session_state.my_portfolio.remove(stock)
                    save_portfolio(st.session_state.my_portfolio)
                    st.rerun()

    if st.session_state.my_portfolio:
        if st.button("🔄 진단 시작", type="primary", use_container_width=True):
            st.session_state['analysis_cache'] = []
            with st.status("AI 분석 중...", expanded=True) as status:
                res_list = []
                for s_name in st.session_state.my_portfolio:
                    try:
                        row = all_stocks[all_stocks['Name'] == s_name]
                        if row.empty: continue
                        code = str(row['Code'].values[0])
                        r = analyze_logic(code, s_name)
                        if r: res_list.append(r)
                    except: continue
                st.session_state['analysis_cache'] = res_list
                status.update(label="완료!", state="complete", expanded=False)

    if st.session_state['analysis_cache']:
        for i, item in enumerate(st.session_state['analysis_cache']):
            with st.container():
                c_head, c_score = st.columns([2, 1])
                risk_html = "".join([f"<span class='risk-badge'>⚠️{r}</span> " for r in item['risks']])
                c_head.markdown(f"#### {item['name']} {risk_html}", unsafe_allow_html=True)
                c_score.markdown(f"<div style='text-align:right;'><span class='opinion-badge' style='background-color:{item['opinion_css']}'>{item['opinion']} {item['final_score']:.1f}점</span></div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("현재가", f"{item['price']:,}원")
                c2.metric("목표가", f"{item['target_price']:,}원")
                c3.metric("손절가", f"{item['stop_loss']:,}원")
                with st.expander("차트 및 뉴스 확인"):
                    st.plotly_chart(create_chart(item), use_container_width=True, key=f"chart_{i}")
                    st.info(f"📰 {item['headline']}")
            st.divider()

with tab2:
    st.header("📊 오늘의 시장 변동성")
    col_rise, col_fall, col_admin = st.columns(3)
    with col_rise:
        st.subheader("🚀 급등")
        st.dataframe(get_market_ranking('rise'), use_container_width=True, hide_index=True)
    with col_fall:
        st.subheader("📉 급락")
        st.dataframe(get_market_ranking('fall'), use_container_width=True, hide_index=True)
    with col_admin:
        st.subheader("⚠️ 관리종목")
        st.dataframe(get_market_ranking('admin'), use_container_width=True, hide_index=True)

with tab3:
    col_l, col_r = st.columns([1, 2])
    with col_l: scan_cnt = st.slider("분석 수", 10, 50, 10)
    with col_r:
        if st.button("🚀 유망 종목 스캔", type="primary", use_container_width=True):
            bar = st.progress(0)
            target_df = all_stocks.head(scan_cnt)
            m_res = []
            for idx, (i, row) in enumerate(target_df.iterrows()):
                bar.progress(min((idx+1)/len(target_df), 1.0))
                if check_risk_status(row['Code']): continue
                r = analyze_logic(row['Code'], row['Name'])
                if r: m_res.append(r)
            bar.empty()
            st.session_state['market_results'] = sorted(m_res, key=lambda x: x['final_score'], reverse=True)
            st.rerun()

    if st.session_state['market_results']:
        best = st.session_state['market_results'][0]
        st.success(f"🏆 Top Pick: **{best['name']}** ({best['opinion']} {best['final_score']:.1f}점)")
        for i, item in enumerate(st.session_state['market_results']):
            with st.container():
                c_head, c_score = st.columns([2, 1])
                c_head.markdown(f"**{i+1}위. {item['name']}**")
                c_score.markdown(f"<div style='text-align:right;'><span class='opinion-badge' style='background-color:{item['opinion_css']}'>{item['opinion']} {item['final_score']:.1f}점</span></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.metric("현재가", f"{item['price']:,}")
                c2.metric("목표가", f"{item['target_price']:,}")
                with st.expander("상세 보기"):
                    st.plotly_chart(create_chart(item), use_container_width=True, key=f"m_chart_{i}")
            st.divider()

# [지연 로딩] 3. 화면이 다 그려진 후 진짜 뉴스를 가져와서 바꿔치기
real_news = fetch_fast_news()
if real_news:
    news_placeholder.markdown(make_ticker_html(real_news), unsafe_allow_html=True)
