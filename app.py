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

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 디자인
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Logic V11", page_icon="💎", layout="wide")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    
    /* 카드 스타일 */
    div.stMetric {
        background-color: #f8f9fa; border: 1px solid #dee2e6;
        padding: 10px; border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        height: auto; padding: 8px 16px; border-radius: 20px;
        background-color: #f1f3f5; border: none; margin-bottom: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important; color: white !important;
    }
    
    /* 뱃지 스타일 */
    .risk-badge { background-color: #ffebee; color: #c62828; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; font-weight: bold; border: 1px solid #ef9a9a; }
    .sim-badge { background-color: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; font-weight: bold; border: 1px solid #90caf9; }
    
    /* 투자 의견 뱃지 (핵심 신규 기능) */
    .opinion-buy-strong { background-color: #ffcdd2; color: #b71c1c; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .opinion-buy { background-color: #ffcc80; color: #e65100; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .opinion-hold { background-color: #cfd8dc; color: #455a64; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    .opinion-sell { background-color: #bbdefb; color: #0d47a1; padding: 4px 8px; border-radius: 6px; font-weight: bold; }

    .stButton button { width: 100%; border-radius: 8px; height: 45px; }
</style>
""", unsafe_allow_html=True)

st.title("💎 Quant Logic (Actionable)")
st.caption(f"Update: V11.0 (Investment Opinion) | Time: {datetime.datetime.now().strftime('%m-%d %H:%M')}")

# -----------------------------------------------------------------------------
# 2. 데이터 엔진 & 비상용 리스트
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

# 서버 차단 대비: 주요 종목 100개 직접 내장
def get_fallback_stocks():
    data = {
        'Name': [
            '삼성전자', 'SK하이닉스', 'LG에너지솔루션', '삼성바이오로직스', '현대차', '기아', '셀트리온', 'POSCO홀딩스', 'NAVER', '삼성SDI',
            'LG화학', '카카오', '삼성물산', '현대모비스', 'KB금융', '포스코퓨처엠', '신한지주', 'LG전자', '삼성생명', 'SK이노베이션',
            'LG', '한국전력', '삼성화재', '하나금융지주', 'KT&G', 'HD현대중공업', 'SK', '두산에너빌리티', '크래프톤', 'HMM',
            '고려아연', '메리츠금융지주', '우리금융지주', '삼성에스디에스', '한화오션', 'SK텔레콤', 'KT', '대한항공', '기업은행', 'S-Oil',
            'HD한국조선해양', '카카오뱅크', 'LG생활건강', '아모레퍼시픽', 'SK바이오사이언스', '엔씨소프트', '한화에어로스페이스', 'LG디스플레이', 'CJ제일제당', '강원랜드',
            '에코프로비엠', '에코프로', 'HLB', '알테오젠', '펄어비스', '카카오게임즈', '셀트리온제약', 'JYP Ent.', '에스엠', '스튜디오드래곤',
            '엘앤에프', '위메이드', '천보', '리노공업', '솔브레인', '동진쎄미켐', '원익IPS', '와이지엔터테인먼트', '하이브', '현대오토에버',
            '현대미포조선', '한화시스템', '한국항공우주', '한미반도체', '현대로템', '금양', '코스모신소재', '이수페타시스', '한미약품', '유한양행'
        ],
        'Code': [
            '005930', '000660', '373220', '207940', '005380', '000270', '068270', '005490', '035420', '006400',
            '051910', '035720', '028260', '012330', '105560', '003670', '055550', '066570', '032830', '096770',
            '003550', '015760', '000810', '086790', '033780', '329180', '034730', '034020', '259960', '011200',
            '010130', '138040', '316140', '018260', '042660', '017670', '030200', '003490', '024110', '010950',
            '009540', '323410', '051900', '090430', '302440', '036570', '012450', '034220', '097950', '035250',
            '247540', '086520', '028300', '196170', '263750', '293490', '068760', '035900', '041510', '253450',
            '066970', '112040', '278280', '058470', '357780', '005290', '240810', '122870', '352820', '307950',
            '010620', '272210', '047810', '042700', '064350', '001570', '005070', '007660', '128940', '000100'
        ]
    }
    return pd.DataFrame(data)

@st.cache_data
def get_stock_listing(market):
    try:
        df = fdr.StockListing(market)
        df = df[~df['Name'].str.contains('우|스팩|ETN|ETF|홀딩스')]
        return df[['Code', 'Name', 'Market']]
    except:
        fallback = get_fallback_stocks()
        fallback['Market'] = market
        return fallback

# -----------------------------------------------------------------------------
# 3. 로직 함수 (하이브리드 & 투자의견 산출)
# -----------------------------------------------------------------------------
def generate_mock_data(days=400):
    dates = pd.date_range(end=datetime.datetime.now(), periods=days)
    np.random.seed(int(datetime.datetime.now().timestamp()))
    price = 50000 + np.cumsum(np.random.randn(days) * 1000)
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = price + np.random.randn(days) * 500
    df['High'] = df[['Open', 'Close']].max(axis=1) + abs(np.random.randn(days) * 500)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - abs(np.random.randn(days) * 500)
    df['Volume'] = np.abs(np.random.randn(days) * 100000) + 10000
    df[df < 100] = 100
    return df

def get_stock_data_hybrid(code, days=400):
    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    try:
        df = fdr.DataReader(code, start_date)
        if df is not None and not df.empty: return df, False
    except: pass
    
    try:
        for suffix in ['.KS', '.KQ']:
            df = yf.download(f"{code}{suffix}", start=start_date, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                return df, False
    except: pass
    return generate_mock_data(days), True

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
        return (10, title.get_text().strip()) if title else (0, "뉴스 없음")
    except: return 0, "뉴스 연결 실패"

# [신규] 투자 의견 산출 함수
def get_investment_opinion(score):
    if score >= 70:
        return "🔥 강력 매수 (Strong Buy)", "opinion-buy-strong"
    elif score >= 60:
        return "📈 매수 (Buy)", "opinion-buy"
    elif score >= 40:
        return "✋ 관망/보유 (Hold)", "opinion-hold"
    else:
        return "📉 매도 (Sell)", "opinion-sell"

def analyze_logic(code, name):
    risk_labels = check_risk_status(code)
    df, is_sim = get_stock_data_hybrid(code)
    
    if df is None: return None
    if len(df) < 60: return None

    try:
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        df['Target_Price'] = df['Open'] + ((df['High'] - df['Low']).shift(1) * 0.5)
        df['VBO_Signal'] = np.where(df['Close'] > df['Target_Price'], 1, 0)
        df['Noise'] = 1 - (np.abs(df['Close']-df['Open']) / (df['High']-df['Low'] + 0.001))
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        cols = ['Close', 'RSI', 'VBO_Signal', 'Noise', 'MA5', 'MA20', 'MA60']
        df_clean = df.dropna(subset=cols).copy()
        
        if len(df_clean) < 10: return None
        
        X = df_clean[cols].iloc[:-1]
        y = df_clean['Target'].iloc[:-1]
        last_row = df_clean[cols].iloc[[-1]]
        
        model = xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)
        model.fit(X, y)
        score = model.predict_proba(last_row)[0][1] * 100
        
        last_close = float(df['Close'].iloc[-1])
        volatility = (df['High'] - df['Low']).rolling(5).mean().iloc[-1]
        target_price = last_close + (volatility * 2.0)
        stop_loss = last_close - (volatility * 1.5)
        
        if is_sim: sent, head = 0, "🧪 서버 차단으로 인한 데모 데이터"
        else: sent, head = get_sentiment(code)
            
        final = round((score * 0.7) + (sent + 50) * 0.3, 1)
        
        # 투자 의견 도출
        opinion, opinion_css = get_investment_opinion(final)
        
        return {
            'code': code, 'name': name, 'price': int(last_close),
            'final_score': final, 'target_price': int(target_price),
            'stop_loss': int(stop_loss), 'headline': head, 'sentiment': sent,
            'last_data': df, 'risks': risk_labels, 'is_sim': is_sim,
            'opinion': opinion, 'opinion_css': opinion_css
        }
    except: return None

def create_chart(item):
    df_chart = item['last_data'][-60:]
    title_text = f"<b>{item['name']}</b> ({item['opinion'].split(' ')[0]})"
    if item['is_sim']: title_text += " (Demo)"
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'))
    fig.add_hline(y=item['target_price'], line_dash="dash", line_color="#00C853", annotation_text="Target")
    fig.add_hline(y=item['stop_loss'], line_dash="dash", line_color="#D50000", annotation_text="Cut")
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15)),
        height=300, xaxis_rangeslider_visible=False, 
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------------------------------------------------------
# 4. 메인 UI
# -----------------------------------------------------------------------------
all_stocks = get_stock_listing("KOSPI")

tab1, tab2 = st.tabs(["💼 내 포트폴리오", "🚀 시장 추천"])

with tab1:
    with st.expander("➕ 종목 추가 및 관리", expanded=not bool(st.session_state.my_portfolio)):
        col_sel, col_add = st.columns([3, 1])
        with col_sel:
            selected_stock = st.selectbox("종목 검색", ["선택..."] + all_stocks['Name'].tolist(), label_visibility="collapsed")
        with col_add:
            if st.button("추가"):
                if selected_stock != "선택..." and selected_stock not in st.session_state.my_portfolio:
                    st.session_state.my_portfolio.append(selected_stock)
                    save_portfolio(st.session_state.my_portfolio)
                    st.rerun()
        
        if st.session_state.my_portfolio:
            cols = st.columns(3)
            for i, stock in enumerate(st.session_state.my_portfolio):
                if cols[i % 3].button(f"🗑️ {stock}", key=f"del_{stock}"):
                    st.session_state.my_portfolio.remove(stock)
                    save_portfolio(st.session_state.my_portfolio)
                    st.rerun()

    if st.session_state.my_portfolio:
        if st.button("🔄 내 종목 진단 실행", type="primary"):
            st.session_state['analysis_cache'] = []
            with st.status("AI 분석 중...", expanded=True) as status:
                res_list = []
                for s_name in st.session_state.my_portfolio:
                    try:
                        row = all_stocks[all_stocks['Name'] == s_name]
                        if row.empty:
                            status.write(f"⚠️ {s_name}: 코드 정보 없음")
                            continue
                        code = str(row['Code'].values[0])
                        
                        r = analyze_logic(code, s_name)
                        if r: 
                            res_list.append(r)
                            status.write(f"✅ {s_name} 완료")
                    except: continue
                st.session_state['analysis_cache'] = res_list
                status.update(label="진단 완료!", state="complete", expanded=False)
    else:
        st.info("종목을 추가해주세요.")

    if st.session_state['analysis_cache']:
        for item in st.session_state['analysis_cache']:
            with st.container():
                c_head, c_score = st.columns([2.5, 1])
                badges = ""
                if item['risks']: badges += f" <span class='risk-badge'>⚠️{item['risks'][0]}</span>"
                if item['is_sim']: badges += f" <span class='sim-badge'>🧪데모</span>"
                
                c_head.markdown(f"**{item['name']}** {badges}", unsafe_allow_html=True)
                
                # [수정] 투자 의견 표시
                c_score.markdown(f"<span class='{item['opinion_css']}'>{item['opinion'].split(' ')[0]} {item['final_score']:.1f}</span>", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("현재가", f"{item['price']:,}")
                c2.metric("목표가", f"{item['target_price']:,}")
                
                # 투자 의견 텍스트 (확실하게 보여주기)
                st.info(f"💡 **투자 판단**: {item['opinion']}")

                with st.expander("차트 보기"):
                    st.plotly_chart(create_chart(item), use_container_width=True, key=f"chart_{item['code']}")
                st.divider()

with tab2:
    if st.button("🚀 Top Picks 스캔", type="primary"):
        st.info("시장 스캔 중...")
        bar = st.progress(0)
        target_df = all_stocks.head(15) 
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
        st.success(f"🏆 1위: **{best['name']}** - {best['opinion']}")
        st.divider()
        for i, item in enumerate(st.session_state['market_results']):
            with st.container():
                c_head, c_score = st.columns([2.5, 1])
                badges = ""
                if item['is_sim']: badges += f" <span class='sim-badge'>🧪데모</span>"
                
                c_head.markdown(f"**{item['name']}** {badges}", unsafe_allow_html=True)
                
                # [수정] 투자 의견 표시
                c_score.markdown(f"<span class='{item['opinion_css']}'>{item['opinion'].split(' ')[0]} {item['final_score']:.1f}</span>", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("현재가", f"{item['price']:,}")
                c2.metric("목표가", f"{item['target_price']:,}")
                
                # 투자 의견 텍스트
                st.caption(f"💡 {item['opinion']}")
                
                with st.expander("상세 보기"):
                    st.plotly_chart(create_chart(item), use_container_width=True, key=f"m_chart_{i}")
                st.divider()
