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
import yfinance as yf # [신규] 야후 파이낸스 추가

# -----------------------------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quant Logic Hybrid", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main { background-color: #ffffff; }
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
    .risk-badge {
        background-color: #ffebee; color: #c62828; padding: 2px 6px;
        border-radius: 4px; font-size: 0.75em; font-weight: bold; border: 1px solid #ef9a9a;
    }
    .stButton button { width: 100%; border-radius: 8px; height: 45px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Quant Logic (Hybrid)")
st.caption(f"서버 상태: {'정상' if True else '우회 모드'} | 기준: {datetime.datetime.now().strftime('%m-%d %H:%M')}")

# -----------------------------------------------------------------------------
# 2. 데이터 엔진 (핵심: 네이버 실패 시 야후로 우회)
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

# [안전장치] 종목 리스트 로딩 실패 시 사용할 비상용 리스트
def get_fallback_stocks():
    return pd.DataFrame({
        'Code': ['005930', '000660', '373220', '207940', '005380', '000270', '068270', '005490', '035420', '000810'],
        'Name': ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '삼성바이오로직스', '현대차', '기아', '셀트리온', 'POSCO홀딩스', 'NAVER', '삼성화재'],
        'Market': ['KOSPI']*10
    })

@st.cache_data
def get_stock_listing(market):
    try:
        # 1차 시도: 네이버 금융 (FDR)
        df = fdr.StockListing(market)
        df = df[~df['Name'].str.contains('우|스팩|ETN|ETF|홀딩스')]
        return df
    except Exception as e:
        # 2차 시도: 실패 시 비상용 리스트 반환 (앱 멈춤 방지)
        print(f"FDR Listing Fail: {e}")
        return get_fallback_stocks()

def get_stock_data_hybrid(code, days=400):
    """
    하이브리드 데이터 수집: FDR(네이버) 실패 시 Yfinance(야후) 사용
    """
    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    # 1. FDR 시도
    try:
        df = fdr.DataReader(code, start_date)
        if df is not None and not df.empty:
            return df
    except:
        pass # 조용히 다음 단계로 넘어감
        
    # 2. Yfinance 시도 (FDR 실패 시)
    try:
        # 한국 종목 코드는 뒤에 .KS(코스피) 또는 .KQ(코스닥) 붙여야 함
        # 정확히 모르니 둘 다 시도
        yf_code = f"{code}.KS"
        df = yf.download(yf_code, start=start_date, progress=False)
        
        if df.empty:
            yf_code = f"{code}.KQ"
            df = yf.download(yf_code, start=start_date, progress=False)
            
        if not df.empty:
            # Yfinance는 컬럼이 멀티인덱스일 수 있어 정리 필요
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception as e:
        print(f"Yahoo Fail: {e}")
        
    return None

def check_risk_status(code):
    # 리스크 체크는 네이버 크롤링 필수라 실패하면 '알 수 없음' 처리
    try:
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3)
        html = response.text
        risks = []
        if "alt=\"관리종목\"" in html: risks.append("관리")
        if "alt=\"거래정지\"" in html: risks.append("정지")
        if "alt=\"투자경고\"" in html: risks.append("경고")
        return risks
    except:
        return [] # 실패 시 리스크 없음으로 간주 (앱 중단 방지)

def get_sentiment(code):
    try:
        url = f"https://finance.naver.com/item/news_news.naver?code={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3)
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
    except: return 0, "뉴스 수집 불가"

def analyze_logic(code, name):
    risk_labels = check_risk_status(code)
    
    # [변경] 하이브리드 함수 사용
    df = get_stock_data_hybrid(code)
    
    if df is None: return None
    
    # 데이터 부족 처리
    if len(df) < 60: return None

    try:
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        
        # RSI 계산 (ZeroDivision 방지)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001) # 0 나누기 방지
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Target_Price'] = df['Open'] + ((df['High'] - df['Low']).shift(1) * 0.5)
        df['VBO_Signal'] = np.where(df['Close'] > df['Target_Price'], 1, 0)
        
        denom = (df['High']-df['Low']).replace(0, 0.001)
        df['Noise'] = 1 - (np.abs(df['Close']-df['Open']) / denom)
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
        sent, head = get_sentiment(code)
        final = round((score * 0.7) + (sent + 50) * 0.3, 1)
        
        return {
            'code': code, 'name': name, 'price': int(last_close),
            'final_score': final, 'target_price': int(target_price),
            'stop_loss': int(stop_loss), 'headline': head, 'sentiment': sent,
            'last_data': df, 'risks': risk_labels
        }
    except Exception as e:
        print(f"Logic Error {name}: {e}")
        return None

def create_chart(item):
    df_chart = item['last_data'][-60:]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'))
    fig.add_hline(y=item['target_price'], line_dash="dash", line_color="#00C853", annotation_text="Target")
    fig.add_hline(y=item['stop_loss'], line_dash="dash", line_color="#D50000", annotation_text="Cut")
    fig.update_layout(
        title=dict(text=f"<b>{item['name']}</b>", font=dict(size=15)),
        height=300, xaxis_rangeslider_visible=False, 
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------------------------------------------------------
# 4. 메인 UI
# -----------------------------------------------------------------------------
all_stocks = get_stock_listing("KOSPI")

# 리스트 로딩 실패 시 비상 리스트 사용 확인
if len(all_stocks) < 15:
    st.toast("⚠️ 네이버 접속 불안정: 비상용 리스트를 사용합니다.", icon="🛡️")

tab1, tab2 = st.tabs(["💼 내 포트폴리오", "🚀 시장 추천"])

with tab1:
    with st.expander("➕ 종목 추가 및 관리", expanded=not bool(st.session_state.my_portfolio)):
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
            if st.session_state.my_portfolio:
                st.caption("보유 목록 (삭제하려면 클릭):")
                cols = st.columns(3)
                for i, stock in enumerate(st.session_state.my_portfolio):
                    if cols[i % 3].button(f"🗑️ {stock}", key=f"del_{stock}"):
                        st.session_state.my_portfolio.remove(stock)
                        save_portfolio(st.session_state.my_portfolio)
                        st.rerun()
        else:
            st.error("종목 리스트 로딩 실패")

    if st.session_state.my_portfolio:
        if st.button("🔄 내 종목 진단 실행", type="primary"):
            st.session_state['analysis_cache'] = []
            with st.status("AI 분석 중... (하이브리드 엔진 동작)", expanded=True) as status:
                res_list = []
                for s_name in st.session_state.my_portfolio:
                    try:
                        # 코드 찾기 (비상용 리스트일 경우 대비)
                        stock_row = all_stocks[all_stocks['Name'] == s_name]
                        if stock_row.empty:
                            status.write(f"⚠️ {s_name}: 코드 정보 없음")
                            continue
                        code = str(stock_row['Code'].values[0])
                        
                        r = analyze_logic(code, s_name)
                        if r: 
                            res_list.append(r)
                            status.write(f"✅ {s_name} 완료")
                        else:
                            status.write(f"❌ {s_name} 데이터 수집 실패")
                    except: continue
                st.session_state['analysis_cache'] = res_list
                status.update(label="진단 완료!", state="complete", expanded=False)

    if st.session_state['analysis_cache']:
        for item in st.session_state['analysis_cache']:
            with st.container():
                c_head, c_score = st.columns([2.5, 1])
                risk_tags = "".join([f" <span class='risk-badge'>⚠️{r}</span>" for r in item['risks']])
                c_head.markdown(f"**{item['name']}** {risk_tags}", unsafe_allow_html=True)
                
                score = item['final_score']
                color = "green" if score >= 70 else "orange" if score >= 50 else "red"
                c_score.markdown(f"<span style='color:{color}; font-size:1.2em; font-weight:bold'>{score:.1f}점</span>", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                c1.metric("현재가", f"{item['price']:,}")
                c2.metric("목표가", f"{item['target_price']:,}")
                
                with st.expander("차트 보기"):
                    st.plotly_chart(create_chart(item), use_container_width=True, key=f"chart_{item['code']}")
                st.divider()

with tab2:
    if st.button("🚀 Top Picks 스캔", type="primary"):
        st.info("시장 데이터 스캔 중...")
        bar = st.progress(0)
        # 리스트가 너무 많으면 시간이 오래 걸리니 상위 20개만
        target_df = all_stocks.head(20) 
        m_res = []
        total = len(target_df)
        
        for idx, (i, row) in enumerate(target_df.iterrows()):
            bar.progress(min((idx+1)/total, 1.0))
            # 시장 추천에서는 리스크 있는 종목 자동 패스
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
