from fastapi import FastAPI, Header, HTTPException
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import datetime
import uvicorn
from pyngrok import ngrok

app = FastAPI()

# ======================================================
# [중요] 나만의 비밀번호 설정 (복잡할수록 안전함)
# ======================================================
MY_SECRET_KEY = "quant-logic-password-2025" 

# 데이터 수집 함수 (기존과 동일)
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

@app.get("/")
def read_root():
    return {"status": "Secure Server is running"}

# [보안 강화] x_api_key 헤더 확인
@app.get("/stock/{code}")
def read_stock(code: str, x_api_key: str = Header(None)):
    # 1. 비밀번호 검사
    if x_api_key != MY_SECRET_KEY:
        print(f"🚨 침입 시도 감지! (잘못된 키: {x_api_key})")
        raise HTTPException(status_code=401, detail="누구세요? (Unauthorized)")
    
    # 2. 통과 시 데이터 제공
    print(f"✅ 인증 성공: 종목코드 {code} 요청")
    df = get_stock_data_hybrid(code)
    if df is None: return {"error": "Data not found"}
    
    df = df.reset_index()
    if 'Date' in df.columns: df['Date'] = df['Date'].astype(str)
    
    return df.to_dict(orient="records")

if __name__ == "__main__":
    port = 8000
    # ngrok 실행
    public_url = ngrok.connect(port).public_url
    print(f"\n========================================================")
    print(f"🔒 보안 서버가 실행되었습니다.")
    print(f"🌍 접속 주소: {public_url}")
    print(f"🔑 설정된 비밀번호: {MY_SECRET_KEY}")
    print(f"========================================================\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)