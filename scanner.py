"""CRYPTO DASHBOARD SCANNER — CoinSwitch universe via yfinance.
TFs: Monthly, Weekly, Daily, 4H, 30m, 5m. Outputs data.json for the dashboard.
Reads symbols.json + yf_coverage.json (same files as the Excel scanner)."""
import json, time, logging, warnings
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd, yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
IST = timezone(timedelta(hours=5, minutes=30))
CHUNK, OUT = 40, "data.json"

cs = json.load(open("symbols.json"))["coinswitchx"]
cov = json.load(open("yf_coverage.json"))
inr, usd = set(cov["covered_inr"]), set(cov["covered_usd"])
items = []
for p in cs:
    b = p.split("/")[0]
    if b in inr: items.append((p, b, f"{b}-INR", "INR"))
    elif b in usd: items.append((p, b, f"{b}-USD", "USD"))
print(f"Universe: {len(items)} pairs")

def r2(v, nd=6):
    try: f = float(v)
    except (TypeError, ValueError): return None
    return None if (pd.isna(f) or np.isinf(f)) else round(f, nd)

def pat(o, h, l, c):
    body, rng = abs(c-o), h-l
    if rng == 0 or pd.isna(rng): return "Flat"
    b, u, lo = body/rng, (h-max(o,c))/rng, (min(o,c)-l)/rng
    if b < 0.10: return "Doji"
    if b > 0.80: return "Bullish Marubozu" if c > o else "Bearish Marubozu"
    if lo > 0.50 and b < 0.30 and u < 0.20: return "Hammer"
    if u > 0.50 and b < 0.30 and lo < 0.20: return "Shooting Star"
    return "Bullish" if c > o else "Bearish"

def cblock(bar):
    if bar is None: return None
    o,h,l,c = (float(bar[k]) for k in ("Open","High","Low","Close"))
    return {"o":r2(o),"h":r2(h),"l":r2(l),"c":r2(c),"pat":pat(o,h,l,c),
            "f50":r2(l+.5*(h-l)),"f618":r2(l+.618*(h-l))}

def rsi(cl, n=14):
    cl = pd.Series(cl).dropna()
    if len(cl) < n+1: return None
    d = cl.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    L = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return r2((100-(100/(1+g/L))).iloc[-1], 2)

def ema(cl, n):
    cl = pd.Series(cl).dropna()
    return None if len(cl) < n else r2(cl.ewm(span=n, adjust=False).mean().iloc[-1])

def eblk(cl): return {"e9":ema(cl,9),"e21":ema(cl,21),"e50":ema(cl,50),"e200":ema(cl,200)}

def rect(cl, mn_bars=4):
    cl = pd.Series(cl).dropna()
    if len(cl) < mn_bars: return None, None, None
    mx, mn = float(cl.max()), float(cl.min())
    if mn <= 0: return None, None, None
    return r2(mx), r2(mn), r2((mx-mn)/mn*100, 2)

def vwap20(df):
    d = df.tail(20).dropna(subset=["High","Low","Close","Volume"])
    if d.empty or d["Volume"].sum() == 0: return None
    tp = (d["High"]+d["Low"]+d["Close"])/3
    return r2((tp*d["Volume"]).sum()/d["Volume"].sum())

def split_live(df, mins):
    if df.empty: return df, None
    last = df.index[-1]
    last = last.tz_localize(IST) if last.tzinfo is None else last.tz_convert(IST)
    return (df.iloc[:-1], df.iloc[-1]) if last + timedelta(minutes=mins) > datetime.now(IST) else (df, None)

def hhmm(ts): 
    t = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
    return (t + timedelta(minutes=5)).strftime("%H:%M")

def cross_times(t5, up, dn):
    out = {k: None for k in list(up)+list(dn)}
    if t5 is None or t5.empty: return out
    for ts, bar in t5.iterrows():
        c = bar.get("Close")
        if c is None or pd.isna(c): continue
        for k, lv in up.items():
            if out[k] is None and lv is not None and c > lv: out[k] = hhmm(ts)
        for k, lv in dn.items():
            if out[k] is None and lv is not None and c < lv: out[k] = hhmm(ts)
    return out

AGG = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}

def batch(tks, **kw):
    try: return yf.download(tks, group_by="ticker", threads=True, progress=False, auto_adjust=False, **kw)
    except Exception: return pd.DataFrame()

def pick(data, t):
    try:
        df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
        return df.dropna(how="all")
    except Exception: return pd.DataFrame()

def process(pair, base, quote, d1, h1, m30, m5, w1, mo):
    d1 = d1.dropna(subset=["Close"])
    if len(d1) < 30: return None
    ltp = float(d1["Close"].iloc[-1])
    h4 = pd.DataFrame()
    if not h1.empty and "Close" in h1: h4 = h1.dropna(subset=["Close"]).resample("4h").agg(AGG).dropna(subset=["Close"])
    m30 = m30.dropna(subset=["Close"]) if not m30.empty and "Close" in m30 else pd.DataFrame()
    m5 = m5.dropna(subset=["Close"]) if not m5.empty and "Close" in m5 else pd.DataFrame()
    w1 = w1.dropna(subset=["Close"]) if not w1.empty and "Close" in w1 else pd.DataFrame()
    mo = mo.dropna(subset=["Close"]) if not mo.empty and "Close" in mo else pd.DataFrame()
    # crypto trades 24/7 -> last bar of every frame is live
    dC, wC, moC = d1.iloc[:-1], (w1.iloc[:-1] if len(w1) else w1), (mo.iloc[:-1] if len(mo) else mo)
    h4C,_ = split_live(h4,240); m30C,_ = split_live(m30,30); m5C,_ = split_live(m5,5)
    mMax,mMin,mW = rect(moC["Close"].tail(23) if len(moC) else [], 6)
    wMax,wMin,wW = rect(wC["Close"].tail(21) if len(wC) else [], 6)
    dMax,dMin,dW = rect(dC["Close"].tail(30), 10)
    h4Max,h4Min,h4W = rect(h4C["Close"].tail(13) if len(h4C) else [], 6)
    m30Max,m30Min,m30W = rect(m30C["Close"].tail(8) if len(m30C) else [], 4)
    m5Max,m5Min,m5W = rect(m5C["Close"].tail(5) if len(m5C) else [], 4)
    pm = cblock(moC.iloc[-1]) if len(moC) else None
    pw = cblock(wC.iloc[-1]) if len(wC) else None
    pdB = cblock(dC.iloc[-1]) if len(dC) else None
    p4 = cblock(h4C.iloc[-1]) if len(h4C) else None
    p30 = cblock(m30C.iloc[-1]) if len(m30C) else None
    p5 = cblock(m5C.iloc[-1]) if len(m5C) else None
    t5 = pd.DataFrame()
    if not m5.empty:
        idx = m5.index
        ii = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
        t5 = m5[ii.date == max(ii.date)]
    bt = {}
    for tf, mx, mn, cb in (("m",mMax,mMin,pm),("w",wMax,wMin,pw),("d",dMax,dMin,pdB),
                           ("h4",h4Max,h4Min,p4),("m30",m30Max,m30Min,p30),("m5",m5Max,m5Min,p5)):
        up, dn = {"u":mx}, {"l":mn}
        if cb: up["dh"], up["dc"], dn["dl"] = cb["h"], cb["c"], cb["l"]
        res = cross_times(t5, up, dn)
        if any(res.values()): bt[tf] = {k:v for k,v in res.items() if v}
    vol = d1["Volume"].dropna()
    return {"s":pair.replace("/","-"), "base":base, "q":quote, "ltp":r2(ltp),
        "mMax":mMax,"mMin":mMin,"mW":mW,"wMax":wMax,"wMin":wMin,"wW":wW,
        "dMax":dMax,"dMin":dMin,"dW":dW,"h4Max":h4Max,"h4Min":h4Min,"h4W":h4W,
        "m30Max":m30Max,"m30Min":m30Min,"m30W":m30W,"m5Max":m5Max,"m5Min":m5Min,"m5W":m5W,
        "pm":pm,"pw":pw,"pd":pdB,"p4":p4,"p30":p30,"p5":p5,"bt":bt or None,
        "rsiM":rsi(moC["Close"] if len(moC) else []),"rsiW":rsi(wC["Close"] if len(wC) else []),
        "rsiD":rsi(dC["Close"]),"rsi4":rsi(h4C["Close"] if len(h4C) else []),
        "rsi30":rsi(m30C["Close"] if len(m30C) else []),"rsi5":rsi(m5C["Close"] if len(m5C) else []),
        "ema":{"m":eblk(moC["Close"] if len(moC) else []),"w":eblk(wC["Close"] if len(wC) else []),
               "d":eblk(dC["Close"]),"h4":eblk(h4C["Close"] if len(h4C) else []),
               "m30":eblk(m30C["Close"] if len(m30C) else []),"m5":eblk(m5C["Close"] if len(m5C) else [])},
        "vwap":vwap20(d1), "v7":r2(vol.iloc[:-1].tail(7).mean() if len(vol)>1 else None, 2)}

rows, failed = [], 0
tickers = [it[2] for it in items]
for i in range(0, len(items), CHUNK):
    ck = items[i:i+CHUNK]; tks = [c[2] for c in ck]
    d1a = batch(tks, period="1y", interval="1d")
    w1a = batch(tks, period="2y", interval="1wk")
    moa = batch(tks, period="5y", interval="1mo")
    h1a = batch(tks, period="60d", interval="1h")
    m30a = batch(tks, period="1mo", interval="30m")
    m5a = batch(tks, period="5d", interval="5m")
    for pair, base, tk, q in ck:
        try:
            r = process(pair, base, q, pick(d1a,tk), pick(h1a,tk), pick(m30a,tk), pick(m5a,tk), pick(w1a,tk), pick(moa,tk))
            rows.append(r) if r else None
            if not r: failed += 1
        except Exception: failed += 1
    print(f"{min(i+CHUNK,len(items))}/{len(items)} | ok={len(rows)} fail={failed}")
    time.sleep(1.0)

json.dump({"updated": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
           "updatedUtc": datetime.now(timezone.utc).isoformat(),
           "count": len(rows), "rows": rows}, open(OUT,"w"), separators=(",",":"))
print(f"Saved {OUT}: {len(rows)} rows, {failed} failed")
