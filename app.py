from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import ta as ta_lib
import math, json, time, os
import urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

app = Flask(__name__)

WATCHLIST = [
    # US Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    'ORCL', 'CRM', 'AMD', 'ADBE', 'NOW', 'INTC', 'AVGO', 'QCOM',
    'AMAT', 'MU', 'CRWD', 'PANW', 'NET', 'DDOG', 'SNOW', 'ACN', 'PLTR', 'UBER', 'SHOP',
    # US Finance
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'V', 'MA', 'BLK', 'C', 'PYPL', 'COIN',
    # US Consumer & Retail
    'WMT', 'COST', 'HD', 'MCD', 'SBUX', 'NKE', 'DIS', 'NFLX',
    # US Energy & Industrial
    'XOM', 'CVX', 'BA', 'CAT', 'HON', 'GE', 'NEE', 'LIN',
    # US Pharma & Biotech & Medtech
    'LLY', 'JNJ', 'PFE', 'ABBV', 'MRK', 'BMY', 'MRNA', 'GILD', 'REGN', 'VRTX', 'ISRG', 'MDT',
    # Swiss
    'NESN.SW', 'ROG.SW', 'NOVN.SW', 'ABBN.SW', 'UBS.SW', 'ZURN.SW', 'LONN.SW', 'SIKA.SW',
    # German
    'MB.DE', 'BMW.DE', 'SAP.DE', 'SIE.DE', 'VOW3.DE', 'ALV.DE', 'DTE.DE', 'ADS.DE', 'BAYN.DE', 'DBK.DE',
    # French
    'MC.PA', 'OR.PA', 'TTE.PA', 'LVMH.PA', 'SAN.PA', 'AIR.PA', 'BNP.PA', 'SU.PA', 'AI.PA', 'KER.PA',
    # Other EU
    'ASML.AS', 'NOVO-B.CO',
    # UK & Global
    'AZN', 'GSK', 'RIO', 'BHP', 'SPOT',
]

TICKER_SECTOR = {
    # Tech
    'AAPL':'Tech','MSFT':'Tech','GOOGL':'Tech','AMZN':'Tech','NVDA':'Tech','META':'Tech',
    'TSLA':'Tech','ORCL':'Tech','CRM':'Tech','AMD':'Tech','ADBE':'Tech','NOW':'Tech',
    'INTC':'Tech','AVGO':'Tech','QCOM':'Tech','AMAT':'Tech','MU':'Tech','CRWD':'Tech',
    'PANW':'Tech','NET':'Tech','DDOG':'Tech','SNOW':'Tech','ACN':'Tech','PLTR':'Tech',
    'UBER':'Tech','SHOP':'Tech','SAP.DE':'Tech','ASML.AS':'Tech','SPOT':'Tech',
    # Pharma & Medtech
    'LLY':'Pharma','JNJ':'Pharma','PFE':'Pharma','ABBV':'Pharma','MRK':'Pharma',
    'BMY':'Pharma','MRNA':'Pharma','GILD':'Pharma','REGN':'Pharma','VRTX':'Pharma',
    'ISRG':'Pharma','MDT':'Pharma','AZN':'Pharma','GSK':'Pharma','SAN.PA':'Pharma',
    'BAYN.DE':'Pharma','LONN.SW':'Pharma','NOVN.SW':'Pharma','ROG.SW':'Pharma','NOVO-B.CO':'Pharma',
    # Finance
    'JPM':'Finance','BAC':'Finance','GS':'Finance','MS':'Finance','WFC':'Finance',
    'V':'Finance','MA':'Finance','BLK':'Finance','C':'Finance','PYPL':'Finance',
    'COIN':'Finance','UBS.SW':'Finance','ZURN.SW':'Finance','BNP.PA':'Finance',
    'DBK.DE':'Finance','ALV.DE':'Finance',
    # Consumer
    'WMT':'Consumer','COST':'Consumer','HD':'Consumer','MCD':'Consumer','SBUX':'Consumer',
    'NKE':'Consumer','DIS':'Consumer','NFLX':'Consumer','MC.PA':'Consumer','OR.PA':'Consumer',
    'LVMH.PA':'Consumer','KER.PA':'Consumer','NESN.SW':'Consumer','ADS.DE':'Consumer',
    # Energy
    'XOM':'Energy','CVX':'Energy','TTE.PA':'Energy','RIO':'Energy','BHP':'Energy',
    # Industrial
    'BA':'Industrial','CAT':'Industrial','HON':'Industrial','GE':'Industrial','NEE':'Industrial',
    'LIN':'Industrial','AIR.PA':'Industrial','SIE.DE':'Industrial','ABBN.SW':'Industrial',
    'SU.PA':'Industrial','AI.PA':'Industrial','SIKA.SW':'Industrial',
    # Auto
    'MB.DE':'Auto','BMW.DE':'Auto','VOW3.DE':'Auto',
    # Telecom
    'DTE.DE':'Telecom',
}

FUNDAMENTALS_FILE = Path(__file__).parent / 'data' / 'fundamentals.json'
_reco_cache = {'data': None, 'timestamp': 0}
CACHE_TTL = 3600

SENTIMENT_THRESHOLDS = {
    'euphoric': 0.60, 'accum': 0.25, 'neutral': -0.10,
    'caution': -0.35, 'bearish': -0.60
}

MOMENTUM_WEIGHTS = {'1M': 0.15, '3M': 0.35, '6M': 0.30, '1Y': 0.20}


# ── helpers ──────────────────────────────────────────────────────────────────

def _f(x):
    if x is None: return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def clean(val):
    if val is None: return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return None
    return val

def clean_list(lst): return [clean(v) for v in lst]

def fmt_large(n):
    if n is None: return None
    if abs(n) >= 1e12: return f"{n/1e12:.2f}T"
    if abs(n) >= 1e9:  return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:  return f"{n/1e6:.2f}M"
    return str(round(n, 2))


# ── technical indicators ──────────────────────────────────────────────────────

def enrich_df(df):
    df['RSI']         = ta_lib.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd              = ta_lib.trend.MACD(df['Close'])
    df['MACD']        = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist']   = macd.macd_diff()
    df['SMA50']       = ta_lib.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA200']      = ta_lib.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
    df['MFI']         = ta_lib.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
    df['OBV']         = ta_lib.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    return df


def calc_sentiment_score(mfi, obv_dir, above_dma200):
    scores, weights = [], []
    if mfi is not None:
        scores.append((mfi - 50.0) / 50.0); weights.append(0.40)
    if obv_dir is not None:
        scores.append(float(obv_dir)); weights.append(0.30)
    if above_dma200 is not None:
        scores.append(1.0 if above_dma200 else -1.0); weights.append(0.30)
    if not scores: return None, None
    score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    t = SENTIMENT_THRESHOLDS
    if   score >= t['euphoric']: label = 'Euphoric'
    elif score >= t['accum']:    label = 'Accumulation'
    elif score >= t['neutral']:  label = 'Neutral'
    elif score >= t['caution']:  label = 'Caution'
    elif score >= t['bearish']:  label = 'Bearish'
    else:                        label = 'Extreme Fear'
    return _f(score), label


def calc_momentum_returns(df):
    close = df['Close']
    n = len(close)
    windows = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252}
    result = {}
    for label, w in windows.items():
        result[label] = _f((close.iloc[-1] / close.iloc[-(w+1)] - 1) * 100) if n > w else None
    return result


def obv_direction(obv_series, window=10):
    recent = obv_series.dropna().tail(window + 1)
    if len(recent) < 5: return 0
    diff = recent.iloc[-1] - recent.iloc[0]
    pct = diff / abs(recent.iloc[0]) if recent.iloc[0] != 0 else 0
    return 1 if pct > 0.01 else (-1 if pct < -0.01 else 0)


def smart_money_label(price_change_pct, rvol):
    rising, high_vol = price_change_pct > 0, rvol >= 1.2
    if rising and high_vol:     return 'Accumulation'
    if not rising and high_vol: return 'Distribution'
    if rising:                  return 'Weak Rally'
    return 'Thin Slide'


def build_signals(df):
    score, reasons = 0, []
    cur = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]
    price_change_pct = ((cur - prev) / prev) * 100

    mom = calc_momentum_returns(df)
    mom_score = sum(
        (mom[k] / 100) * w for k, w in MOMENTUM_WEIGHTS.items() if mom.get(k) is not None
    )

    rsi  = _f(df['RSI'].iloc[-1])
    mfi  = _f(df['MFI'].iloc[-1])
    sma50  = _f(df['SMA50'].iloc[-1])
    sma200 = _f(df['SMA200'].iloc[-1])
    macd_v = _f(df['MACD'].iloc[-1])
    macd_s = _f(df['MACD_signal'].iloc[-1])

    avg_vol = df['Volume'].tail(21).iloc[:-1].mean()
    rvol = _f(df['Volume'].iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
    obv_dir = obv_direction(df['OBV'])
    above_dma200 = int(cur > sma200) if sma200 else None
    sentiment_score, sentiment_label = calc_sentiment_score(mfi, obv_dir, above_dma200)

    sm_signal = smart_money_label(price_change_pct, rvol or 1.0)

    # RSI
    if rsi is not None:
        if rsi < 30:   score += 2; reasons.append(('RSI', f'RSI {rsi:.1f} — oversold, historically a buying zone', 'bullish'))
        elif rsi < 45: score += 1; reasons.append(('RSI', f'RSI {rsi:.1f} — below midpoint, room to grow', 'bullish'))
        elif rsi > 70: score -= 2; reasons.append(('RSI', f'RSI {rsi:.1f} — overbought, high reversal risk', 'bearish'))
        elif rsi > 55: score -= 1; reasons.append(('RSI', f'RSI {rsi:.1f} — elevated, watch for slowdown', 'bearish'))
        else:                       reasons.append(('RSI', f'RSI {rsi:.1f} — neutral zone', 'neutral'))

    # MACD
    if macd_v is not None and macd_s is not None:
        if macd_v > macd_s: score += 1; reasons.append(('MACD', 'MACD above signal line — bullish momentum building', 'bullish'))
        else:               score -= 1; reasons.append(('MACD', 'MACD below signal line — bearish momentum', 'bearish'))

    # Moving averages
    if sma50:
        if cur > sma50: score += 1; reasons.append(('Trend', f'Price above 50-day MA ({sma50:.2f}) — short-term uptrend', 'bullish'))
        else:           score -= 1; reasons.append(('Trend', f'Price below 50-day MA ({sma50:.2f}) — short-term downtrend', 'bearish'))
    if sma200:
        if cur > sma200: score += 1; reasons.append(('Trend', f'Price above 200-day MA ({sma200:.2f}) — long-term uptrend', 'bullish'))
        else:            score -= 1; reasons.append(('Trend', f'Price below 200-day MA ({sma200:.2f}) — long-term downtrend', 'bearish'))

    # MFI
    if mfi is not None:
        if mfi < 20:   score += 2; reasons.append(('Money Flow', f'MFI {mfi:.1f} — extremely oversold, smart money may be entering', 'bullish'))
        elif mfi < 40: score += 1; reasons.append(('Money Flow', f'MFI {mfi:.1f} — weak money flow, potential reversal building', 'bullish'))
        elif mfi > 80: score -= 2; reasons.append(('Money Flow', f'MFI {mfi:.1f} — extremely overbought, selling pressure likely', 'bearish'))
        elif mfi > 60: score -= 1; reasons.append(('Money Flow', f'MFI {mfi:.1f} — elevated, watch for peak', 'bearish'))
        else:                       reasons.append(('Money Flow', f'MFI {mfi:.1f} — neutral', 'neutral'))

    # OBV
    if obv_dir == 1:   score += 1; reasons.append(('Volume', 'OBV trending up — institutional accumulation detected', 'bullish'))
    elif obv_dir == -1: score -= 1; reasons.append(('Volume', 'OBV trending down — distribution pressure, smart money selling', 'bearish'))
    else:                            reasons.append(('Volume', 'OBV neutral — no clear institutional direction', 'neutral'))

    # Smart Money
    if sm_signal == 'Accumulation':  score += 1; reasons.append(('Smart Money', f'High volume + rising price (RVOL {rvol:.2f}x) — conviction buying', 'bullish'))
    elif sm_signal == 'Distribution': score -= 1; reasons.append(('Smart Money', f'High volume + falling price (RVOL {rvol:.2f}x) — conviction selling', 'bearish'))
    elif sm_signal == 'Weak Rally':               reasons.append(('Smart Money', f'Rising on low volume (RVOL {rvol:.2f}x) — weak rally, no conviction', 'neutral'))
    else:                                          reasons.append(('Smart Money', f'Falling on low volume (RVOL {rvol:.2f}x) — thin slide, limited pressure', 'neutral'))

    # Multi-timeframe momentum
    if mom_score > 0.05:   score += 1; reasons.append(('Momentum', f"Strong multi-timeframe momentum (1M:{mom.get('1M',0) or 0:+.1f}% 3M:{mom.get('3M',0) or 0:+.1f}% 6M:{mom.get('6M',0) or 0:+.1f}% 1Y:{mom.get('1Y',0) or 0:+.1f}%)", 'bullish'))
    elif mom_score < -0.05: score -= 1; reasons.append(('Momentum', f"Weak multi-timeframe momentum (1M:{mom.get('1M',0) or 0:+.1f}% 3M:{mom.get('3M',0) or 0:+.1f}% 6M:{mom.get('6M',0) or 0:+.1f}% 1Y:{mom.get('1Y',0) or 0:+.1f}%)", 'bearish'))
    else:                               reasons.append(('Momentum', f"Mixed momentum (1M:{mom.get('1M',0) or 0:+.1f}% 3M:{mom.get('3M',0) or 0:+.1f}% 6M:{mom.get('6M',0) or 0:+.1f}% 1Y:{mom.get('1Y',0) or 0:+.1f}%)", 'neutral'))

    signal = 'BUY' if score >= 3 else ('SELL' if score <= -3 else 'HOLD')

    return dict(
        signal=signal, score=score, max_score=12,
        reasons=[{'category': r[0], 'text': r[1], 'direction': r[2]} for r in reasons],
        rsi=rsi, mfi=mfi, obv_dir=obv_dir, sm_signal=sm_signal,
        rvol=_f(rvol), mom=mom,
        sentiment_score=_f(sentiment_score), sentiment_label=sentiment_label,
        sma50=_f(sma50), sma200=_f(sma200),
    )


# ── fundamentals ─────────────────────────────────────────────────────────────

def load_manual_fundamentals():
    if FUNDAMENTALS_FILE.exists():
        return json.loads(FUNDAMENTALS_FILE.read_text())
    return {}

def save_manual_fundamentals(data):
    FUNDAMENTALS_FILE.write_text(json.dumps(data, indent=2))

def fetch_fundamentals(stock, ticker):
    try:
        info = stock.info
    except Exception:
        info = {}

    auto = {
        'pe_ratio':       _f(info.get('trailingPE')),
        'forward_pe':     _f(info.get('forwardPE')),
        'eps':            _f(info.get('trailingEps')),
        'revenue':        info.get('totalRevenue'),
        'profit_margin':  _f(info.get('profitMargins')),
        'revenue_growth': _f(info.get('revenueGrowth')),
        'debt_equity':    _f(info.get('debtToEquity')),
        'free_cashflow':  info.get('freeCashflow'),
        'roe':            _f(info.get('returnOnEquity')),
        'market_cap':     info.get('marketCap'),
        'sector':         info.get('sector'),
        'industry':       info.get('industry'),
        'analyst_target': _f(info.get('targetMeanPrice')),
        'analyst_rating': info.get('recommendationKey'),
    }

    # Quarterly financials (last 4 quarters)
    quarterly = []
    try:
        qf = stock.quarterly_financials
        if qf is not None and not qf.empty:
            rev_key = next((k for k in qf.index if 'Revenue' in k or 'revenue' in k), None)
            ni_key  = next((k for k in qf.index if 'Net Income' in k or 'net income' in k.lower()), None)
            cols = list(qf.columns[:4])
            for col in cols:
                entry = {'period': str(col.date()) if hasattr(col, 'date') else str(col)}
                entry['revenue']    = _f(qf.loc[rev_key, col]) if rev_key else None
                entry['net_income'] = _f(qf.loc[ni_key, col])  if ni_key  else None
                quarterly.append(entry)
    except Exception:
        pass

    # Merge with manual overrides
    manual = load_manual_fundamentals().get(ticker.upper(), {})
    merged = {**auto, **{k: v for k, v in manual.items() if v not in (None, '', 'null')}}

    return {'auto': auto, 'manual': manual, 'merged': merged, 'quarterly': quarterly}


# ── recommendations ──────────────────────────────────────────────────────────

def compute_signal(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='1y')
        if df.empty or len(df) < 20: return None
        df = enrich_df(df)
        sig = build_signals(df)
        cur = df['Close'].iloc[-1]; prev = df['Close'].iloc[-2]
        # Skip tickers with broken / missing price data
        if cur is None or prev is None or math.isnan(cur) or math.isnan(prev) or prev == 0:
            return None
        try:
            info = stock.info
            name = info.get('longName') or info.get('shortName') or ticker
            currency = info.get('currency', '')
        except Exception:
            name = ticker; currency = ''
        return {'ticker': ticker, 'name': name, 'currency': currency,
                'price': round(cur, 2), 'price_change': round((cur-prev)/prev*100, 2),
                'score': sig['score'], 'signal': sig['signal'],
                'sector': TICKER_SECTOR.get(ticker, 'Other')}
    except Exception:
        return None


# ── routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/search')
def search():
    q = request.args.get('q', '').strip()
    if len(q) < 2: return jsonify([])
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(q)}&quotesCount=8&newsCount=0&listsCount=0"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return jsonify([
            {'ticker': i.get('symbol',''), 'name': i.get('longname') or i.get('shortname') or i.get('symbol',''), 'exchange': i.get('exchDisp','')}
            for i in data.get('quotes', []) if i.get('quoteType') in ('EQUITY','ETF')
        ])
    except Exception:
        return jsonify([])


@app.route('/api/recommendations')
def recommendations():
    now = time.time()
    if _reco_cache['data'] and (now - _reco_cache['timestamp']) < CACHE_TTL:
        return jsonify(_reco_cache['data'])
    results = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(compute_signal, t): t for t in WATCHLIST}
        for f in as_completed(futures):
            r = f.result()
            if r: results.append(r)
    buys  = sorted([r for r in results if r['signal']=='BUY'],  key=lambda x: x['score'], reverse=True)[:5]
    sells = sorted([r for r in results if r['signal']=='SELL'], key=lambda x: x['score'])[:5]

    # Group by sector
    sectors = {}
    for r in results:
        s = r['sector']
        if s not in sectors:
            sectors[s] = {'buys': [], 'sells': []}
        if r['signal'] == 'BUY':
            sectors[s]['buys'].append(r)
        elif r['signal'] == 'SELL':
            sectors[s]['sells'].append(r)
    for s in sectors:
        sectors[s]['buys']  = sorted(sectors[s]['buys'],  key=lambda x: x['score'], reverse=True)[:3]
        sectors[s]['sells'] = sorted(sectors[s]['sells'], key=lambda x: x['score'])[:3]

    # Only include sectors that have at least one signal
    sectors = {k: v for k, v in sectors.items() if v['buys'] or v['sells']}

    payload = {'buys': buys, 'sells': sells, 'sectors': sectors, 'scanned': len(results)}
    _reco_cache.update({'data': payload, 'timestamp': now})
    return jsonify(payload)


@app.route('/api/analyze')
def analyze():
    ticker = request.args.get('ticker', '').upper().strip()
    if not ticker: return jsonify({'error': 'No ticker provided'}), 400
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period='1y')
        if df.empty: return jsonify({'error': f'No data found for {ticker}.'}), 404
        df = enrich_df(df)

        cur = df['Close'].iloc[-1]; prev = df['Close'].iloc[-2]
        price_change = (cur - prev) / prev * 100

        sig   = build_signals(df)
        funds = fetch_fundamentals(stock, ticker)

        try:
            info = stock.info
            name     = info.get('longName', ticker)
            currency = info.get('currency', '')
        except Exception:
            name = ticker; currency = ''

        chart_df = df.tail(90)
        return jsonify({
            'ticker': ticker, 'name': name, 'currency': currency,
            'price': round(cur, 2), 'price_change': round(price_change, 2),
            **sig,
            'fundamentals': funds,
            'chart': {
                'dates':       chart_df.index.strftime('%b %d').tolist(),
                'prices':      clean_list(chart_df['Close'].round(2).tolist()),
                'sma50':       clean_list(chart_df['SMA50'].round(2).tolist()),
                'sma200':      clean_list(chart_df['SMA200'].round(2).tolist()),
                'rsi':         clean_list(chart_df['RSI'].round(2).tolist()),
                'mfi':         clean_list(chart_df['MFI'].round(2).tolist()),
                'macd':        clean_list(chart_df['MACD'].round(4).tolist()),
                'macd_signal': clean_list(chart_df['MACD_signal'].round(4).tolist()),
                'macd_hist':   clean_list(chart_df['MACD_hist'].round(4).tolist()),
                'obv':         clean_list(chart_df['OBV'].round(0).tolist()),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/fundamentals/<ticker>', methods=['GET'])
def get_fundamentals(ticker):
    data = load_manual_fundamentals()
    return jsonify(data.get(ticker.upper(), {}))


@app.route('/api/fundamentals/<ticker>', methods=['POST'])
def save_fundamentals(ticker):
    data = load_manual_fundamentals()
    data[ticker.upper()] = request.json
    save_manual_fundamentals(data)
    return jsonify({'ok': True})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
