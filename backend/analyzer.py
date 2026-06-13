"""Dual Strategy Analyzer for REAL and OTC trading pairs.

REAL Strategy  — Trend-Focused Confluence Engine
─────────────────────────────────────────────────
Layer 1 (Trend)       : Dual EMA (20 + 50) — trend direction + strength + slope
Layer 2 (Momentum)    : RSI(14) — trend-aware (overbought ≠ sell in strong trend)
Layer 3 (Fast Entry)  : Stochastic Oscillator (5, 3, 3) — pullback entries
Layer 4 (Confirm)     : MACD (12, 26, 9) — direction + histogram momentum
Layer 5 (Price Action): Patterns + SNR + Consecutive Candles (continuation!)
Layer 6 (Breakout)    : Historical breakout probability (Zeiierman-inspired)
Scoring               : Weighted confluence — trend-following bias
Signal fires ONLY when net score is high enough (probability >= 72%)

OTC Strategy  — Multi-Indicator Confluence Engine
─────────────────────────────────────────────────
Layer 1 (Trend/Extreme)  : Bollinger Bands (20, 2) + RSI(14)
Layer 2 (Fast Momentum)  : Stochastic Oscillator (5, 3, 3)
Layer 3 (Trend Confirm)  : MACD (12, 26, 9)
Layer 4 (Price Action)   : Candle Patterns + Consecutive Counter (reversal!)
Layer 5 (Divergence)     : RSI Divergence Detection
Layer 6 (Breakout)       : Historical breakout probability (Zeiierman-inspired)
Scoring                  : Each layer votes → weighted confluence score
Signal fires ONLY when probability >= 65%
"""

import time
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def analyze(asset_name: str, candles: list[dict]) -> dict:
    """Main entry point. Routes to REAL or OTC strategy."""
    is_otc = "_otc" in asset_name.lower()
    market_type = "OTC" if is_otc else "REAL"

    if len(candles) < 30:          # MACD needs 26+, Stoch needs 5+, buffer = 30
        return _error_result(asset_name, market_type)

    df = pd.DataFrame(candles)

    print(f"[DEBUG ANALYZER] Raw columns for {asset_name}: {df.columns.tolist()}")

    # Fix column names: pyquotex returns capitalized, ta library expects lowercase
    df.columns = [c.lower() for c in df.columns]
    print(f"[DEBUG ANALYZER] Fixed columns: {df.columns.tolist()}")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return _error_result(asset_name, market_type)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=required, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 30:
        return _error_result(asset_name, market_type)

    if "time" not in df.columns:
        df["time"] = range(len(df))

    # Get the last candle's open time for accurate countdown
    last_candle_time = int(df["time"].iloc[-1]) if "time" in df.columns else int(time.time())

    if is_otc:
        return _analyze_otc(asset_name, df, last_candle_time)
    else:
        return _analyze_real(asset_name, df, last_candle_time)


def _error_result(asset: str, market: str) -> dict:
    return {
        "status": "Error",
        "asset": asset,
        "market_type": market,
        "next_candle_prediction": "Neutral",
        "trend": "Insufficient data",
        "probability": 0,
        "suggestion": "No Skip (Wait For High Accuracy)",
        "candle_seconds_remaining": 0,
        "entry_timing": "WAIT",
    }


# ═══════════════════════════════════════════════════════════════════
#  STRATEGY 1: REAL PAIRS — TREND-FOCUSED CONFLUENCE ENGINE
# ═══════════════════════════════════════════════════════════════════

def _analyze_real(asset: str, df: pd.DataFrame, last_candle_time: int) -> dict:
    """
    6-layer TREND-FOCUSED confluence engine for REAL binary prediction.
    Key difference from OTC: REAL markets trend, so we FOLLOW momentum,
    use pullbacks as entries, and only reverse on VERY strong evidence.
    """
    try:
        price = float(df["close"].iloc[-1])

        # ── LAYER 1: Dual EMA Trend (20 + 50) ────────────────────────────
        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        ema20 = df["close"].ewm(span=20, adjust=False).mean()

        ema50_val = float(ema50.iloc[-1])
        ema20_val = float(ema20.iloc[-1])

        # Determine trend context (stronger signal when both EMAs agree)
        if price > ema50_val and ema20_val > ema50_val:
            trend_context = "STRONG_UP"
            trend = "Strong Up trend"
        elif price > ema50_val:
            trend_context = "UP"
            trend = "Up trend"
        elif price < ema50_val and ema20_val < ema50_val:
            trend_context = "STRONG_DOWN"
            trend = "Strong Down trend"
        else:
            trend_context = "DOWN"
            trend = "Down trend"

        up_score   = 0
        down_score = 0

        # EMA trend votes
        if "UP" in trend_context:
            up_score += 10
            if trend_context == "STRONG_UP":
                up_score += 8   # Golden cross bonus (EMA20 > EMA50)
            if price > ema20_val:
                up_score += 4   # Price above fast EMA
        else:
            down_score += 10
            if trend_context == "STRONG_DOWN":
                down_score += 8 # Death cross bonus (EMA20 < EMA50)
            if price < ema20_val:
                down_score += 4

        # EMA slope: accelerating trend = stronger signal
        ema20_prev = float(ema20.iloc[-3]) if len(df) > 3 else ema20_val
        ema20_slope = ema20_val - ema20_prev
        if "UP" in trend_context and ema20_slope > 0:
            up_score += 6       # EMA20 rising = accelerating uptrend
        elif "DOWN" in trend_context and ema20_slope < 0:
            down_score += 6     # EMA20 falling = accelerating downtrend

        # ── LAYER 2: RSI (Trend-Aware) ───────────────────────────────────
        # REAL rule: Don't sell overbought in uptrend, don't buy oversold in downtrend
        rsi_ser = RSIIndicator(close=df["close"], window=14).rsi()
        rsi_val = float(rsi_ser.iloc[-1])

        if "UP" in trend_context:
            if 50 <= rsi_val <= 70:
                up_score += 10     # Healthy uptrend momentum
            elif rsi_val > 75:
                pass               # Don't sell — strong trends stay overbought
            elif rsi_val < 40:
                down_score += 8    # Losing momentum in uptrend = warning
        else:
            if 30 <= rsi_val <= 50:
                down_score += 10   # Healthy downtrend momentum
            elif rsi_val < 25:
                pass               # Don't buy — strong trends stay oversold
            elif rsi_val > 60:
                up_score += 8      # Gaining momentum in downtrend = warning

        # RSI momentum direction: rising RSI = bullish momentum, falling = bearish
        rsi_prev = float(rsi_ser.iloc[-3]) if len(df) > 3 else rsi_val
        rsi_slope = rsi_val - rsi_prev
        if "UP" in trend_context and rsi_slope > 0:
            up_score += 4         # RSI rising in uptrend = momentum confirmed
        elif "DOWN" in trend_context and rsi_slope < 0:
            down_score += 4       # RSI falling in downtrend = momentum confirmed
        elif "UP" in trend_context and rsi_slope < -3:
            down_score += 3       # RSI dropping fast in uptrend = momentum fading
        elif "DOWN" in trend_context and rsi_slope > 3:
            up_score += 3         # RSI rising fast in downtrend = momentum fading

        # ── LAYER 3: Stochastic (Pullback Entries in Trend) ──────────────
        stoch = StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"],
            window=5, smooth_window=3
        )
        stoch_k      = float(stoch.stoch().iloc[-1])
        stoch_d      = float(stoch.stoch_signal().iloc[-1])
        stoch_k_prev = float(stoch.stoch().iloc[-2]) if len(df) > 2 else stoch_k
        stoch_d_prev = float(stoch.stoch_signal().iloc[-2]) if len(df) > 2 else stoch_d

        stoch_cross_up   = (stoch_k_prev < stoch_d_prev) and (stoch_k > stoch_d)
        stoch_cross_down = (stoch_k_prev > stoch_d_prev) and (stoch_k < stoch_d)

        if "UP" in trend_context:
            if stoch_cross_up and stoch_k < 50:
                up_score += 14    # Pullback buy: stoch crossing up from low zone
            elif stoch_cross_up:
                up_score += 8
            elif stoch_k < 20:
                up_score += 10    # Oversold in uptrend = great buy opportunity
                if stoch_k > stoch_k_prev:
                    up_score += 6 # Stoch rising from oversold = strong reversal
            elif stoch_k > 80:
                pass              # Don't sell overbought in uptrend
            elif stoch_k > stoch_d:
                up_score += 4
            else:
                down_score += 3
        else:
            if stoch_cross_down and stoch_k > 50:
                down_score += 14  # Pullback sell: stoch crossing down from high zone
            elif stoch_cross_down:
                down_score += 8
            elif stoch_k > 80:
                down_score += 10  # Overbought in downtrend = great sell
                if stoch_k < stoch_k_prev:
                    down_score += 6  # Stoch falling from overbought = strong reversal
            elif stoch_k < 20:
                pass              # Don't buy oversold in downtrend
            elif stoch_k < stoch_d:
                down_score += 4
            else:
                up_score += 3

        # ── LAYER 4: MACD (Trend Direction + Momentum) ───────────────────
        macd_obj    = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        macd_line   = macd_obj.macd()
        macd_signal = macd_obj.macd_signal()
        macd_hist   = macd_obj.macd_diff()

        macd_now      = float(macd_line.iloc[-1])
        macd_sig_now  = float(macd_signal.iloc[-1])
        macd_hist_now = float(macd_hist.iloc[-1])
        macd_prev     = float(macd_line.iloc[-2]) if len(df) > 2 else macd_now
        macd_sig_prev = float(macd_signal.iloc[-2]) if len(df) > 2 else macd_sig_now
        macd_hist_prev= float(macd_hist.iloc[-2]) if len(df) > 2 else macd_hist_now

        # NaN guard: MACD needs 26+ candles; skip votes if values are NaN
        macd_valid = not (np.isnan(macd_now) or np.isnan(macd_sig_now))

        macd_cross_up   = macd_valid and (macd_prev < macd_sig_prev) and (macd_now > macd_sig_now)
        macd_cross_down = macd_valid and (macd_prev > macd_sig_prev) and (macd_now < macd_sig_now)

        if macd_valid:
            if macd_cross_up:
                up_score   += 12
            elif macd_cross_down:
                down_score += 12
            elif macd_now > macd_sig_now:
                up_score   += 6
            else:
                down_score += 6

            # Histogram momentum (accelerating?)
            if not np.isnan(macd_hist_now) and not np.isnan(macd_hist_prev):
                if macd_hist_now > 0 and macd_hist_now > macd_hist_prev:
                    up_score   += 4
                elif macd_hist_now < 0 and macd_hist_now < macd_hist_prev:
                    down_score += 4

            # MACD zero-line: above zero in uptrend = established trend
            if macd_now > 0 and "UP" in trend_context:
                up_score   += 4   # MACD above zero = bullish trend established
            elif macd_now < 0 and "DOWN" in trend_context:
                down_score += 4   # MACD below zero = bearish trend established

        # ── LAYER 5: Price Action + SNR + Consecutive ────────────────────
        pattern = _detect_pattern_advanced(df)
        support, resistance = _find_snr(df, lookback=100)
        consec = _count_consecutive_candles(df, lookback=6)
        gap = _find_gap(df, lookback=10)

        near_resistance = resistance is not None and abs(price - resistance) / resistance < 0.0005
        near_support = support is not None and abs(price - support) / support < 0.0005

        # Candle patterns: WITH trend = full weight, AGAINST trend = reduced
        if "UP" in trend_context:
            if pattern in ("Morning Star", "Bullish Engulfing", "Hammer", "Doji_at_Low"):
                up_score   += 12  # Pattern with trend = strong
            elif pattern in ("Evening Star", "Bearish Engulfing", "Shooting Star", "Doji_at_High"):
                down_score += 4   # Pattern against trend = weak
        else:
            if pattern in ("Evening Star", "Bearish Engulfing", "Shooting Star", "Doji_at_High"):
                down_score += 12
            elif pattern in ("Morning Star", "Bullish Engulfing", "Hammer", "Doji_at_Low"):
                up_score   += 4

        # SNR — MUCH more important in REAL markets than OTC
        if near_support:
            up_score += 10
        if near_resistance:
            down_score += 10

        # Consecutive candles — CONTINUATION in REAL (opposite logic of OTC!)
        if "UP" in trend_context:
            if 3 <= consec <= 4:
                up_score += 8     # Momentum continuation
            elif consec >= 5:
                up_score += 5     # Extended trend (might pause, reduced score)
            elif consec <= -3:
                up_score += 6     # Pullback in uptrend = buy opportunity
        else:
            if -3 >= consec >= -4:
                down_score += 8
            elif consec <= -5:
                down_score += 5
            elif consec >= 3:
                down_score += 6   # Pullback in downtrend = sell opportunity

        # Gap — only trust if in trend direction
        if gap == "above" and "UP" in trend_context:
            up_score += 6
        elif gap == "below" and "DOWN" in trend_context:
            down_score += 6

        # Candle body strength: large body = strong momentum (volume proxy)
        last_candle = df.iloc[-1]
        body = abs(last_candle["close"] - last_candle["open"])
        candle_range = last_candle["high"] - last_candle["low"] + 1e-10
        body_ratio = body / candle_range
        is_bullish = last_candle["close"] > last_candle["open"]

        if body_ratio > 0.6:
            if is_bullish and "UP" in trend_context:
                up_score += 6     # Large bullish candle in uptrend = strong momentum
            elif not is_bullish and "DOWN" in trend_context:
                down_score += 6   # Large bearish candle in downtrend = strong momentum
            elif is_bullish and "DOWN" in trend_context:
                up_score += 3     # Large bullish candle against downtrend = possible reversal
            elif not is_bullish and "UP" in trend_context:
                down_score += 3   # Large bearish candle against uptrend = possible reversal
        elif body_ratio < 0.15:
            # Doji-like: indecision, reduce confidence in current direction
            if "UP" in trend_context:
                down_score += 2
            else:
                up_score += 2

        # RSI Divergence — only count if AGAINST the trend (strong reversal warning)
        rsi_div = _detect_rsi_divergence(df, rsi_ser)
        if rsi_div == "bearish" and "UP" in trend_context:
            down_score += 16     # Trend reversal warning
        elif rsi_div == "bullish" and "DOWN" in trend_context:
            up_score += 16

        # ── LAYER 6: Breakout Probability ────────────────────────────────
        bp_up, bp_down = _breakout_probability(df, perc=1.0)
        if bp_up > bp_down and bp_up > 0.55:
            up_score += int((bp_up - 0.5) * 40)
        elif bp_down > bp_up and bp_down > 0.55:
            down_score += int((bp_down - 0.5) * 40)

        # ── CONFLUENCE SCORING → DIRECTION + PROBABILITY ─────────────────
        total_score = up_score + down_score

        if up_score > down_score:
            prediction = "Up"
            net = up_score - down_score
        elif down_score > up_score:
            prediction = "Down"
            net = down_score - up_score
        else:
            prediction = "Up" if "UP" in trend_context else "Down"
            net = 0

        # Count how many layers voted with the winning direction
        confluence_count = 0
        if "UP" in trend_context:
            if ema20_slope > 0:                     confluence_count += 1
            if 50 <= rsi_val <= 70 and rsi_slope > 0: confluence_count += 1
            if stoch_k > stoch_d or stoch_cross_up: confluence_count += 1
            if macd_valid and macd_now > macd_sig_now: confluence_count += 1
            if pattern in ("Morning Star", "Bullish Engulfing", "Hammer"): confluence_count += 1
            if near_support:                         confluence_count += 1
            if body_ratio > 0.6 and is_bullish:     confluence_count += 1
        else:
            if ema20_slope < 0:                     confluence_count += 1
            if 30 <= rsi_val <= 50 and rsi_slope < 0: confluence_count += 1
            if stoch_k < stoch_d or stoch_cross_down: confluence_count += 1
            if macd_valid and macd_now < macd_sig_now: confluence_count += 1
            if pattern in ("Evening Star", "Bearish Engulfing", "Shooting Star"): confluence_count += 1
            if near_resistance:                      confluence_count += 1
            if body_ratio > 0.6 and not is_bullish: confluence_count += 1

        # Tiered probability: stronger net = higher base + steeper boost
        if net >= 26:
            prob_base = 60
            prob_boost = min(35, int(net * 0.75))
        elif net >= 11:
            prob_base = 57
            prob_boost = min(35, int(net * 0.65))
        else:
            prob_base = 55
            prob_boost = min(30, int(net * 0.5))
        probability = prob_base + prob_boost
        probability = max(45, min(95, probability))

        # ── SIGNAL QUALITY & SUGGESTION ──────────────────────────────────
        if probability < 72:
            suggestion = "No Skip (Wait For High Accuracy)"
        elif rsi_div and (
            (rsi_div == "bearish" and prediction == "Down") or
            (rsi_div == "bullish" and prediction == "Up")
        ):
            suggestion = "Yes (Trend Reversal — Divergence Confirmed)"
        elif prediction == ("Up" if "UP" in trend_context else "Down"):
            if confluence_count >= 6:
                if near_support and prediction == "Up":
                    suggestion = "Yes (Strong — Support Bounce + Multi-Layer)"
                elif near_resistance and prediction == "Down":
                    suggestion = "Yes (Strong — Resistance Reject + Multi-Layer)"
                else:
                    suggestion = "Yes (Strong — High Confluence)"
            elif confluence_count >= 4:
                if near_support and prediction == "Up":
                    suggestion = "Yes (Trend Continuation — Support Bounce)"
                elif near_resistance and prediction == "Down":
                    suggestion = "Yes (Trend Continuation — Resistance Reject)"
                elif stoch_cross_up and "UP" in trend_context and stoch_k < 50:
                    suggestion = "Yes (Trend Pullback — Stochastic Entry)"
                elif stoch_cross_down and "DOWN" in trend_context and stoch_k > 50:
                    suggestion = "Yes (Trend Pullback — Stochastic Entry)"
                elif macd_cross_up or macd_cross_down:
                    suggestion = "Yes (MACD Crossover + Trend)"
                else:
                    suggestion = "Yes (Trend is Friend — Multi-Confirm)"
            else:
                suggestion = "Caution (Moderate Signal — Few Layers Agree)"
        else:
            suggestion = "Caution (Counter-Trend Signal — Small Size)"

        # Entry timing (avoid last 20s of candle)
        now = int(time.time())

        # Use the passed last_candle_time parameter, fallback to local clock
        try:
            candle_ts = int(last_candle_time)
            if candle_ts > 4102444800:
                candle_ts = candle_ts // 1000
            seconds_remaining = 60 - (now - candle_ts)
            if not (0 <= seconds_remaining <= 60):
                raise ValueError(f"out of range: {seconds_remaining}")
        except Exception:
            seconds_remaining = 60 - (now % 60)

        entry_ok = seconds_remaining >= 20

        print(
            f"[DEBUG REAL] {asset}: price={price:.5f}  "
            f"ema20={ema20_val:.5f} ema50={ema50_val:.5f} [{trend_context}]  "
            f"rsi={rsi_val:.1f}  stoch_k={stoch_k:.1f}/d={stoch_d:.1f}  "
            f"macd={'▲' if (macd_valid and macd_now > macd_sig_now) else '▼' if macd_valid else '?'}  "
            f"pattern={pattern}  consec={consec:+d}  "
            f"snr=S:{f'{support:.5f}' if support else 'None'} R:{f'{resistance:.5f}' if resistance else 'None'}  "
            f"bp_up={bp_up:.2f}/bp_down={bp_down:.2f}  "
            f"up={up_score} down={down_score} conf={confluence_count} -> {prediction} ({probability}%)"
        )

        return {
            "status": "Result",
            "asset": asset,
            "market_type": "REAL",
            "next_candle_prediction": prediction,
            "trend": trend,
            "probability": round(probability, 1),
            "suggestion": suggestion if entry_ok else "No Skip (Candle Almost Closed — Wait Next)",
            "candle_seconds_remaining": seconds_remaining,
            "entry_timing": "GOOD" if entry_ok else "WAIT",
        }

    except Exception as e:
        print(f"[ERROR REAL] {e}")
        import traceback
        traceback.print_exc()
        return _error_result(asset, "REAL")


# ═══════════════════════════════════════════════════════════════════
#  STRATEGY 2: OTC PAIRS — MULTI-INDICATOR CONFLUENCE (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════

def _analyze_otc(asset: str, df: pd.DataFrame, last_candle_time: int) -> dict:
    """
    6-layer confluence engine for OTC binary prediction.
    Each layer votes Up/Down with a weighted score.
    Direction = majority vote side; Probability = 50 + f(net_score).
    """
    try:
        price = float(df["close"].iloc[-1])

        # ── LAYER 1: Bollinger Bands (20, 2) + RSI (14) ──────────────────
        bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        bb_upper = float(bb.bollinger_hband().iloc[-1])
        bb_lower = float(bb.bollinger_lband().iloc[-1])
        bb_mid   = float(bb.bollinger_mavg().iloc[-1])

        rsi_ser = RSIIndicator(close=df["close"], window=14).rsi()
        rsi_val = float(rsi_ser.iloc[-1])

        up_score   = 0
        down_score = 0

        if price > bb_upper and rsi_val > 70:
            down_score += 25
        elif price < bb_lower and rsi_val < 30:
            up_score   += 25
        elif price > bb_upper:
            down_score += 15
        elif price < bb_lower:
            up_score   += 15
        elif rsi_val > 65:
            down_score += 8
        elif rsi_val < 35:
            up_score   += 8

        if price > bb_mid:
            up_score   += 5
        else:
            down_score += 5

        # ── LAYER 2: Stochastic Oscillator (5, 3, 3) ─────────────────────
        stoch = StochasticOscillator(
            high=df["high"], low=df["low"], close=df["close"],
            window=5, smooth_window=3
        )
        stoch_k_ser = stoch.stoch()
        stoch_d_ser = stoch.stoch_signal()

        stoch_k      = float(stoch_k_ser.iloc[-1])
        stoch_d      = float(stoch_d_ser.iloc[-1])
        stoch_k_prev = float(stoch_k_ser.iloc[-2]) if len(df) > 2 else stoch_k
        stoch_d_prev = float(stoch_d_ser.iloc[-2]) if len(df) > 2 else stoch_d

        stoch_cross_up   = (stoch_k_prev < stoch_d_prev) and (stoch_k > stoch_d)
        stoch_cross_down = (stoch_k_prev > stoch_d_prev) and (stoch_k < stoch_d)

        if stoch_k > 80:
            down_score += 12
            if stoch_cross_down:
                down_score += 8
        elif stoch_k < 20:
            up_score += 12
            if stoch_cross_up:
                up_score += 8
        elif stoch_cross_up:
            up_score   += 10
        elif stoch_cross_down:
            down_score += 10
        elif stoch_k > stoch_d:
            up_score   += 4
        else:
            down_score += 4

        # ── LAYER 3: MACD (12, 26, 9) ────────────────────────────────────
        macd_obj    = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        macd_line   = macd_obj.macd()
        macd_signal = macd_obj.macd_signal()
        macd_hist   = macd_obj.macd_diff()

        macd_line_now  = float(macd_line.iloc[-1])
        macd_sig_now   = float(macd_signal.iloc[-1])
        macd_hist_now  = float(macd_hist.iloc[-1])
        macd_line_prev = float(macd_line.iloc[-2]) if len(df) > 2 else macd_line_now
        macd_sig_prev  = float(macd_signal.iloc[-2]) if len(df) > 2 else macd_sig_now
        macd_hist_prev = float(macd_hist.iloc[-2]) if len(df) > 2 else macd_hist_now

        macd_cross_up   = (macd_line_prev < macd_sig_prev) and (macd_line_now > macd_sig_now)
        macd_cross_down = (macd_line_prev > macd_sig_prev) and (macd_line_now < macd_sig_now)

        # NaN guard for MACD
        macd_valid = not (np.isnan(macd_line_now) or np.isnan(macd_sig_now))

        if macd_valid:
            if macd_cross_up:
                up_score   += 12
            elif macd_cross_down:
                down_score += 12
            elif macd_line_now > macd_sig_now:
                up_score   += 6
            else:
                down_score += 6

            if not np.isnan(macd_hist_now) and not np.isnan(macd_hist_prev):
                if macd_hist_now > 0 and macd_hist_now > macd_hist_prev:
                    up_score   += 4
                elif macd_hist_now < 0 and macd_hist_now < macd_hist_prev:
                    down_score += 4

        # ── LAYER 4: Candle Patterns + Consecutive Candle Count ──────────
        pattern = _detect_pattern_advanced(df)
        consec  = _count_consecutive_candles(df, lookback=6)

        if pattern in ("Morning Star", "Bullish Engulfing", "Hammer", "Doji_at_Low"):
            up_score   += 12
        elif pattern in ("Evening Star", "Bearish Engulfing", "Shooting Star", "Doji_at_High"):
            down_score += 12
        elif pattern == "Three_White_Soldiers":
            up_score   += 6
        elif pattern == "Three_Black_Crows":
            down_score += 6

        # OTC: Consecutive = MEAN REVERSION (opposite of REAL!)
        if consec >= 5:
            down_score += 15
        elif consec == 4:
            down_score += 10
        elif consec == 3:
            down_score += 5
        elif consec <= -5:
            up_score   += 15
        elif consec == -4:
            up_score   += 10
        elif consec == -3:
            up_score   += 5

        # ── LAYER 5: RSI Divergence ───────────────────────────────────────
        rsi_div = _detect_rsi_divergence(df, rsi_ser)
        if rsi_div == "bearish":
            down_score += 18
        elif rsi_div == "bullish":
            up_score   += 18

        # ── LAYER 6: Breakout Probability ────────────────────────────────
        bp_up, bp_down = _breakout_probability(df, perc=1.0)
        if bp_up > bp_down and bp_up > 0.55:
            up_score += int((bp_up - 0.5) * 40)
        elif bp_down > bp_up and bp_down > 0.55:
            down_score += int((bp_down - 0.5) * 40)

        # ── CONFLUENCE SCORING → DIRECTION + PROBABILITY ─────────────────
        total_score = up_score + down_score

        if up_score > down_score:
            prediction    = "Up"
            net           = up_score - down_score
            agreement_pct = up_score / total_score if total_score > 0 else 0.5
        elif down_score > up_score:
            prediction    = "Down"
            net           = down_score - up_score
            agreement_pct = down_score / total_score if total_score > 0 else 0.5
        else:
            prediction    = "Up" if price <= bb_mid else "Down"
            net           = 0
            agreement_pct = 0.5

        prob_boost  = min(45, int(net * 0.65))
        probability = 50 + prob_boost
        probability = max(45, min(95, probability))

        # ── SIGNAL QUALITY & SUGGESTION ──────────────────────────────────
        if probability < 65:
            suggestion = "No Skip (Wait For High Accuracy)"
        elif probability < 72:
            suggestion = "Caution (Moderate Signal — Use Small Size)"
        elif rsi_div:
            suggestion = "Yes (RSI Divergence Confirmed)"
        elif (price > bb_upper or price < bb_lower) and (stoch_k > 80 or stoch_k < 20):
            suggestion = "Yes (BB Extreme + Stochastic Aligned)"
        elif macd_cross_up or macd_cross_down:
            suggestion = "Yes (MACD Crossover + Confluence)"
        elif stoch_cross_up or stoch_cross_down:
            suggestion = "Yes (Stochastic Crossover)"
        elif abs(consec) >= 4:
            suggestion = "Yes (Mean Reversion — Consecutive Candles)"
        else:
            suggestion = "Yes (Multi-Indicator Confluence)"

        # Entry timing (avoid last 20s of candle)
        now = int(time.time())

        # Use the passed last_candle_time parameter, fallback to local clock
        try:
            candle_ts = int(last_candle_time)
            if candle_ts > 4102444800:
                candle_ts = candle_ts // 1000
            seconds_remaining = 60 - (now - candle_ts)
            if not (0 <= seconds_remaining <= 60):
                raise ValueError(f"out of range: {seconds_remaining}")
        except Exception:
            seconds_remaining = 60 - (now % 60)

        entry_ok = seconds_remaining >= 20

        print(
            f"[DEBUG OTC] {asset}: price={price:.5f}  bb=[{bb_lower:.5f}~{bb_upper:.5f}]  "
            f"rsi={rsi_val:.1f}  stoch_k={stoch_k:.1f}/d={stoch_d:.1f}  "
            f"macd={'▲' if (macd_valid and macd_line_now > macd_sig_now) else '▼' if macd_valid else '?'}  "
            f"pattern={pattern}  consec={consec:+d}  rsi_div={rsi_div}  "
            f"bp_up={bp_up:.2f}/bp_down={bp_down:.2f}  "
            f"up={up_score} down={down_score} -> {prediction} ({probability}%)"
        )

        return {
            "status": "Result",
            "asset": asset,
            "market_type": "OTC",
            "next_candle_prediction": prediction,
            "trend": "Up trend" if price > bb_mid else "Down trend",
            "probability": round(probability, 1),
            "suggestion": suggestion if entry_ok else "No Skip (Candle Almost Closed — Wait Next)",
            "candle_seconds_remaining": seconds_remaining,
            "entry_timing": "GOOD" if entry_ok else "WAIT",
        }

    except Exception as e:
        print(f"[ERROR OTC] {e}")
        import traceback
        traceback.print_exc()
        return _error_result(asset, "OTC")


# ═══════════════════════════════════════════════════════════════════
#  SHARED HELPER FUNCTIONS (used by BOTH strategies)
# ═══════════════════════════════════════════════════════════════════

def _detect_rsi_divergence(df: pd.DataFrame, rsi_ser: pd.Series, lookback: int = 12) -> str | None:
    """Detect RSI divergence over the last `lookback` candles."""
    if len(df) < lookback:
        return None

    closes   = df["close"].values[-lookback:]
    rsi_vals = rsi_ser.values[-lookback:]

    valid = ~np.isnan(rsi_vals)
    if valid.sum() < 6:
        return None
    closes   = closes[valid]
    rsi_vals = rsi_vals[valid]

    n       = len(closes)
    half    = n // 2
    recent  = slice(max(0, n - 4), n)
    older   = slice(0, half)

    recent_price_high = closes[recent].max()
    older_price_high  = closes[older].max()
    recent_rsi_high   = rsi_vals[recent].max()
    older_rsi_high    = rsi_vals[older].max()

    recent_price_low  = closes[recent].min()
    older_price_low   = closes[older].min()
    recent_rsi_low    = rsi_vals[recent].min()
    older_rsi_low     = rsi_vals[older].min()

    price_higher = recent_price_high > older_price_high * 1.0002
    rsi_lower    = recent_rsi_high   < older_rsi_high   - 3.0
    if price_higher and rsi_lower:
        return "bearish"

    price_lower  = recent_price_low  < older_price_low  * 0.9998
    rsi_higher   = recent_rsi_low    > older_rsi_low    + 3.0
    if price_lower and rsi_higher:
        return "bullish"

    return None


def _count_consecutive_candles(df: pd.DataFrame, lookback: int = 6) -> int:
    """Count consecutive same-colored candles from the most recent one."""
    tail = df.tail(lookback)
    if len(tail) == 0:
        return 0

    colors = []
    for _, row in tail.iterrows():
        body = abs(row["close"] - row["open"])
        rng  = row["high"] - row["low"] + 1e-10
        if body / rng < 0.05:
            colors.append(0)   # Doji
        elif row["close"] >= row["open"]:
            colors.append(1)   # Green
        else:
            colors.append(-1)  # Red

    last_color = colors[-1]
    if last_color == 0:
        return 0

    count = 1
    for i in range(len(colors) - 2, -1, -1):
        if colors[i] == last_color:
            count += 1
        else:
            break

    return count * last_color


def _detect_pattern_advanced(df: pd.DataFrame) -> str:
    """Advanced candlestick pattern detection (shared by REAL & OTC)."""
    n = len(df)
    if n < 2:
        return "None"

    def body(c):         return abs(c["close"] - c["open"])
    def rng(c):          return c["high"] - c["low"] + 1e-10
    def is_bull(c):      return c["close"] > c["open"]
    def is_bear(c):      return c["close"] < c["open"]
    def is_doji(c):      return body(c) / rng(c) < 0.1
    def upper_wick(c):   return c["high"] - max(c["open"], c["close"])
    def lower_wick(c):   return min(c["open"], c["close"]) - c["low"]

    c3 = df.iloc[-1]
    c2 = df.iloc[-2]

    # ── THREE-CANDLE PATTERNS ─────────────────────────────────────────
    if n >= 3:
        c1 = df.iloc[-3]

        if (is_bear(c1) and body(c1) > rng(c1) * 0.4 and
                body(c2) < body(c1) * 0.5 and
                is_bull(c3) and c3["close"] > (c1["open"] + c1["close"]) / 2):
            return "Morning Star"

        if (is_bull(c1) and body(c1) > rng(c1) * 0.4 and
                body(c2) < body(c1) * 0.5 and
                is_bear(c3) and c3["close"] < (c1["open"] + c1["close"]) / 2):
            return "Evening Star"

        # Three White Soldiers: 3 consecutive bullish candles with ascending closes
        if (is_bull(c1) and is_bull(c2) and is_bull(c3) and
                body(c1) > rng(c1) * 0.5 and
                body(c2) > rng(c2) * 0.5 and
                body(c3) > rng(c3) * 0.5 and
                c2["close"] > c1["close"] and c3["close"] > c2["close"]):
            return "Three_White_Soldiers"

        # Three Black Crows: 3 consecutive bearish candles with descending closes
        if (is_bear(c1) and is_bear(c2) and is_bear(c3) and
                body(c1) > rng(c1) * 0.5 and
                body(c2) > rng(c2) * 0.5 and
                body(c3) > rng(c3) * 0.5 and
                c2["close"] < c1["close"] and c3["close"] < c2["close"]):
            return "Three_Black_Crows"

    # ── TWO-CANDLE PATTERNS ───────────────────────────────────────────
    if (is_bear(c2) and is_bull(c3) and
            c3["open"] <= c2["close"] and c3["close"] >= c2["open"]):
        return "Bullish Engulfing"

    if (is_bull(c2) and is_bear(c3) and
            c3["open"] >= c2["close"] and c3["close"] <= c2["open"]):
        return "Bearish Engulfing"

    # ── SINGLE-CANDLE PATTERNS ────────────────────────────────────────
    uw = upper_wick(c3)
    lw = lower_wick(c3)
    bd = body(c3)

    if is_doji(c3) and uw > lw * 1.5:
        return "Doji_at_High"
    if is_doji(c3) and lw > uw * 1.5:
        return "Doji_at_Low"
    if is_doji(c3):
        return "Doji_Neutral"
    if bd > 0 and lw >= bd * 2.0 and uw <= bd * 0.5:
        return "Hammer"
    if bd > 0 and uw >= bd * 2.0 and lw <= bd * 0.5:
        return "Shooting Star"

    return "None"


# ═══════════════════════════════════════════════════════════════════
#  REAL-ONLY HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _find_snr(df: pd.DataFrame, lookback: int = 100):
    """Find nearest support and resistance levels using swing highs/lows."""
    highs  = df["high"].values[-lookback:]
    lows   = df["low"].values[-lookback:]
    price  = df["close"].values[-1]
    n      = len(highs)

    swing_highs = []
    swing_lows  = []
    for i in range(2, n - 2):
        if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and
                highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
            swing_highs.append(float(highs[i]))
        if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and
                lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
            swing_lows.append(float(lows[i]))

    resistances = sorted([r for r in swing_highs if r > price])
    resistance  = resistances[0] if resistances else None

    supports = sorted([s for s in swing_lows if s < price], reverse=True)
    support  = supports[0] if supports else None

    return support, resistance


def _find_gap(df: pd.DataFrame, lookback: int = 10) -> str | None:
    """Detect recent price gaps."""
    closes = df["close"].values[-lookback:]
    for i in range(len(closes) - 1, 0, -1):
        pct = abs(closes[i] - closes[i - 1]) / (closes[i - 1] + 1e-10)
        if pct > 0.003:
            return "above" if closes[i] > closes[i - 1] else "below"
    return None


def _breakout_probability(df: pd.DataFrame, perc: float = 1.0):
    """
    Historical conditional breakout probability.
    Inspired by Zeiierman's Breakout Probability indicator (Pine Script).
    Calculates: given last candle was green/red, how often did price
    break above prev_high+step or below prev_low-step historically?
    Returns (up_prob, down_prob) as floats between 0.0 and 1.0.
    """
    green_total = red_total = 0
    green_hi = green_lo = red_hi = red_lo = 0

    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        step = prev["close"] * (perc / 100)

        is_green = prev["close"] > prev["open"]
        is_red   = prev["close"] < prev["open"]

        if is_green:
            green_total += 1
            if curr["high"] >= prev["high"] + step:
                green_hi += 1
            if curr["low"] <= prev["low"] - step:
                green_lo += 1
        elif is_red:
            red_total += 1
            if curr["high"] >= prev["high"] + step:
                red_hi += 1
            if curr["low"] <= prev["low"] - step:
                red_lo += 1

    last = df.iloc[-1]
    last_is_green = last["close"] > last["open"]

    if last_is_green and green_total > 0:
        return green_hi / green_total, green_lo / green_total
    elif not last_is_green and red_total > 0:
        return red_hi / red_total, red_lo / red_total
    return 0.5, 0.5