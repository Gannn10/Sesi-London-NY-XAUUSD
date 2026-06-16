import pandas as pd
import numpy as np
import logging
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Hidden Markov Model (HMM) untuk mendeteksi rezim market.
    Membagi market menjadi 3 state:
    - Low Volatility (Trending/Calm)
    - High Volatility (Trending/News)
    - Choppy (Sideways/Whipsaw)
    """
    def __init__(self, n_states=3, lookback=500):
        self.n_states = n_states
        self.lookback = lookback
        self.model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
        self.is_fitted = False
        self.state_map = {}

    def fit_predict(self, df: pd.DataFrame) -> str:
        """
        Melatih HMM pada N candle terakhir dan memprediksi rezim saat ini.
        df harus memiliki kolom: open, high, low, close
        """
        if len(df) < self.n_states:
            return "UNKNOWN"

        # Gunakan N candle terakhir agar cepat dan relevan
        data = df.tail(self.lookback).copy()

        # Ekstrak fitur: Log Return & High-Low Range (Volatility)
        data['log_return'] = np.log(data['close'] / data['close'].shift(1)).fillna(0) + np.random.normal(0, 1e-6, len(data))
        data['range'] = ((data['high'] - data['low']) / data['close']).fillna(0) + np.random.normal(0, 1e-6, len(data))
        data = data.dropna()

        if len(data) < self.n_states:
            return "UNKNOWN"

        X = data[['log_return', 'range']].values

        try:
            # Fit & Predict
            self.model.fit(X)
            self.is_fitted = True
            hidden_states = self.model.predict(X)

            # Map states berdasarkan volatilitas (varians range)
            state_variances = []
            for i in range(self.n_states):
                state_data = data.iloc[hidden_states == i]
                if len(state_data) > 0:
                    var = state_data['range'].mean()
                    state_variances.append((i, var))
                else:
                    state_variances.append((i, 0))
            
            # Sort states by average range (volatility)
            state_variances.sort(key=lambda x: x[1])
            
            # Asumsi:
            # 0 (Varians terendah) = Low Volatility / Calm
            # 1 (Varians menengah) = Choppy / Sideways
            # 2 (Varians tertinggi) = High Volatility / News
            
            # Update state map
            if len(state_variances) == 3:
                self.state_map[state_variances[0][0]] = "LOW_VOLATILITY"
                self.state_map[state_variances[1][0]] = "CHOPPY"
                self.state_map[state_variances[2][0]] = "HIGH_VOLATILITY"
            elif len(state_variances) == 2:
                self.state_map[state_variances[0][0]] = "LOW_VOLATILITY"
                self.state_map[state_variances[1][0]] = "HIGH_VOLATILITY"
            else:
                self.state_map[state_variances[0][0]] = "UNKNOWN"

            # Prediksi candle terakhir
            current_state = hidden_states[-1]
            return self.state_map.get(current_state, "UNKNOWN")

        except Exception as e:
            logger.error(f"[HMM] Error fitting model: {e}")
            return "UNKNOWN"

def detect_market_regime(df_m15: pd.DataFrame) -> str:
    """Wrapper function untuk dipanggil dari scalping_bot.py"""
    detector = RegimeDetector(n_states=3, lookback=500)
    return detector.fit_predict(df_m15)
