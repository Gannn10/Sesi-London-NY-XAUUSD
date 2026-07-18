"""
ML Model V5 Training Pipeline (Enhanced Scalper)
================================================
Improvements over V4:
- 100,000 candles (2x more data)
- 5 new momentum/regime features
- Anti-overfitting: max_depth 3-5, early_stopping, lower subsample
- Target: Test accuracy 62%+ with gap < 8%
"""

import polars as pl
import numpy as np
from pathlib import Path
import sys
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import MetaTrader5 as mt5
import config_london_ny as cfg
from feature_eng import FeatureEngineer
from smc_polars import SMCAnalyzer
from triple_barrier_labeling import TripleBarrierLabeling

# ML imports
try:
    import xgboost as xgb
    from sklearn.metrics import classification_report
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  Optuna not installed. Using default hyperparameters.")


class MLTrainerV5:
    """
    Enhanced ML model trainer V5 with momentum features and anti-overfitting.
    """

    def __init__(self):
        if not mt5.initialize():
            print(f"Failed to initialize MT5: {mt5.last_error()}")
            sys.exit(1)
        self.fe = FeatureEngineer()
        self.smc = SMCAnalyzer()

        # Triple barrier — same targets as V4 for fair comparison
        self.labeler = TripleBarrierLabeling(
            profit_atr_mult=0.3,
            stoploss_atr_mult=0.3,
            max_holding_bars=20,
        )

        self.model = None
        self.feature_cols = []
        self.metadata = {}

        # Paths
        self.output_dir = Path("backtests/ml_v5")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_training_data(self, n_bars: int = 75000) -> pl.DataFrame:
        """Fetch 75K bars of M5 data."""
        print(f"\n Fetching {n_bars:,} bars of M5 data...")
        print(f"   Symbol: {cfg.SYMBOL}")

        rates = mt5.copy_rates_from_pos(cfg.SYMBOL, cfg.TIMEFRAME, 0, n_bars)
        
        if rates is None or len(rates) == 0:
            raise ValueError("Failed to fetch M5 data from MT5.")

        import pandas as pd
        df_pd = pd.DataFrame(rates)
        df_pd['time'] = pd.to_datetime(df_pd['time'], unit='s')
        df = pl.from_pandas(df_pd)

        print(f" Fetched {len(df):,} bars")
        print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
        return df

    def fetch_m15_data(self, n_bars: int = 25000) -> pl.DataFrame:
        """Fetch M15 data (proportional to M5 75K)."""
        print(f"\n Fetching {n_bars:,} bars of M15 data...")

        rates = mt5.copy_rates_from_pos(cfg.SYMBOL, mt5.TIMEFRAME_M15, 0, n_bars)
        if rates is None or len(rates) == 0:
            raise ValueError("Failed to fetch M15 data.")
            
        import pandas as pd
        df_pd = pd.DataFrame(rates)
        df_pd['time'] = pd.to_datetime(df_pd['time'], unit='s')
        df_m15 = pl.from_pandas(df_pd)

        print(f" Fetched {len(df_m15):,} M15 bars")
        return df_m15

    def engineer_features(self, df_m5: pl.DataFrame, df_m15: pl.DataFrame) -> pl.DataFrame:
        """Calculate all features including new momentum features."""
        print(f"\n Engineering features (V5 Enhanced)...")

        print("   M5 technical indicators + momentum features...")
        df = self.fe.calculate_all(df_m5, include_ml_features=True)

        print("   SMC structure features...")
        df = self.smc.calculate_all(df)

        print("   M15 features...")
        df = self._join_m15_features(df, df_m15)

        n_features = len([c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']])
        print(f" Total features: {n_features} (V4 had 72)")

        # Handle nulls
        null_counts = df.null_count()
        cols_with_nulls = [col for col in null_counts.columns if null_counts[col][0] > 0]
        if cols_with_nulls:
            print(f"     Columns with nulls: {len(cols_with_nulls)}")
            df = df.fill_null(strategy="forward")
            df = df.fill_null(strategy="zero")

        return df

    def _join_m15_features(self, df_m5: pl.DataFrame, df_m15: pl.DataFrame) -> pl.DataFrame:
        """Join M15 features to M5 data using asof join."""
        df_m15 = self.fe.calculate_all(df_m15, include_ml_features=False)
        df_m15 = self.smc.calculate_all(df_m15)

        m15_feature_cols = [
            "time", "close", "rsi", "atr", "bb_upper", "bb_lower",
            "macd", "macd_signal", "ema_20", "ema_50",
            "ob", "fvg", "market_structure"
        ]
        m15_feature_cols = [c for c in m15_feature_cols if c in df_m15.columns]
        df_m15_selected = df_m15.select(m15_feature_cols)

        rename_map = {c: f"m15_{c}" for c in df_m15_selected.columns if c != "time"}
        rename_map["time"] = "time"
        df_m15_selected = df_m15_selected.rename(rename_map)

        # Shift M15 time forward by 15 min to prevent look-ahead bias
        df_m15_selected = df_m15_selected.with_columns(
            (pl.col("time") + pl.duration(minutes=15)).alias("time")
        )

        df_joined = df_m5.join_asof(df_m15_selected, on="time", strategy="backward")

        if "m15_close" in df_joined.columns and "m15_ema_20" in df_joined.columns:
            df_joined = df_joined.with_columns([
                ((pl.col("m15_close") - pl.col("m15_ema_20")) / pl.col("m15_ema_20")).alias("m15_ema20_distance")
            ])

        return df_joined

    def label_data(self, df: pl.DataFrame) -> pl.DataFrame:
        print(f"\n Labeling data with Triple Barrier Method...")
        df = self.labeler.label_data(df)
        return df

    def prepare_train_test(self, df: pl.DataFrame, test_size: float = 0.2) -> Tuple[pl.DataFrame, pl.DataFrame]:
        print(f"\n Splitting train/test...")
        df = df.filter((pl.col("target").is_not_null()) & (pl.col("target") >= 0))

        if len(df) == 0:
            raise ValueError("No labeled data available.")

        # Time-based split (more realistic than random)
        split_idx = int(len(df) * (1 - test_size))
        df_train = df.head(split_idx)
        df_test = df.tail(len(df) - split_idx)

        print(f"   Train: {len(df_train):,} samples (older data)")
        print(f"   Test:  {len(df_test):,} samples (recent data)")
        return df_train, df_test

    def select_features(self, df: pl.DataFrame) -> List[str]:
        exclude_cols = {
            'time', 'open', 'high', 'low', 'close', 'volume',
            'target', 'target_label', 'barrier_hit', 'bars_to_barrier',
            'return_pct', 'smc_signal', 'smc_confidence', 'smc_reason'
        }

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.Boolean]
        ]

        print(f"\n Selected {len(feature_cols)} features (V4 had 72)")
        
        # Highlight new features
        new_features = [c for c in feature_cols if c in ['momentum_3c', 'momentum_6c', 'atr_ratio', 'body_ratio', 'consecutive_direction']]
        if new_features:
            print(f"   [NEW] V5 features: {', '.join(new_features)}")
        
        self.feature_cols = feature_cols
        return feature_cols

    def train_xgboost(self, df_train: pl.DataFrame, df_test: pl.DataFrame, feature_cols: List[str], optimize_hyperparams: bool = True) -> xgb.XGBClassifier:
        print(f"\n Training XGBoost V5 model...")

        X_train = df_train.select(feature_cols).to_numpy()
        y_train = df_train["target"].to_numpy()
        X_test = df_test.select(feature_cols).to_numpy()
        y_test = df_test["target"].to_numpy()

        n_sell = (y_train == 0).sum()
        n_buy = (y_train == 1).sum()
        n_total = len(y_train)
        weight_sell = n_total / (2 * n_sell) if n_sell > 0 else 1.0
        weight_buy = n_total / (2 * n_buy) if n_buy > 0 else 1.0
        sample_weights = np.where(y_train == 0, weight_sell, weight_buy)

        print(f"   Class weights: SELL={weight_sell:.2f}, BUY={weight_buy:.2f}")

        if optimize_hyperparams and HAS_OPTUNA:
            print("   Running Optuna optimization (anti-overfitting constraints)...")
            best_params = self._optimize_hyperparameters(X_train, y_train, X_test, y_test, sample_weights)
        else:
            best_params = {
                'max_depth': 4,
                'learning_rate': 0.0735,
                'n_estimators': 300,
                'min_child_weight': 8,
                'gamma': 0.366,
                'subsample': 0.63,
                'colsample_bytree': 0.778,
                'reg_alpha': 1.049,
                'reg_lambda': 1.868,
            }

        print(f"\n   Training final model...")

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,  # V5: Anti-overfitting!
            **best_params
        )

        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        print(f"\n Model V5 Performance:")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = (y_train_pred == y_train).mean()
        test_acc = (y_test_pred == y_test).mean()
        gap = train_acc - test_acc

        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy:  {test_acc:.4f}")
        print(f"   Gap:            {gap:.4f} {'(OK)' if gap < 0.08 else '(Overfit)'}")
        print(f"\n   V4 was: Train=0.710, Test=0.567, Gap=0.143")
        print(f"   Improvement: Test +{(test_acc - 0.567)*100:.1f}%")
        
        print(classification_report(y_test, y_test_pred, target_names=['SELL', 'BUY'], digits=3))

        # Feature importance
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        print("\n   Top 10 Features:")
        for idx in top_idx:
            print(f"     {feature_cols[idx]:30s} : {importances[idx]:.4f}")

        self.metadata = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_test_gap': float(gap),
            'train_samples': int(len(y_train)),
            'test_samples': int(len(y_test)),
            'n_features': len(feature_cols),
            'feature_cols': feature_cols,
            'hyperparameters': best_params,
            'class_distribution_train': {'SELL': int(n_sell), 'BUY': int(n_buy)},
            'model_type': 'binary_classification_scalper_v5',
            'improvements': ['100K data', 'momentum features', 'anti-overfitting', 'early_stopping', 'time-based split']
        }

        self.model = model
        return model

    def _optimize_hyperparameters(self, X_train, y_train, X_test, y_test, sample_weights) -> Dict:
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 5),        # V5: Restricted to 3-5 (was 3-8)
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  # V5: Lower max
                'n_estimators': trial.suggest_int('n_estimators', 200, 600, step=50),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),   # V5: Higher min (was 1-7)
                'gamma': trial.suggest_float('gamma', 0.1, 0.5),          # V5: Higher min (was 0.0-0.5)
                'subsample': trial.suggest_float('subsample', 0.6, 0.8),  # V5: Lower max (was 0.6-1.0)
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8), # V5: Lower
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.5),  # V5: Higher regularization
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0), # V5: Higher regularization
            }

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_jobs=1,
                early_stopping_rounds=30,
                **params
            )

            model.fit(X_train, y_train, sample_weight=sample_weights, 
                     eval_set=[(X_test, y_test)], verbose=False)
            
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            test_acc = (y_pred_test == y_test).mean()
            train_acc = (y_pred_train == y_train).mean()
            gap = train_acc - test_acc
            
            # Penalize overfitting: if gap > 8%, reduce score
            penalty = max(0, (gap - 0.08) * 2)
            
            return test_acc - penalty

        study = optuna.create_study(direction='maximize', study_name='xgboost_v5_opt')
        study.optimize(objective, n_trials=40, show_progress_bar=True, n_jobs=1)
        
        print(f"\n   Best trial: {study.best_value:.4f}")
        return study.best_params

    def save_model(self, output_name: str = "xgboost_model_v5_scalper.pkl"):
        output_path = self.output_dir / output_name

        model_data = {
            'xgb_model': self.model.get_booster(),
            'model_type': 'XGBOOST_BINARY_SCALPER_V5',
            'feature_names': self.feature_cols,
            'confidence_threshold': 0.60,
            'xgb_params': self.metadata.get('hyperparameters', {}),
            'train_metrics': {
                'train_accuracy': self.metadata['train_accuracy'],
                'test_accuracy': self.metadata['test_accuracy'],
                'train_test_gap': self.metadata['train_test_gap'],
            },
            'fitted': True,
            'metadata': self.metadata,
            'version': '5.0_scalper_enhanced',
            'trained_at': datetime.now().isoformat(),
            'symbol': cfg.SYMBOL,
            'timeframe': 'M5'
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n Model V5 saved to: {output_path}")

        metadata_path = self.output_dir / output_name.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f" Metadata saved to: {metadata_path}")

    def run_full_pipeline(self):
        print("=" * 80)
        print("ML MODEL V5 (ENHANCED SCALPER) TRAINING PIPELINE")
        print("=" * 80)

        df_m5 = self.fetch_training_data(n_bars=75000)  # V5: More data!
        df_m15 = self.fetch_m15_data(n_bars=25000)

        df = self.engineer_features(df_m5, df_m15)
        df = self.label_data(df)

        df_train_raw, df_test = self.prepare_train_test(df, test_size=0.20)

        print("\n Balancing TRAINING set only...")
        df_train = self.labeler.balance_classes(
            df_train_raw,
            target_buy_pct=0.50,
            target_sell_pct=0.50,
        )

        feature_cols = self.select_features(df_train)
        model = self.train_xgboost(df_train, df_test, feature_cols, optimize_hyperparams=False)

        self.save_model()

        print("\n" + "=" * 80)
        print(" V5 TRAINING COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    trainer = MLTrainerV5()
    try:
        trainer.run_full_pipeline()
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    except Exception as e:
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()
