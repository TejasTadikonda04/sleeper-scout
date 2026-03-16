import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import joblib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

@dataclass
class ModelMetrics:
    r2: float
    spearman: float
    mae: float
    rmse: float
    n_test: int

class FeatureEngineer:
    """Handles feature creation from raw combine/college data"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features"""
        df_eng = df.copy()
        
        # Physical traits
        df_eng['bmi'] = df_eng['wt'] / ((df_eng['ht_in']/12)**2)
        df_eng['speed_score'] = df_eng['wt'] * (100 / df_eng['forty'])**3
        
        # Explosiveness
        df_eng['athletic_score'] = (df_eng['vertical'] + df_eng['broad_jump']/12)
        
        # Agility normalized
        df_eng['agility_index'] = (df_eng['cone'] + df_eng['shuttle']) / 2
        
        return df_eng
    
    @staticmethod
    def define_position_groups(pos: str) -> str:
        """Map positions to modeling groups"""
        pos_map = {
            'QB': 'QB',
            'RB': 'SKILL', 'WR': 'SKILL', 'TE': 'SKILL',
            'OT': 'OL', 'OG': 'OL', 'C': 'OL',
            'DT': 'DL', 'DE': 'DL', 'EDGE': 'DL',
            'LB': 'LB', 'ILB': 'LB', 'OLB': 'LB',
            'CB': 'DB', 'S': 'DB', 'FS': 'DB', 'SS': 'DB'
        }
        return pos_map.get(pos, 'OTHER')

class AVSharePredictor:
    """Main XGBoost model for predicting AV Share by position group"""

    BASE_NUMERIC = ['ht_in', 'wt', 'forty', 'bench', 'vertical',
                    'broad_jump', 'cone', 'shuttle', 'bmi',
                    'speed_score', 'athletic_score', 'agility_index']

    def __init__(self, extra_numeric_features: Optional[List[str]] = None):
        self.models: Dict[str, Pipeline] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.position_groups = ['QB', 'SKILL', 'OL', 'DL', 'LB', 'DB']
        self.extra_numeric_features = extra_numeric_features or []
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for modeling. Uses combine stats + any extra_numeric_features (e.g. college)."""
        df_eng = FeatureEngineer.create_features(df)
        df_eng['pos_group'] = df_eng['pos'].apply(FeatureEngineer.define_position_groups)
        
        numeric_features = self.BASE_NUMERIC + [
            f for f in self.extra_numeric_features if f in df_eng.columns
        ]
        categorical_features = ['pos_group']
        
        # Clean data (keep draft_year for time-based train/val split)
        df_eng = df_eng[numeric_features + categorical_features + ['av_share', 'draft_year']].dropna()
        
        return df_eng, numeric_features + categorical_features
    
    def train_position_models(self, df: pd.DataFrame) -> Dict[str, ModelMetrics]:
        """Train separate XGBoost model per position group"""
        df_eng, feature_cols = self.prepare_features(df)
        self.feature_names = feature_cols
        
        metrics = {}
        
        # Time series split (train on past years, validate future)
        tscv = TimeSeriesSplit(n_splits=3)
        
        for pos_group in self.position_groups:
            pos_df = df_eng[df_eng['pos_group'] == pos_group]
            
            if len(pos_df) < 50:  # Skip small groups
                continue
                
            X = pos_df[feature_cols]
            y = pos_df['av_share']
            
            # Time-based train/val split (80/20 by draft_year)
            train_mask = pos_df['draft_year'] <= pos_df['draft_year'].quantile(0.8)
            X_train, X_val = X[train_mask], X[~train_mask]
            y_train, y_val = y[train_mask], y[~train_mask]
            
            # Pipeline: preprocessor turns pos_group into numeric; XGBoost needs numeric-only eval_set
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), [f for f in feature_cols if f != 'pos_group']),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), ['pos_group'])
            ])
            X_val_transformed = preprocessor.fit(X_train).transform(X_val)

            model = Pipeline([
                ('preproc', preprocessor),
                ('xgb', XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.03,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=42,
                    early_stopping_rounds=50,
                ))
            ])
            model.fit(
                X_train, y_train,
                xgb__eval_set=[(X_val_transformed, y_val)],
                xgb__verbose=False,
            )
            
            # Evaluate
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            spearman, _ = spearmanr(y_val, y_pred)
            
            metrics[pos_group] = ModelMetrics(
                r2=r2, spearman=spearman, mae=np.mean(np.abs(y_val - y_pred)),
                rmse=np.sqrt(np.mean((y_val - y_pred)**2)), n_test=len(y_val)
            )
            
            # Save model
            self.models[pos_group] = model
            
        return metrics
    
    def predict_av_share(self, player_data: pd.DataFrame) -> pd.Series:
        """Predict AV Share for new players"""
        player_data = FeatureEngineer.create_features(player_data)
        player_data['pos_group'] = player_data['pos'].apply(FeatureEngineer.define_position_groups)
        
        predictions = np.zeros(len(player_data))
        
        for pos_group, model in self.models.items():
            mask = player_data['pos_group'] == pos_group
            if mask.sum() > 0:
                pred = model.predict(player_data[mask])
                predictions[mask] = pred
        
        return pd.Series(predictions, index=player_data.index)
    
    def save_model(self, path: str):
        """Save all position models"""
        joblib.dump({
            'models': self.models,
            'feature_names': self.feature_names,
            'position_groups': self.position_groups,
            'extra_numeric_features': getattr(self, 'extra_numeric_features', []),
        }, path)

    @classmethod
    def load_model(cls, path: str):
        """Load trained model"""
        data = joblib.load(path)
        model = cls(extra_numeric_features=data.get('extra_numeric_features'))
        model.models = data['models']
        model.feature_names = data['feature_names']
        model.position_groups = data['position_groups']
        return model

class AVShareForecaster:
    """Forecast year-by-year expected AV Share"""
    
    def __init__(self, predictor: AVSharePredictor):
        self.predictor = predictor
        
    def forecast_5yr_av_share(self, player_data: pd.DataFrame, class_total_av: float = None) -> pd.DataFrame:
        """Predict AV Share trajectory over 5 years"""
        # Get baseline AV Share prediction
        av_share_5yr = self.predictor.predict_av_share(player_data)
        
        # Apply aging curve adjustment (simplified)
        years = [1, 2, 3, 4, 5]
        aging_factors = [0.7, 0.9, 1.1, 1.0, 0.9]  # Peak year 3
        
        result = pd.DataFrame(index=player_data.index)
        for i, year in enumerate(years):
            yearly_share = av_share_5yr * aging_factors[i]
            result[f'year_{i+1}'] = yearly_share
        result['total_5yr_av_share'] = av_share_5yr
            
        return result
