"""Quick fix to test chart display with daily data."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import CONFIG
from data.fetcher import DataFetcher
from data.features import FeatureEngine
from data.processor import DataProcessor

def test_chart_pipeline():
    """Test the full chart data pipeline."""
    print("=" * 60)
    print("CHART PIPELINE TEST")
    print("=" * 60)
    
    stock = "000001"
    fetcher = DataFetcher()
    feature_engine = FeatureEngine()
    processor = DataProcessor()
    
    print(f"\nFetching 1m data for {stock}...")
    df_1m = fetcher.get_history(stock, interval="1m", bars=240, use_cache=True, update_db=False, allow_online=True)
    print(f"  1m data: {len(df_1m) if df_1m is not None else 'None'} bars")
    
    print(f"\nFetching 1d data for {stock}...")
    df_1d = fetcher.get_history(stock, interval="1d", bars=100, use_cache=True, update_db=False, allow_online=True)
    print(f"  1d data: {len(df_1d) if df_1d is not None else 'None'} bars")
    
    if df_1d is not None and not df_1d.empty:
        print(f"\nTesting feature engineering...")
        try:
            df_features = feature_engine.create_features(df_1d)
            print(f"  Features created: {df_features.shape}")
            print(f"  Columns: {len(df_features.columns)}")
            
            # Check for NaN values
            nan_count = df_features.isna().sum().sum()
            print(f"  NaN count: {nan_count}")
            
            # Test sequence preparation
            feature_cols = feature_engine.get_feature_columns()
            print(f"  Feature columns: {len(feature_cols)}")
            
            if len(df_features) >= 60:
                sequence = processor.prepare_inference_sequence(df_features, feature_cols)
                print(f"  Sequence shape: {sequence.shape}")
                print(f"  Sequence ready: YES")
            else:
                print(f"  Sequence ready: NO (need 60 bars, have {len(df_features)})")
                
        except Exception as e:
            print(f"  Feature engineering failed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_chart_pipeline()
