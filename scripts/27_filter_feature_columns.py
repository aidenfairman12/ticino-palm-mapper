import pandas as pd

def filter_columns(fname: str )-> None:
    dfa = pd.read_csv(fname)
    feature_drops_a = ['chm_point', 'chm_mean_r1.0', 'chm_std_r1.0', 'chm_max_r1.0', 'chm_min_r1.0', 'chm_peak_ratio_r1.0',
                     'chm_skew_r1.0', 'chm_kurtosis_r1.0', 'chm_local_roughness_r1.0', 'chm_asymmetry_r1.0']
    dfa.drop(columns=feature_drops_a, inplace=True)
    dfa.to_csv("classical_features_native_10cm_no_chm_r1.csv", index=False)
    
    dfb = pd.read_csv(fname)
    feature_drops_b = ['chm_peak_ratio_r1.0', 'chm_peak_ratio_r3.0', 'chm_peak_ratio_r5.0', 'chm_skew_r1.0', 'chm_skew_r3.0',
                    'chm_skew_r5.0', 'chm_kurtosis_r1.0', 'chm_kurtosis_r3.0', 'chm_kurtosis_r5.0', 'chm_local_roughness_r1.0',
                    'chm_local_roughness_r3.0', 'chm_local_roughness_r5.0', 'chm_asymmetry_r1.0', 'chm_asymmetry_r3.0', 'chm_asymmetry_r5.0']
    dfb.drop(columns=feature_drops_b, inplace=True)
    dfb.to_csv('classical_features_native_10cm_no_shape.csv', index=False)
    
def main() -> None:
    filter_columns("classical_features_native_10cm.csv")
    
    
if __name__ == "__main__":
    main()