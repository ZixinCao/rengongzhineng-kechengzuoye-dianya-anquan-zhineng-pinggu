"""加载已训练模型并按输入 CSV 的行顺序预测，不需要重新训练。"""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, default=root/'结果'/'voltage_mlp.joblib')
parser.add_argument('--input', type=Path, default=root.parent/'电压安全评估案例数据'/'电压安全评估案例数据'/'Test_data_unlabeled.csv')
parser.add_argument('--output', type=Path, default=root/'结果'/'测试样本预测_复核.csv')
args = parser.parse_args()
bundle = joblib.load(args.model)
X = np.loadtxt(args.input, delimiter=',', ndmin=2)
if X.shape[1] != bundle['feature_count'] or not np.isfinite(X).all():
    raise ValueError('输入必须为无表头的 85 列有限数值 CSV。')
pred = bundle['y_scaler'].inverse_transform(bundle['model'].predict(bundle['x_scaler'].transform(X)).reshape(-1,1)).ravel()
pd.DataFrame({'sample_id': np.arange(1,len(X)+1), 'predicted_voltage_deviation': pred}).to_csv(args.output,index=False,encoding='utf-8-sig',float_format='%.9f')
print(args.output)
