"""电压安全神经网络回归：分组划分、训练、验证、按原顺序预测。

运行: python train_voltage.py --data-dir ../../电压安全评估案例数据/电压安全评估案例数据
所有统计量仅用训练集拟合。验证集用于选模型，不作为独立测试成绩。
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from threadpoolctl import threadpool_limits
import joblib


def metrics(y, pred):
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'R2': float(r2_score(y, pred)),
            'P95_AE': float(np.quantile(np.abs(y-pred), .95)),
            'MAX_AE': float(np.max(np.abs(y-pred)))}


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[2]
    parser.add_argument('--data-dir', type=Path, default=base/'电压安全评估案例数据'/'电压安全评估案例数据')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[1]/'结果')
    parser.add_argument('--epochs', type=int, default=700)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    labeled = args.data_dir/'Dataset.csv'
    unlabeled = args.data_dir/'Test_data_unlabeled.csv'
    data = np.loadtxt(labeled, delimiter=',')
    test = np.loadtxt(unlabeled, delimiter=',', ndmin=2)
    assert data.shape == (17696, 86) and test.shape == (10, 85)
    assert np.isfinite(data).all() and np.isfinite(test).all()
    X, y = data[:, :-1], data[:, -1]
    unique, groups, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    lo, hi = np.full(len(unique), np.inf), np.full(len(unique), -np.inf)
    np.minimum.at(lo, groups, y)
    np.maximum.at(hi, groups, y)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
    assert not set(groups[tr]) & set(groups[va])
    sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(y[tr, None])
    xtr, xva = sx.transform(X[tr]), sx.transform(X[va])
    ytr = sy.transform(y[tr, None]).ravel()
    unscale = lambda p: sy.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()
    results, histories, models = [], [], {}
    baseline = np.full(len(va), y[tr].mean())
    results.append({'name': '训练集均值', 'hidden': [], 'best_epoch': 0, 'epochs_run': 0, 'validation': metrics(y[va], baseline)})
    ridge = Ridge(alpha=1.0).fit(xtr, ytr)
    results.append({'name': '岭回归', 'hidden': [], 'best_epoch': 0, 'epochs_run': 0, 'validation': metrics(y[va], unscale(ridge.predict(xva)))})
    for hidden in [(64, 32), (128, 64)]:
        name = 'MLP-' + '-'.join(map(str, hidden))
        model = MLPRegressor(hidden_layer_sizes=hidden, activation='relu', solver='adam',
                             alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                             max_iter=1, shuffle=True, random_state=42, early_stopping=False)
        best, best_epoch, best_model = float('inf'), 0, None
        t0 = time.time()
        for epoch in range(1, args.epochs+1):
            model.partial_fit(xtr, ytr)
            pt, pv = unscale(model.predict(xtr)), unscale(model.predict(xva))
            train_rmse = float(np.sqrt(mean_squared_error(y[tr], pt)))
            val_rmse = float(np.sqrt(mean_squared_error(y[va], pv)))
            histories.append({'model': name, 'epoch': epoch, 'train_RMSE': train_rmse, 'validation_RMSE': val_rmse})
            if val_rmse < best - 1e-6:
                best, best_epoch, best_model = val_rmse, epoch, copy.deepcopy(model)
            if epoch % 50 == 0:
                print(f'{name} epoch={epoch} train={train_rmse:.6f} val={val_rmse:.6f} best={best:.6f} sec={time.time()-t0:.1f}', flush=True)
            if epoch-best_epoch >= 80:
                break
        models[name] = best_model
        record = {'name': name, 'hidden': list(hidden), 'best_epoch': best_epoch, 'epochs_run': epoch,
                  'parameters': int(sum(w.size for w in best_model.coefs_)+sum(b.size for b in best_model.intercepts_)),
                  'train': metrics(y[tr], unscale(best_model.predict(xtr))),
                  'validation': metrics(y[va], unscale(best_model.predict(xva)))}
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    chosen = min([r for r in results if r['hidden']], key=lambda r: r['validation']['RMSE'])
    model = models[chosen['name']]
    prediction = unscale(model.predict(sx.transform(test)))
    pval = unscale(model.predict(xva))
    bundle = {'model': model, 'x_scaler': sx, 'y_scaler': sy, 'train_indices': tr,
              'validation_indices': va, 'selected': chosen, 'feature_count': 85}
    joblib.dump(bundle, out/'voltage_mlp.joblib')
    # 重新加载，核对模型保存前后的输出。
    loaded = joblib.load(out/'voltage_mlp.joblib')
    check = loaded['y_scaler'].inverse_transform(loaded['model'].predict(loaded['x_scaler'].transform(test)).reshape(-1,1)).ravel()
    assert np.allclose(check, prediction, rtol=0, atol=1e-12)
    pd.DataFrame({'sample_id': np.arange(1, 11), 'predicted_voltage_deviation': prediction}).to_csv(out/'测试样本预测.csv', index=False, encoding='utf-8-sig', float_format='%.9f')
    np.savetxt(out/'测试样本预测_仅数值.txt', prediction, fmt='%.9f')
    pd.DataFrame({'original_row': va+1, 'true': y[va], 'predicted': pval, 'residual_pred_minus_true': pval-y[va]}).to_csv(out/'验证集预测.csv', index=False, encoding='utf-8-sig')
    split = np.full(len(y), 'train', dtype=object); split[va] = 'validation'
    pd.DataFrame({'original_row': np.arange(1, len(y)+1), 'input_group': groups, 'split': split}).to_csv(out/'样本划分.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(histories).to_csv(out/'训练曲线.csv', index=False, encoding='utf-8-sig')
    summary = {'seed': 42, 'n_samples': len(y), 'n_features': X.shape[1], 'n_train': len(tr), 'n_validation': len(va),
        'n_unique_inputs': len(unique), 'duplicate_extra_rows': len(y)-len(unique), 'duplicate_groups': int(sum(counts>1)),
        'inconsistent_label_groups': int(sum(hi-lo>1e-8)), 'max_same_input_label_range': float(max(hi-lo)),
        'constant_feature_columns_1based': (np.where(np.ptp(X[tr],axis=0)==0)[0]+1).tolist(),
        'n_topologies': len(np.unique(X[:,:80],axis=0)),
        'train_topologies': len(np.unique(X[tr,:80],axis=0)), 'validation_topologies': len(np.unique(X[va,:80],axis=0)),
        'train_groups': len(np.unique(groups[tr])), 'validation_groups': len(np.unique(groups[va])),
        'y_min': float(y.min()), 'y_max': float(y.max()), 'y_mean': float(y.mean()), 'y_std': float(y.std()),
        'train_y_mean': float(y[tr].mean()), 'val_y_mean': float(y[va].mean()),
        'test_input_matches_labeled': [int(np.sum(np.all(X==v,axis=1))) for v in test],
        'test_columns_outside_train_range_1based': (np.where(np.any((test<X[tr].min(axis=0))|(test>X[tr].max(axis=0)),axis=0))[0]+1).tolist(),
        'results': results, 'selected': chosen, 'predictions': prediction.tolist(),
        'max_epochs': args.epochs, 'patience': 80, 'min_delta': 1e-6,
        'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [labeled, unlabeled]},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scikit-learn': sklearn.__version__, 'matplotlib': matplotlib.__version__}}
    (out/'实验指标.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'], 'axes.unicode_minus':False, 'font.size':10})
    h = pd.DataFrame(histories)
    fig, axs = plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    for ax, r in zip(axs, [r for r in results if r['hidden']]):
        z=h[h.model==r['name']]
        ax.plot(z.epoch,z.train_RMSE,label='训练集',color='#246396',lw=1.2)
        ax.plot(z.epoch,z.validation_RMSE,label='验证集',color='#b76128',lw=1.2)
        ax.axvline(r['best_epoch'],color='#777777',ls='--',lw=.8,label='保留轮次')
        ax.set(xlabel='训练轮次',ylabel='RMSE（原始指标尺度）',title=r['name']); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'训练过程.png',dpi=220);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    axs[0].scatter(y[va],pval,s=6,alpha=.3,color='#246396',edgecolors='none')
    lim=[min(y[va].min(),pval.min())-.02,max(y[va].max(),pval.max())+.02]
    axs[0].plot(lim,lim,'--',color='#b76128',lw=1)
    axs[0].set(xlabel='真实值',ylabel='预测值',title='验证集预测与真实值',xlim=lim,ylim=lim)
    axs[1].hist(pval-y[va],bins=55,color='#246396',edgecolor='white',linewidth=.3)
    axs[1].axvline(0,color='#b76128',ls='--',lw=1)
    axs[1].set(xlabel='残差（预测值 − 真实值）',ylabel='样本数',title='验证集残差分布')
    for ax in axs: ax.grid(alpha=.2)
    fig.savefig(out/'验证效果.png',dpi=220);plt.close(fig)
    print('SELECTED',json.dumps(chosen,ensure_ascii=False),flush=True)
    print('PREDICTIONS',prediction,flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
