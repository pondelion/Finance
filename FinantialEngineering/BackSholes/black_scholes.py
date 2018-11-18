import numpy as np
from scipy.stats import norm
import functools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def N(x):
    """
    標準正規分布関数
    """
    
    return norm.cdf(x=x, loc=0, scale=1)


def calloption_price(t, sig, s, k, r):
    """
    ブラックショールズモデル式からコールオプション計算
    Parameters
    -----------
    t : 満期
    sig : 株価変化率の標準偏差
    s : 現在の株価
    k : 権利行使価格
    r : 安定資産利回り
    
    Return
    -----------
    コールオプション価格
    """
    
    d1 = (np.log(s/k) + (r + (sig**2)/2)*t) / (sig*np.sqrt(t))
    d2 = (np.log(s/k) + (r - (sig**2)/2)*t) / (sig*np.sqrt(t))
    return s*N(d1) - k*np.exp(r*t)*N(d2)


if __name__ == '__main__':
    T = 10  # 満期 
    sigma = np.arange(0, 2.5, 0.01) # 株価の標準偏差
    S = 2000  # 現在の株価
    K = 2400  # 権利行使価格
    r = np.arange(0, 0.15, 0.005)  # 安定資産利回り
    
    SIGMA, R = np.meshgrid(sigma, r)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    calloption_prices = np.array([calloption_price(t=T, sig=_sig, s=S, k=K, r=_r) for _sig, _r in zip(SIGMA.flatten(), R.flatten())]).reshape(len(r), len(sigma))
    ax.plot_wireframe(SIGMA, R, calloption_prices)
    ax.set_xlabel('sigma')
    ax.set_ylabel('r')
    ax.set_zlabel('calloption price')

    plt.show()