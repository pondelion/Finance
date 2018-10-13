import numpy as np 


class Market:
    """
    
    action     
        0 : ノーポジ
        1 : 買い
        2 : 売り
        3 : 保持
    """

    def __init__(self, prices, window, model, cost, asset=300000, reward_data_idx=0):
        self._prices            = prices    # 価格の時系列データ。T（期間)×N（時系列データ種類数)
        self._no_posi_flg       = True      # ノーポジならTrue
        self._window            = window    # 価格のウィンドウ幅。モデルに入力する価格データ数。 
        self._model             = model
        self._cost              = cost      # 決済コスト
        self._reward_data_idx   = reward_data_idx     # 報酬計算に使用する時系列データのインデックス。ex. 1ならばself._prices[:,1]のデータを報酬計算に使用
        self._t                 = window
        self._asset				= asset
        self._init_asset		= asset

    def get_state(self, t):
        """
        時刻tにおける状態(価格の過去window分のスライスデータ)を取得する
        """
        try:
            state = self._prices[t - self._window: t, :]
        except Exception:
            raise ValueError('Index is wrong, slicing prices failed.')

        # スライスデータの標準化
        means = np.mean(state, axis=0)
        state = (state/means - 1.)*100

        return state 

    def get_valid_actions(self):
        if self._no_posi_flg:
            return [0, 1]    # ノーポジ、買い
        else:
            return [2, 3]    # 売り、保持

    def get_reward(self, t, action):
        reward = None
        if action == 0:  # ノーポジ
            reward = 0
        elif action == 1:  # 買い
            reward = self._prices[t+1, self._reward_data_idx] - self._prices[t, self._reward_data_idx] - self._cost
        elif action == 2:  # 売り
            reward = -(self._prices[t+1, self._reward_data_idx] - self._prices[t, self._reward_data_idx]) - self._cost
        elif action == 3:  # 保持
            reward = self._prices[t+1, self._reward_data_idx] - self._prices[t, self._reward_data_idx]
        else:
            raise ValueError('action error')

        return reward 

    def step(self, action):

        reward = self.get_reward(self._t, action)

        if (action == 0) or (action == 2):  # ノーポジor売りならばノーポジフラグを立てる
            self._no_posi_flg = True 
        else:
            self._no_posi_flg = False

        if action == 1:  # 買い
        	self._asset -= (self._prices[self._t, self._reward_data_idx] + self._cost)
        elif action == 2:  # 売り
        	self._asset += (self._prices[self._t, self._reward_data_idx] - self._cost)

        self._t += 1

        # アクションをとった後(タイムステップを1進めた)の状態を取得
        state = self.get_state(self._t)

        done = self._prices.shape[0] == (self._t + 1)

        return state, reward, done, self.get_valid_actions(), self._asset

    def reset(self):
        self._no_posi_flg = True
        self._t = self._window
        self._asset = self._init_asset

        return self.get_state(self._t), self.get_valid_actions()
