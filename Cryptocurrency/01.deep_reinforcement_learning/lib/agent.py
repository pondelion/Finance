import random
import numpy as np 


class Agent:

    def __init__(self, model, batch_size=32, discount_factor=0.95, eps=0.05):
        self._model = model
        self._memory = []
        self._batch_size = batch_size  # 1回のリプレイあたりで学習させるデータ数
        self._discount_factor = discount_factor
        self._eps = 0.05  # ランダムにアクションを選択する確率

    def replay(self):
        """
        過去の状態・行動・etcの履歴からランダムにバッチサイズ分ランダムサンプリングし、
        それぞれに対しQ値を更新し、入力を状態、出力を更新したQ値でモデルを学習させる。
        """
        batch = random.sample(self._memory, min(len(self._memory), self._batch_size))
        for state, action, reward, next_state, done, next_valid_actions in batch:
            q = reward
            if not done:
                q += self._discount_factor * np.nanmax(self.get_q_values(next_state, next_valid_actions))
            self._model.fit(state=state, action=action, action_q=q)

    def get_q_values(self, state, valid_actions):
        """
        指定した状態をモデルに入力してpredictを行いQ値を求め、
        有効なアクションに対するQ値を返す。
        非有効なアクションに対するQ値はnp.nan。
        """

        q = self._model.predict(state)
        q_valid = [np.nan] * len(q)
        for action in valid_actions:
            q_valid[action] = q[action]
        return q_valid

    def remember(self, state, action, reward, next_state, 
                 done, next_valid_actions):
        self._memory.append((state, action, reward, next_state, 
                            done, next_valid_actions))

    def act(self, state, valid_actions):
        action = None
        if np.random.random() > self._eps:
            q = self.get_q_values(state, valid_actions)
            if np.nanmin(q) != np.nanmax(q):
                action = np.nanargmax(q)
        else:
            action = random.sample(valid_actions, 1)[0]
        
        return action 
