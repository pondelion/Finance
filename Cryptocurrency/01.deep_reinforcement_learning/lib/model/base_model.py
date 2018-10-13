from abc import *
import numpy as np
import os
import pickle
import keras


class BaseKerasModel:

	def __init__(self, state_shape, n_actions, ):
		self.state_shape = state_shape
		self.n_actions = n_actions
		self._model_tag = 'None'
		self.attr2save = ['state_shape','n_actions','_model_tag']
		self.init()

	def init(self):
		# 継承クラスで行いたい任意初期化処理があればoverrideする
		pass

	@abstractmethod
	def build_model(self):
		# MUST : 継承クラスでkerasのモデルをself._modelとして初期化する。
		raise NotImplementedError()

	def fit(self, state, action, action_q):
		"""
		モデルの学習を行う。入力：状態、出力：各アクションのQ値

		Parameters:
		state		: 状態(価格時系列データのスライス)
		action  	: state状態においてとった行動
		action_q	: state状態においてactionをとったときの更新済みQ値
		"""
		q = self.predict(state)
		q[action] = action_q  # 対象の行動のQ値だけ更新する。
		
		self._model.fit(
			np.reshape(state, (1,) + self.state_shape),
			np.reshape(q, (1,) + (len(q),)),
			epochs=1, verbose=0
		)

	def predict(self, state):
		"""
		モデルから予測を行う。入力：状態、出力：各アクションのQ値

		Parameters:
		state		: 状態(価格時系列データのスライス)
		"""

		q = self._model.predict(
			np.reshape(state, (1,) + state.shape)
		)[0]

		return q 

	def save(self, fld):
		if not os.path.exists(fld):
			os.makedirs(fld)
		with open(os.path.join(fld, 'model.json'), 'w') as json_file:
			json_file.write(self._model.to_json())
		self._model.save_weights(os.path.join(fld, 'weights.hdf5'))

		attr = dict()
		for a in self.attr2save:
			attr[a] = getattr(self, a)
		pickle.dump(attr, open(os.path.join(fld, 'Qmodel_attr.pickle'),'wb'))

	def load(self, fld, learning_rate):
		json_str = open(os.path.join(fld, 'model.json')).read()
		self._model = keras.models.model_from_json(json_str)
		self._model.load_weights(os.path.join(fld, 'weights.hdf5'))
		self._model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

		attr = pickle.load(open(os.path.join(fld, 'Qmodel_attr.pickle'), 'rb'))
		for a in attr:
			setattr(self, a, attr[a])
