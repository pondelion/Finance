from .base_model import BaseKerasModel
from .base_rnn import BaseRNN
import keras


class LSTM(BaseRNN):

	def init(self):
		self._model_tag	= 'LSTM'

	def build_model(self, n_hidden, dense_units, learning_rate, activation='relu'):
		Layer = keras.layers.LSTM
		self._build_model(Layer, n_hidden, dense_units, learning_rate, activation)
