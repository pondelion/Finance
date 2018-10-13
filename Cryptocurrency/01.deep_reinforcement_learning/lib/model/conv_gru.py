from .base_model import BaseKerasModel
from .base_conv_rnn import BaseConvRNN
import keras


class ConvGRU(BaseConvRNN):

	def init(self):
		self._model_tag	= 'ConvGRU'

	def build_model(self, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size=3, use_pool=False, activation='relu'):
		Layer = keras.layers.GRU
		self._build_model(Layer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size, use_pool, activation)
