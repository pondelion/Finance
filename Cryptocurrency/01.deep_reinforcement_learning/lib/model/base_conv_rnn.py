from .base_model import BaseKerasModel
import keras


class BaseConvRNN(BaseKerasModel):

	def init(self):
		self._model_tag	= 'ConvRNN'

	def _build_model(self, RNNLayer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size=3, use_pool=False, activation='relu'):

		model = keras.models.Sequential()
		model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))

		for i in range(len(conv_n_hidden)):
			model.add(keras.layers.Conv1D(conv_n_hidden[i], kernel_size=conv_kernel_size, 
				activation=activation, use_bias=True))
			if use_pool:
				model.add(keras.layers.MaxPooling1D(pool_size=2))
		m = len(RNN_n_hidden)
		for i in range(m):
			model.add(RNNLayer(RNN_n_hidden[i],
				return_sequences=(i<m-1)))
		for i in range(len(dense_units)):
			model.add(keras.layers.Dense(dense_units[i], activation=activation))

		model.add(keras.layers.Dense(self.n_actions, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self._model = model
		self._model_name = self._model_tag + str(conv_n_hidden) + str(RNN_n_hidden) + str(dense_units)
