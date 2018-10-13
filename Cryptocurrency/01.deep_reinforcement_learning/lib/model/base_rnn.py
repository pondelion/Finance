from .base_model import BaseKerasModel
import keras


class BaseRNN(BaseKerasModel):

	def init(self):
		self._model_tag	= 'RNN'

	def _build_model(self, Layer, n_hidden, dense_units, learning_rate, activation='relu'):

		model = keras.models.Sequential()
		model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))
		m = len(n_hidden)
		for i in range(m):
			model.add(Layer(n_hidden[i],
				return_sequences=(i<m-1)))
		for i in range(len(dense_units)):
			model.add(keras.layers.Dense(dense_units[i], activation=activation))
		model.add(keras.layers.Dense(self.n_actions, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self._model = model
		self._model_name = self._model_tag + str(n_hidden) + str(dense_units)
