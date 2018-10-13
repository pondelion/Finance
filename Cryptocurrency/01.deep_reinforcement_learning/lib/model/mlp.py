from .base_model import BaseKerasModel
import keras


class MLP(BaseKerasModel):

	def init(self):
		self._model_tag	= 'MLP'
		print('init')

	def build_model(self, n_hidden, learning_rate, activation='relu'):

		model = keras.models.Sequential()
		model.add(keras.layers.Reshape(
			(self.state_shape[0]*self.state_shape[1],), 
			input_shape=self.state_shape))

		for i in range(len(n_hidden)):
			model.add(keras.layers.Dense(n_hidden[i], activation=activation))
			#model.add(keras.layers.Dropout(drop_rate))
		
		model.add(keras.layers.Dense(self.n_actions, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self._model = model
		self._model_name = self._model_tag + str(n_hidden)
