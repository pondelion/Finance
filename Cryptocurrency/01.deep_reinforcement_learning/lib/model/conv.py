from .base_model import BaseKerasModel
import keras


class Conv(BaseKerasModel):

	def init(self):
		self._model_tag	= 'conv'

	def build_model(self, 
		filter_num, filter_size, dense_units, 
		learning_rate, activation='relu', dilation=None, use_pool=None):

		if use_pool is None:
			use_pool = [True]*len(filter_num)
		if dilation is None:
			dilation = [1]*len(filter_num)

		model = keras.models.Sequential()
		model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))
		
		for i in range(len(filter_num)):
			model.add(keras.layers.Conv1D(filter_num[i], kernel_size=filter_size[i], dilation_rate=dilation[i], 
				activation=activation, use_bias=True))
			if use_pool[i]:
				model.add(keras.layers.MaxPooling1D(pool_size=2))
		
		model.add(keras.layers.Flatten())
		for i in range(len(dense_units)):
			model.add(keras.layers.Dense(dense_units[i], activation=activation))
		model.add(keras.layers.Dense(self.n_actions, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		
		self._model = model

		self._model_name = self._model_tag + str([a for a in
			zip(filter_num, filter_size, dilation, use_pool)
			])+' + '+str(dense_units)
