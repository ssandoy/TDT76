from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization, Flatten
from keras.models import Sequential, load_model, save_model
import keras.optimizers as optimizers
import numpy as np

from generalist import Generalist
from helpers import datapreparation

filepath = "models/generalist.best-train.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min',
)
callbacks_list = [checkpoint]
NAME = "generalist"


class Specialist:

    def __init__(self, number_of_layers, hidden_size, input_size, training_batch_size, prediction_batch_size=1, optimizer=optimizers.Adam(), loss_function='binary_crossentropy', fs=None):
        self.composer_size = 4
        self.batch_size = training_batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = number_of_layers
        self.layer_size = hidden_size
        self.fs = fs
        # self.prediction_model = Sequential()
        # self.training_model = Sequential()
        # self.training_model.add(LSTM(128, input_shape=(1, self.input_size), batch_size=self.batch_size, stateful=True, return_sequences=True))
        # self.training_model.add(Dropout(0.2))
        # self.training_model.add(LSTM(128, input_shape=(1, self.input_size), return_sequences=True, batch_size=self.batch_size, stateful=True))
        # #self.training_model.add(Flatten())
        # self.training_model.add(Dense(256))
        # self.training_model.add(Dropout(0.3))
        # self.training_model.add(Dense(self.input_size))
        # self.training_model.add(Activation('softmax'))
        # self.training_model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.training_model = self.init_model(
            training_batch_size,
            optimizer=optimizer,
            loss_function=loss_function
        )
        self.prediction_model = self.init_model(
            prediction_batch_size,
            optimizer=optimizer,
            loss_function=loss_function
        )
        self.composer_model = Sequential()
        self.composer_model.add(Dense(4, input_dim=self.composer_size))
        self.composer_model.add(Dense(10, activation="relu"))
        self.composer_model.add(Dense(self.input_size))


    def init_model(self, batch_size,  optimizer='adam', loss_function='mse'):
        model = Sequential()
        model.add(LSTM(self.hidden_size, input_shape=(1, self.input_size), batch_size=batch_size, return_sequences=True, stateful=True))
        for _ in range(self.num_layers):
            model.add(LSTM(self.hidden_size, input_shape=(1, self.input_size), return_sequences=True, stateful=True))
            model.add(Dropout(0.3))
        #model.add(BatchNormalization())
        model.add(Dense(self.input_size))
        # TODO: TESTME SIGMOID
        model.add(Activation('sigmoid'))

        model.compile(optimizer=optimizer, loss=loss_function,)

        return model

    def load_model(self, filepath):
        return load_model(filepath)

    def forward(self, input, tags, hidden=None):
        return self.prediction_model.predict(input), None

    def train(self, songs, dataset_epochs=1,batch_epochs=5, num_songs=None, song_index=None):
        inputs, outputs, tags = datapreparation.get_songs(songs, num_songs, song_index)
        for epoch in range(dataset_epochs):
            for i, (input, output, tag) in enumerate(zip(inputs, outputs, tags)):
                # TODO: ALSO THIS IN PREDICT.
                self.train_composer_model(tag)
                input = input[1:(len(input) - (len(input)%self.batch_size))]
                input = np.append(input, np.ones(shape=(1,1, len(input[0][0]))), axis=0)
                output = output[:(len(output) - (len(output)%self.batch_size))]
                print("Epoch # " + str(epoch+1) + "/" + str(dataset_epochs))
                print("Song # " + str(i+1) + "/" + str(len(inputs)))
                self.training_model.reset_states()
                self.training_model.fit(input, output, epochs=batch_epochs, batch_size=self.batch_size, shuffle=False)
        save_model(self.training_model, "models/{}-num_layers{}-layer_size{}-song#{}-batch_epochs{}-fs{}.h5".format(NAME, str(self.num_layers), str(self.layer_size), str(song_index), str(batch_epochs) ,str(self.fs)))
        self.prediction_model.set_weights(self.training_model.get_weights())


    def train_composer_model(self, composer_input):
        input = np.zeros(self.composer_size)
        np.put(input,composer_input,1)
        self.composer_model.predict(input)



