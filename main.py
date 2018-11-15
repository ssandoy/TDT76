import keras
import numpy as np
import torch

import generalist as Generalist
from helpers import dataset, datapreparation
# LOAD DATA
import specialist as Specialist

fs = 1
data_set = dataset.pianoroll_dataset_batch("/home/ssandoy/school/tdt76/neural-composer-assignement/datasets/training/piano_roll_fs" + str(fs) + "/")
filepath = "models/generalist-fs" + str(fs) + ".h5"
prediction_size = 1
training_song_index = 10
song_length = 100
batch_epochs = 15000

layers = [0,1]
layer_sizes = [128, 256, 512]


def embed_play_rolls():
    for num_layers in layers:
        for layer_size in layer_sizes:
            generalist = Generalist.Generalist(
                num_layers,
                layer_size,
                training_batch_size=32,
                prediction_batch_size=prediction_size,
                input_size=128,
                optimizer=keras.optimizers.Adam(lr=0.00001),
                loss_function="binary_crossentropy",
                fs=fs
            )
            generalist.training_model = generalist.load_model("models/" + get_model_name(num_layers, layer_size, training_song_index, fs) + ".h5")
            print(generalist.training_model.layers[0])
            generalist.prediction_model.set_weights(generalist.training_model.get_weights())
            test_song = 11
            input = datapreparation.get_songs(data_set, song_index=test_song)[0][0][:prediction_size]
            test_visualize_and_store_midi(generalist, input, test_song)



def test_visualize_and_store_midi(generalist, input, test_song):
    init = torch.from_numpy(input)
    # init = torch.round(torch.exp(init))
    init = torch.round(init / torch.max(init))
    input = init.numpy()
    if len(input.shape) == 3:
        roll = np.array(np.reshape(input, (prediction_size, input.shape[2])))
    else:
        roll = np.array(input)
    for i in range((song_length // prediction_size) - 1):
        if input.shape != (prediction_size, 1, 128):
            input = np.reshape(input, (prediction_size, 1, input.shape[1]))
        input = generalist.prediction_model.predict(input)
        init = torch.from_numpy(input)
        # init = torch.round(torch.exp(init))
        init = torch.round(init / torch.max(init))
        input = init.numpy()
        roll = np.append(roll, np.reshape(input, (prediction_size, input.shape[2])), axis=0)
        # generalist.prediction_model.reset_states()
    print("Roll for generalist with {} layers with size {} run on batch_epochs {} on song # {}".format(str(generalist.num_layers), str(generalist.layer_size), str(batch_epochs), str(test_song)))
    print("Prediction size: " + str(prediction_size))
    datapreparation.visualize_piano_roll(roll, filename=get_model_name(generalist.num_layers, generalist.layer_size, training_song_index, fs, prediction_size=prediction_size) + "-test_song#" + str(test_song) + ".png", fs=5)
    datapreparation.piano_roll_to_mid_file(np.transpose(roll) * 100, "midi/{}-num_layers{}-layer_size{}-song#{}-batch_epochs{}-pred_size{}-fs{}-test_song#{}.mid"
                                           .format("generalist", str(generalist.num_layers), str(generalist.layer_size), str(training_song_index), str(15_000), str(prediction_size), str(fs), str(test_song)), fs=5)



def train(generalist):
    for num_layers in layers:
        for layer_size in layer_sizes:

            print("Training layer {} with size {}".format(str(num_layers), str(layer_size)))
            generalist.train(data_set, dataset_epochs=1, batch_epochs=batch_epochs, song_index=training_song_index)
            input = datapreparation.get_songs(data_set, song_index=training_song_index)[0][0][:prediction_size]
            test_visualize_and_store_midi(generalist, input)

def get_model_name(num_layers, layer_size, song_index, fs, prediction_size=None):
    if prediction_size != None:
        return "{}-num_layers{}-layer_size{}-song#{}-batch_epochs{}-pred_size{}-fs{}".format("generalist", str(num_layers), str(layer_size), str(song_index), str(batch_epochs),str(prediction_size),str(fs))
    else:
        return "{}-num_layers{}-layer_size{}-song#{}-batch_epochs{}-fs{}".format("generalist", str(num_layers), str(layer_size), str(song_index), str(batch_epochs),str(fs))

# generalist.training_model = generalist.load_model(filepath)
# generalist.train(data_set, dataset_epochs=1, batch_epochs=5_000,num_songs=1)

# datapreparation.visualize_piano_roll(np.reshape(data_set[12][0].numpy(), (data_set[12][0].shape[0], data_set[12][0].shape[2])),fs=1)

#train()
#embed_play_rolls()

def run_specialist():
    specialist = Specialist.Specialist(
        2,
        256,
        training_batch_size=32,
        prediction_batch_size=prediction_size,
        input_size=128,
        optimizer=keras.optimizers.Adam(lr=0.00001),
        loss_function="binary_crossentropy",
        fs=fs
    )
    specialist.train_composer_model(1)

run_specialist()
