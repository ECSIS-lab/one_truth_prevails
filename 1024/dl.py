import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
import numpy as np
import sys
import shutil
import os
import pandas as pd
from tqdm import tqdm

def set_gpus(useOnlyOneGPU = False, gpu_index = 0):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

    if useOnlyOneGPU == True:
        tf.config.set_visible_devices(physical_devices[gpu_index], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_index], True)
    else:
        tf.config.set_visible_devices([physical_devices[n] for n in range(len(physical_devices))], 'GPU')
        for n in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[n], True)

def data_shaping_for_cnn(data):
    data = data.reshape(-1, data.shape[1], 1)
    return data

def conv_layer_functional(num_filter, x):
    x = kl.Conv1D(num_filter, kernel_size=(3), padding='SAME')(x)
    x = kl.BatchNormalization()(x)
    x = kl.ReLU()(x)
    x = kl.MaxPooling1D((2))(x)
    return x

def main(model_name, batch_size=2000, epochs=25):

    output =  model_name + "_e" + str(epochs) + "_batch" + str(batch_size)

    outputdir = "../result/" + output
    modelpath = "../result/" + output + "/{epoch:02d}.h5"
    logdir = outputdir + "/log"
    logpath = logdir + "/" + output
    shutil.rmtree(logpath, ignore_errors=True)

    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    tf.random.set_seed(10101)
    
    traces_read = np.load("./p.npy")
    val_traces = np.load("./a.npy")
    labels = np.loadtxt("./p_labels.txt")
    val_labels = np.load("./a_labels.npy")


    train_x = traces_read
    train_y = labels
    val_x = val_traces
    val_y = val_labels

    train_x = data_shaping_for_cnn(train_x)
    val_x = data_shaping_for_cnn(val_x)

    np.shape(train_x) 
    np.shape(train_y)
    x_size = train_x.shape[1]

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    with strategy.scope():
        normalizer = kl.experimental.preprocessing.Normalization()
        normalizer.adapt(val_x)

        input = kl.Input(shape=(x_size, 1))
        x = normalizer(input)
        x = conv_layer_functional(16, x)
        x = conv_layer_functional(16, x)
        x = conv_layer_functional(32, x)
        x = conv_layer_functional(32, x)
        x = conv_layer_functional(64, x)
        x = kl.Conv1D(64, kernel_size=(3), padding='SAME')(x)
        x = kl.BatchNormalization()(x)
        x = kl.ReLU()(x)
        x = kl.GlobalMaxPooling1D()(x)
        x = kl.Dense(128)(x)
        x = kl.ReLU()(x)
        x = kl.Dense(64)(x)
        x = kl.ReLU()(x)
        output = kl.Dense(1, activation='sigmoid')(x)
        model = k.Model(input, output)
        model.compile(
            loss="binary_crossentropy",
            optimizer=k.optimizers.Adam(lr=5e-5),
            metrics=['acc'])

    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))

    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')]

    history = model.fit(
        train_data,
        epochs=epochs,
        callbacks=callbacks,
        validation_data = val_data
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(outputdir + "/history_"+"a"+".csv")
    with open("model_"+"a"+"_summary.txt", "w") as fp:
        model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

if __name__ == "__main__":
    set_gpus()
    model_name = "cnn"
    main(model_name=model_name)
