import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    MaxPool2D,
    UpSampling2D,
    Conv2D,
    concatenate,
    Flatten,
    Activation,
    Reshape,
)
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical


class Generator:
    def __init__(self, latent_dim, condition_dim, RGB):

        generator_input_1 = Input(shape=(latent_dim,), name="g_1")
        generator_input_2 = Input(shape=(condition_dim,), name="g_2")
        generator_input = concatenate([generator_input_1, generator_input_2])

        x = Dense(1024)(generator_input)
        x = Activation("tanh")(x)
        x = Dense(128 * 7 * 7)(x)
        x = Activation("tanh")(x)
        x = Reshape((7, 7, 128))(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, 5, padding="same")(x)
        x = Activation("tanh")(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(RGB, 5, padding="same")(x)
        x = Activation("tanh")(x)

        self.generator = Model(
            inputs=[generator_input_1, generator_input_2],
            outputs=[x, generator_input_2],
        )

    def get_model(self):
        return self.generator


class Discriminator:
    def __init__(self, height, width, RGB, condition_dim):

        discriminator_input_1 = Input(
            shape=(
                height,
                width,
                RGB,
            ),
            name="d_1",
        )
        discriminator_input_2 = Input(shape=(condition_dim,), name="d_2")

        _discriminator_input_2 = Reshape((1, 1, condition_dim))(discriminator_input_2)
        _discriminator_input_2 = UpSampling2D((height, width))(_discriminator_input_2)

        discriminator_input = concatenate(
            [discriminator_input_1, _discriminator_input_2]
        )

        x = Conv2D(64, 5, strides=2, padding="same")(discriminator_input)
        x = Activation("tanh")(x)
        x = MaxPool2D()(x)
        x = Conv2D(128, 5, strides=2, padding="same")(x)
        x = Activation("tanh")(x)
        x = MaxPool2D()(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation("tanh")(x)
        x = Dense(1)(x)
        x = Activation("sigmoid")(x)

        self.discriminator = Model(
            inputs=[discriminator_input_1, discriminator_input_2], outputs=x
        )

    def get_model(self):
        return self.discriminator


class ConditionalDCGAN:
    def __init__(self, latent_dim, height, width, RGB, condition_dim):

        self._latent_dim = latent_dim
        g = Generator(latent_dim, condition_dim, RGB)
        self._generator = g.get_model()

        d = Discriminator(height, width, RGB, condition_dim)
        self._discriminator = d.get_model()

        discriminator_optimizer = keras.optimizers.Adam(lr=0.0008, decay=1e-8)
        self._discriminator.compile(
            optimizer=discriminator_optimizer, loss="binary_crossentropy"
        )

        self._discriminator.trainable = False

        dcgan_input_1 = Input(shape=(latent_dim,))
        dcgan_input_2 = Input(shape=(condition_dim,))
        dcgan_output = self._discriminator(
            self._generator([dcgan_input_1, dcgan_input_2])
        )

        self._dcgan = Model([dcgan_input_1, dcgan_input_2], dcgan_output)
        dcgan_optimizer = keras.optimizers.Adam(lr=0.0004, decay=1e-8)
        self._dcgan.compile(optimizer=dcgan_optimizer, loss="binary_crossentropy")

        print(self._dcgan.summary())

    def train(self, real_images, conditions, batch_size):

        random_latent_vectors = np.random.normal(size=(batch_size, self._latent_dim))
        generated_images = self._generator.predict([random_latent_vectors, conditions])
        labels = np.ones((batch_size, 1))
        labels += 0.05 * np.random.random(labels.shape)
        d_loss_1 = self._discriminator.train_on_batch(generated_images, labels)

        labels = np.zeros((batch_size, 1))
        labels += 0.05 * np.random.random(labels.shape)
        d_loss_2 = self._discriminator.train_on_batch([real_images, conditions], labels)
        d_loss = (d_loss_1 + d_loss_2) / 2.0

        random_latent_vectors = np.random.normal(size=(batch_size, self._latent_dim))
        misleading_targets = np.zeros((batch_size, 1))
        g_loss = self._dcgan.train_on_batch(
            [random_latent_vectors, conditions], misleading_targets
        )

        return d_loss, g_loss

    def predict(self, latent_vector, condition):
        return self._generator.predict([latent_vector, condition])[0]

    def load_weights(self, file_path, by_name=False):
        self._dcgan.load_weights(file_path, by_name)

    def save_weights(self, file_path, overwrite=True):
        self._dcgan.save_weights(file_path, overwrite)


def normalize(X):
    return (X - 127.5) / 127.5


def denormalize(X):
    return (X + 1.0) * 127.5


def train(latent_dim, height, width, RGB, num_class, epochs, path):

    (X_Train, Y_Train), (_, _) = keras.datasets.mnist.load_data()
    Y_Train = to_categorical(Y_Train, num_class)[0:10000]
    X_Train = X_Train[:, :, :, None].astype("float32")
    X_Train = normalize(X_Train)[0:10000]

    batch_size = 64
    iterations = X_Train.shape[0] // batch_size

    dcgan = ConditionalDCGAN(latent_dim, height, width, RGB, num_class)

    _d_loss = []
    _g_loss = []

    e_range = range(epochs)

    for i in e_range:

        d_loss = []
        g_loss = []

        print("Beginning of Epoch " + str(i + 1) + ":\n")

        for j in range(iterations):

            real_images = X_Train[j * batch_size : (j + 1) * batch_size]
            conditions = Y_Train[j * batch_size : (j + 1) * batch_size]
            d_loss_, g_loss_ = dcgan.train(real_images, conditions, batch_size)
            d_loss = np.append(d_loss, d_loss_)
            g_loss = np.append(g_loss, g_loss_)

            if (j + 1) % 10 == 0:

                print("Iteration: " + str(j + 1) + "/" + str(iterations))
                print("Discriminator Loss: ", format(d_loss_, ".3f"))
                print("Generator Loss: ", format(g_loss_, ".3f"))
                print()

            if (j + 1) % 10 == 0:
                _d_loss = np.append(_d_loss, d_loss[j])
                _g_loss = np.append(_g_loss, g_loss[j])
                plt.figure(figsize=(10, 8))
                plt.title("Loss vs Iterations")
                plt.xlabel("Units of 100 Iterations of Batch Size 128")
                plt.ylabel("Loss")
                plt.plot(_d_loss, label="Discriminator Loss")
                plt.plot(_g_loss, label="Generator Loss")
                plt.legend()
                plt.savefig(os.path.join(path, "loss_graph_final.png"))
                plt.close()

        if (i + 1) % 1 == 0:

            dcgan.save_weights(
                os.path.join(path, "gan" + "_epoch_" + str(i + 1) + ".h5")
            )
            random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
            generated_images = dcgan.predict(random_latent_vectors, conditions)
            length = generated_images.shape[0]
            c = np.random.randint(0, length)
            img = denormalize(generated_images[c])
            img = image.array_to_img(img, scale=False)
            condition = np.argmax(conditions[c])
            img.save(
                os.path.join(
                    path, "gen_during_" + str(i + 1) + "_" + str(condition) + ".png"
                )
            )

        print("End of Epoch " + str(i + 1) + " :")
        print()


def predict(latent_dim, height, width, RGB, num_class, epochs, path):

    dcgan = ConditionalDCGAN(latent_dim, height, width, RGB, num_class)
    dcgan.load_weights(os.path.join(path, "gan_epoch_" + epochs + ".h5"))
    for num in range(num_class):
        for _ in range(10):
            random_latent_vectors = np.random.normal(size=(1, latent_dim))
            conditions = np.zeros((1, num_class), dtype=np.float32)
            conditions[0, num] = 1
            generated_images = dcgan.predict(random_latent_vectors, conditions)
            img = image.array_to_img(denormalize(generated_images[0]), scale=False)
            img.save(os.path.join(path, "gen_after_" + str(num) + ".png"))


if __name__ == "__main__":

    if (len(sys.argv) != 3):
        raise ValueError("Wrong number of arguments supplied.")
    path = str(sys.argv[1])
    epochs = int(sys.argv[2])

    latent_dim = 100
    height = 28
    width = 28
    RGB = 1
    num_class = 10

    train(latent_dim, height, width, RGB, num_class, epochs, path)
    predict(latent_dim, height, width, RGB, num_class, epochs, path)
