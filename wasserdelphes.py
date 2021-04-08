# cWGAN-GP script for the Delphes dataset - the data is collected from the delphes_source_code
# script and saved in .h5 format, where they are loaded on line 211.

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
import h5py
from functools import partial
from keras.models import Model

# deprecated in keras 2 higher
# from keras.layers.merge import _Merge
from keras.layers import Input, Dense, concatenate, LeakyReLU

# Global Variables

path = "..."
epochs = 500
condition_dim = 3
latent_dim = 100
RND = 777
np.random.seed(RND)
training_ratio = 5
gradient_penalty_weight = 10
batch_size = 128

# Generator


class Generator(object):
    def __init__(self, latent_dim, condition_dim):

        generator_input_1 = Input(shape=(latent_dim,), name="g_1")
        generator_input_2 = Input(shape=(condition_dim,), name="g_2")

        generator_input = concatenate([generator_input_1, generator_input_2])

        x = Dense(2048)(generator_input)
        x = LeakyReLU()(x)
        x = concatenate([x, generator_input_2])
        x = Dense(1024)(x)
        x = LeakyReLU()(x)
        x = concatenate([x, generator_input_2])
        x = Dense(512)(x)
        x = LeakyReLU()(x)
        x = concatenate([x, generator_input_2])
        x = Dense(256)(x)
        x = LeakyReLU()(x)
        x = Dense(condition_dim)(x)

        self.generator = Model(
            inputs=[generator_input_1, generator_input_2],
            outputs=[x, generator_input_2],
        )

        # print(self.generator.summary())

    def get_model(self):
        return self.generator


# Discriminator Model


class Discriminator(object):
    def __init__(self, condition_dim):

        discriminator_input_1 = Input(shape=(condition_dim,), name="d_1")
        discriminator_input_2 = Input(shape=(condition_dim,), name="d_2")

        discriminator_input = concatenate(
            [discriminator_input_1, discriminator_input_2]
        )

        x = Dense(2048)(discriminator_input)
        x = LeakyReLU()(x)
        x = concatenate([x, discriminator_input_2])
        x = Dense(1024)(x)
        x = LeakyReLU()(x)
        x = concatenate([x, discriminator_input_2])
        x = Dense(512)(x)
        x = LeakyReLU()(x)
        x = concatenate([x, discriminator_input_2])
        x = Dense(256)(x)
        x = LeakyReLU()(x)
        x = Dense(1)(x)

        self.discriminator = Model(
            inputs=[discriminator_input_1, discriminator_input_2], outputs=x
        )

        # print(self.discriminator.summary())

    def get_model(self):
        return self.discriminator


# Wasserstein Defintions


class random_weighted_average(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class random_weighted_average_(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(K.sum(y_pred), averaged_samples)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return gradient_penalty


# ConditionalGAN

# ConditionalGAN


class ConditionalGAN(object):
    def __init__(self, latent_dim, condition_dim):

        # Generator

        g = Generator(latent_dim, condition_dim)
        self._generator = g.get_model()

        d = Discriminator(condition_dim)
        self._discriminator = d.get_model()

        for layer in self._discriminator.layers:
            layer.trainable = False
        self._discriminator.trainable = False

        cgan_input_1 = Input(shape=(latent_dim,))
        cgan_input_2 = Input(shape=(condition_dim,))
        cgan_output = self._discriminator(self._generator([cgan_input_1, cgan_input_2]))
        self._cgan = Model([cgan_input_1, cgan_input_2], cgan_output)

        # Compile Generator

        cgan_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
        self._cgan.compile(optimizer=cgan_optimizer, loss=wasserstein_loss)

        # Discriminator

        for layer in self._discriminator.layers:
            layer.trainable = True
        for layer in self._generator.layers:
            layer.trainable = False
        self._discriminator.trainable = True
        self._generator.trainable = False

        dis_input_1 = Input(shape=(condition_dim,))
        dis_input_2 = Input(shape=(condition_dim,))
        dis_input_3 = Input(shape=(latent_dim,))

        fake_image = self._generator([dis_input_3, dis_input_2])
        fake_decision = self._discriminator(fake_image)
        real_decision = self._discriminator([dis_input_1, dis_input_2])
        av_images = random_weighted_average()([dis_input_1, fake_image[0]])
        av_labels = random_weighted_average_()([dis_input_2, fake_image[1]])
        av_decision = self._discriminator([av_images, av_labels])

        partial_gp_loss = partial(
            gradient_penalty_loss,
            averaged_samples=av_images,
            gradient_penalty_weight=gradient_penalty_weight,
        )
        partial_gp_loss.__name__ = "Gradient_Penalty"
        self._discriminator_ = Model(
            inputs=[dis_input_1, dis_input_2, dis_input_3],
            outputs=[real_decision, fake_decision, av_decision],
        )

        print(self._discriminator_.summary())

        # Compile discriminator

        discriminator_optimizer = keras.optimizers.Adam(
            lr=0.0001, beta_1=0.5, beta_2=0.9
        )
        self._discriminator_.compile(
            optimizer=discriminator_optimizer,
            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
        )

    def train_gen(self, noise, condition, batch_size):

        misleading_targets = np.ones((batch_size, 1))
        g_loss = self._cgan.train_on_batch([noise, condition], misleading_targets)

        return g_loss

    def train_dis(self, _image, condition, noise, batch_size):

        y_true = np.ones((batch_size, 1), dtype=np.float32)
        y_false = -y_true
        y_dummy = np.zeros((batch_size, 1), dtype=np.float32)

        d_loss = self._discriminator_.train_on_batch(
            [_image, condition, noise], [y_true, y_false, y_dummy]
        )[0]

        return d_loss

    def predict(self, latent_vector, condition):
        return self._generator.predict([latent_vector, condition])[0]

    def load_weights(self, file_path, by_name=False):
        self._cgan.load_weights(file_path, by_name)

    def save_weights(self, file_path, overwrite=True):
        self._cgan.save_weights(file_path, overwrite)

    def save_weights_gen(self, file_path, overwrite=True):
        self._generator.save_weights(file_path, overwrite)

    def to_json(self, file_path, overwrite=True):
        self._cgan.to_json(file_path, overwrite)

    def save_model(self, file_path, overwrite=True):
        self._cgan.save(file_path, overwrite=True)


# Main Program

# Training Data

h5f = h5py.File("...Delphes_Training_Data.h5", "r")
Training_Data = h5f["Training_Data"][:]
Training_Labels = h5f["Training_Labels"][:]
h5f.close()


def input_noise():

    Z_Train = np.random.normal(0, 1, (100000, 100))

    return Z_Train


# Train function


def train(latent_dim, condition_dim, epochs, path):

    # Get training data

    X_Train = Training_Data
    Y_Train = Training_Labels

    X_Train_ = np.reshape(X_Train, (10240000, 3))[0:100000]
    Y_Train_ = np.reshape(Y_Train, (10240000, 3))[0:100000]

    # Check training data

    # bins = np.linspace(0,5,100)
    # plt.hist(X_Train[0:500][:,0],bins,histtype='step',color='red',label='delphes_sample')
    # plt.hist(Y_Train[0:500][:,0],bins,histtype='step',color='blue',label='pythia_sample')
    # plt.title("Transverse Momentum Distribution of Electrons \n Fired from an Electron Gun")
    # plt.xlabel("Momentum / GeV")
    # plt.ylabel("Number")
    # plt.legend(loc='upper right')

    Z_Train = input_noise()

    # Load CGAN

    cgan = ConditionalGAN(latent_dim, condition_dim)

    # Define empty arrays for storing results

    d_loss_ = []
    discriminator_loss_ = []
    generator_loss_ = []

    # Loop

    for epoch in range(epochs):

        print("\n" + "Epoch: ", str(epoch + 1))

        minibatches_size = batch_size * training_ratio

        for i in range(int(X_Train_.shape[0] // (batch_size * training_ratio))):

            data_mb = X_Train_[i * minibatches_size : (i + 1) * minibatches_size]
            condition_mb = Y_Train_[i * minibatches_size : (i + 1) * minibatches_size]
            noise_mb = Z_Train[i * minibatches_size : (i + 1) * minibatches_size]

            for j in range(training_ratio):

                data = data_mb[j * batch_size : (j + 1) * batch_size]
                condition = condition_mb[j * batch_size : (j + 1) * batch_size]
                noise = noise_mb[j * batch_size : (j + 1) * batch_size]
                d_loss = cgan.train_dis(data, condition, noise, batch_size)
                d_loss_ = np.append(d_loss_, d_loss)

            discriminator_loss = np.average(d_loss_)

            index = np.random.choice(len(Z_Train), batch_size, replace=False)
            noise_ = Z_Train[index]
            condition = Y_Train_[index]

            generator_loss = cgan.train_gen(noise_, condition, batch_size)

        if epoch % 1 == 0:

            print("Training Loss:")

            print("Discriminator Loss: ", format(discriminator_loss, ".3f"))
            print("Generator Loss: ", format(generator_loss, ".3f"))

            discriminator_loss_ = np.append(discriminator_loss_, discriminator_loss)
            generator_loss_ = np.append(generator_loss_, generator_loss)
            plt.figure(figsize=(10, 8))
            plt.title("Loss vs Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.plot(discriminator_loss_, label="Critic Loss", color="g")
            plt.plot(generator_loss_, label="Generator Loss", color="r")
            plt.legend()
            plt.savefig(os.path.join(path, "Final_Loss_Graph.png"))
            plt.close()

        if epoch % 1 == 0:

            cgan.save_weights(os.path.join(path, "weights.h5"))
            data = np.zeros((1000, 3))
            _condition = np.zeros((1000, 3))
            diff_plot = np.zeros((1000, 3))

            rn = np.random.randint(0, X_Train_.shape[0])
            condition = Y_Train_[rn]
            condition_ = np.reshape(condition, (1, -1))

            for j in range(1000):

                random_latent_vector = np.random.normal(0, 2, 100)
                random_latent_vector = np.reshape(random_latent_vector, (1, -1))

                generated_data = cgan.predict(random_latent_vector, condition_)[0]
                diff_plot[j] = generated_data - condition_
                data[j] = generated_data
                _condition[j] = condition_

            bins = np.linspace(-2.5, 2.5, 100)
            plt.hist(
                diff_plot[:, 0], bins, histtype="step", color="red", label="momentum"
            )
            plt.hist(diff_plot[:, 1], bins, histtype="step", color="green", label="eta")
            plt.hist(
                diff_plot[:, 2], bins, histtype="step", color="yellow", label="phi"
            )
            plt.xlabel("Difference between Generated and Condition / GeV")
            plt.ylabel("Number")
            plt.legend(loc="upper right")
            plt.title("Difference Plot")
            plt.savefig("..." + str(epoch) + "Difference")
            plt.close()

            bins = np.linspace(0, 20, 100)
            plt.hist(data[:, 0], bins, histtype="step", color="blue", label="generated")
            plt.hist(
                _condition[:, 0], bins, histtype="step", color="red", label="condition"
            )
            plt.xlabel("Momentum / GeV")
            plt.ylabel("Number")
            plt.legend(loc="upper right")
            plt.title("Generated Momentum Distribution")
            plt.savefig("..." + str(epoch) + "_Momentum")
            plt.close()

            bins = np.linspace(min(data[:, 1]), max(data[:, 1]), 100)
            plt.hist(data[:, 1], bins, histtype="step", color="blue", label="generated")
            plt.hist(
                _condition[:, 1], bins, histtype="step", color="red", label="condition"
            )
            plt.xlabel("Eta / radians")
            plt.ylabel("Number")
            plt.legend(loc="upper right")
            plt.title("Generated Eta Distribution")
            plt.savefig("..." + str(epoch) + "_Eta")
            plt.close()

            bins = np.linspace(min(data[:, 2]), max(data[:, 2]), 100)
            plt.hist(data[:, 2], bins, histtype="step", color="blue", label="generated")
            plt.hist(
                _condition[:, 2], bins, histtype="step", color="red", label="condition"
            )
            plt.xlabel("Phi / radians")
            plt.ylabel("Number")
            plt.legend(loc="upper right")
            plt.title("Generated Phi Distribution")
            plt.savefig("..." + str(epoch) + "_Phi")
            plt.close()

        if (epoch + 1) % epochs == 0:
            model_json = cgan._generator.to_json()
            with open("generator_model.json", "w") as json_file:
                json_file.write(model_json)
            cgan.save_weights_gen(os.path.join(path, "model_weights.h5"))

        print("End of Epoch " + str(epoch + 1) + ":")
        print()


def run_trained():

    json_file = open("generator_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(path, "model_weights.h5"))

    data = np.zeros((1000, 3))
    for i in range(1000):
        conditions = [2, 1.5, 2.5]
        conditions = np.reshape(conditions, (1, -1))
        noise = np.random.normal(0, 2, 100)
        noise = np.reshape(noise, (1, -1))
        data[i] = loaded_model.predict([noise, conditions])[0]
        difference = data - conditions

    bins = np.linspace(-2.0, 2.0, 100)
    plt.hist(difference[:, 0], bins, histtype="step", color="red", label="momentum")
    plt.hist(difference[:, 1], bins, histtype="step", color="green", label="eta")
    plt.hist(difference[:, 2], bins, histtype="step", color="yellow", label="phi")
    plt.xlabel("Difference between generated and condition / GeV")
    plt.ylabel("Number")
    plt.legend(loc="upper right")
    plt.title("Difference Plot")
    plt.savefig("...")
    plt.close()

    bins = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
    plt.hist(data[:, 0], bins, histtype="step", color="black", label="generated")
    plt.xlabel("Momentum / GeV")
    plt.ylabel("Number")
    plt.title("Generated Momentum Distribution")
    plt.savefig("...")
    plt.close()

    bins = np.linspace(min(data[:, 1]), max(data[:, 1]), 10)
    plt.hist(data[:, 1], bins, histtype="step", color="black", label="generated")
    plt.xlabel("Eta / radians")
    plt.ylabel("Number")
    plt.title("Generated Eta Distribution")
    plt.savefig("...")
    plt.close()

    bins = np.linspace(min(data[:, 2]), max(data[:, 2]), 10)
    plt.hist(data[:, 2], bins, histtype="step", color="black", label="generated")
    plt.xlabel("Phi / radians")
    plt.ylabel("Number")
    plt.title("Generated Eta Distribution")
    plt.savefig("...")
    plt.close()


if __name__ == "__main__":
    train(latent_dim, condition_dim, epochs, path)
    run_trained()
