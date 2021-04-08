# Script that compares the outputs of the trained cWGAN-GP and Delphes, used to
# produce most of the scripts in the last section of the report

import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import math

path = "..."

# cGAN Model


def run_trained(events, pt, eta, phi):

    json_file = open("generator_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(path, "model_weights.h5"))

    data = np.zeros((events, 1000, 3))

    for i in range(events):
        for j in range(1000):
            conditions = [pt, eta, phi]
            conditions_ = np.reshape(conditions, (1, -1))
            noise = np.random.normal(0, 2, 100)
            noise = np.reshape(noise, (1, -1))
            data[i, j] = loaded_model.predict([noise, conditions_])[0]

    return data


# Plot cWGAN-GP output

# a = run_trained(5,5,1.5,2.5)
# plt.title('Generated Momentum Distributions of an Incoming Electron')
# plt.hist(a[0,:,0],histtype='step', bins=100,label='first')
# plt.hist(a[1,:,0],histtype='step',bins=100,label='second')
# plt.hist(a[2,:,0],histtype='step',bins=100,label='third')
# plt.hist(a[3,:,0],histtype='step',bins=100,label='fourth')
# plt.hist(a[4,:,0],histtype='step',bins=100,label='fifth')
# plt.axvline(x=5,color='black')
# plt.legend(loc='upper right')
# plt.xlabel('Momentum / GeV')
# plt.ylabel('Number')
# plt.savefig("...")

# Delphes source code (again) but this time only generating a spike at a particular
# point to compare the outputs


def Real_Spike(events, p, eta, phi, times):

    number_of_instances = events
    g_range = range(number_of_instances)
    events_array = np.zeros([number_of_instances, times, 3])

    for i in g_range:

        for j in range(times):

            p = p
            eta = eta
            phi = phi
            pt = p / np.cosh(eta)

            events_array[i, j, 0] = pt
            events_array[i, j, 1] = eta
            events_array[i, j, 2] = phi

    return events_array


# a = Real_Spike(4.7,1.5,2.5,1000)


def Tracking_Efficiency_Electron(events_array):

    output_ = np.zeros(
        (events_array.shape[0], events_array.shape[1], events_array.shape[2])
    )

    for i in range(events_array.shape[0]):
        for j in range(events_array.shape[1]):
            if abs(events_array[i][j][0]) <= 0.1:
                output_[i][j][0] = 0.0
                output_[i][j][1] = 0.0
                output_[i][j][2] = 0.0
            if (
                abs(events_array[i][j][0]) <= 1.5
                and abs(events_array[i][j][0]) > 0.1
                and abs(events_array[i][j][0]) <= 1.0
            ):
                a = np.random.randint(0, 1)
                if a < 0.73:
                    output_[i][j][0] = events_array[i][j][0]
                    output_[i][j][1] = events_array[i][j][1]
                    output_[i][j][2] = events_array[i][j][2]
                else:
                    output_[i][j][0] = 0.0
                    output_[i][j][1] = 0.0
                    output_[i][j][2] = 0.0
            if (
                abs(events_array[i][j][1]) <= 1.5
                and abs(events_array[i][j][0]) > 0.1
                and abs(events_array[i][j][0]) <= 10
            ):
                a = np.random.randint(0, 1)
                if a < 0.95:
                    output_[i][j][0] = events_array[i][j][0]
                    output_[i][j][1] = events_array[i][j][1]
                    output_[i][j][2] = events_array[i][j][2]
                else:
                    output_[i][j][0] = 0.0
                    output_[i][j][1] = 0.0
                    output_[i][j][2] = 0.0
            if abs(events_array[i][j][1]) <= 1.5 and abs(events_array[i][j][0]) > 10:
                a = np.random.randint(0, 1)
                if a < 0.99:
                    output_[i][j][0] = events_array[i][j][0]
                    output_[i][j][1] = events_array[i][j][1]
                    output_[i][j][2] = events_array[i][j][2]
                else:
                    output_[i][j][0] = 0.0
                    output_[i][j][1] = 0.0
                    output_[i][j][2] = 0.0
            if (
                abs(events_array[i][j][1]) > 1.5
                and abs(events_array[i][j][1]) <= 2.5
                and abs(events_array[i][j][0]) > 0.1
                and abs(events_array[i][j][0]) <= 1.0
            ):
                a = np.random.randint(0, 1)
                if a < 0.50:
                    output_[i][j][0] = events_array[i][j][0]
                    output_[i][j][1] = events_array[i][j][1]
                    output_[i][j][2] = events_array[i][j][2]
                else:
                    output_[i][j][0] = 0.0
                    output_[i][j][1] = 0.0
                    output_[i][j][2] = 0.0
            if (
                abs(events_array[i][j][1]) > 1.5
                and abs(events_array[i][j][1]) <= 2.5
                and abs(events_array[i][j][0]) > 0.1
                and abs(events_array[i][j][0]) <= 10
            ):
                a = np.random.randint(0, 1)
                if a < 0.83:
                    output_[i][j][0] = events_array[i][j][0]
                    output_[i][j][1] = events_array[i][j][1]
                    output_[i][j][2] = events_array[i][j][2]
                else:
                    output_[i][j][0] = 0.0
                    output_[i][j][1] = 0.0
                    output_[i][j][2] = 0.0
            if (
                abs(events_array[i][j][1]) > 1.5
                and abs(events_array[i][j][1]) <= 2.5
                and abs(events_array[i][j][0]) > 10
            ):
                a = np.random.randint(0, 1)
                if a < 0.90:
                    output_[i][j][0] = events_array[i][j][0]
                    output_[i][j][1] = events_array[i][j][1]
                    output_[i][j][2] = events_array[i][j][2]
                else:
                    output_[i][j][0] = 0.0
                    output_[i][j][1] = 0.0
                    output_[i][j][2] = 0.0

            if abs(events_array[i][j][1]) > 2.5:
                output_[i][j][0] = 0.0
                output_[i][j][1] = 0.0
                output_[i][j][2] = 0.0

    return output_


# b = Tracking_Efficiency_Electron(a)


def Resolution(input_):

    g_range = range(input_.shape[0])

    res = np.zeros((input_.shape[0], input_.shape[1], input_.shape[2]))

    for i in g_range:
        for j in range(input_.shape[1]):
            if abs(input_[i][j][1]) <= 0.5 and input_[i][j][0] > 0.1:
                res_ = (
                    abs(input_[i][j][0])
                    * abs(input_[i][j][1])
                    * math.sqrt(
                        0.03 * 0.03
                        + (input_[i][j][0] * input_[i][j][0] * (1.30e-3) * (1.30e-3))
                    )
                )
            if (
                abs(input_[i][j][1]) > 0.5
                and abs(input_[i][j][1]) <= 1.5
                and input_[i][j][0] > 0.1
            ):
                res_ = (
                    abs(input_[i][j][0])
                    * abs(input_[i][j][1])
                    * math.sqrt(
                        0.05 * 0.05
                        + (input_[i][j][0] * input_[i][j][0] * (1.70e-3) * (1.70e-3))
                    )
                )
            if (
                abs(input_[i][j][1]) > 1.5
                and abs(input_[i][j][1]) <= 2.5
                and input_[i][j][0] > 0.1
            ):
                res_ = (
                    abs(input_[i][j][0])
                    * abs(input_[i][j][1])
                    * math.sqrt(
                        0.15 * 0.15
                        + (input_[i][j][0] * input_[i][j][0] * (3.10e-3) * (3.10e-3))
                    )
                )
            if abs(input_[i][j][0]) <= 0.1:
                res_ = 0
            res[i, j, 0] = res_
            res[i, j, 1] = input_[i, j, 1]
            res[i, j, 2] = input_[i, j, 2]
    return res


# c = Resolution(b)


def Resolution_Trimming(input_):

    g_range = range(input_.shape[0])

    Res = np.zeros((input_.shape[0], input_.shape[1], input_.shape[2]))

    for i in g_range:
        for j in range(input_.shape[1]):
            if input_[i][j][0] > 1.0:
                Res[i][j][0] = 1.0
            else:
                Res[i][j][0] = input_[i][j][0]
            Res[i][j][1] = input_[i][j][1]
            Res[i][j][2] = input_[i][j][2]
    return Res


# d = Resolution_Trimming(c)


def Momentum_Smearing_Log_Normal(mean, sigma):
    if mean > 0:
        b = math.sqrt(np.log((1.0 + (sigma * sigma) / (mean * mean))))
        a = np.log(mean) - 0.5 * b * b
        return np.exp(a + b * np.random.normal(0.0, 1.0))
    else:
        return 0.0


def Momentum_Smearing_Gauss(mean, sigma):
    return np.random.normal(mean, sigma)


def Smear(input_, resolution):

    g_range = range(input_.shape[0])

    Smear = np.zeros((input_.shape[0], input_.shape[1], input_.shape[2]))

    for i in g_range:
        for j in range(input_.shape[1]):
            Smear[i][j][0] = Momentum_Smearing_Log_Normal(
                input_[i][j][0], resolution[i][j][0] * input_[i][j][0]
            )
            Smear[i][j][1] = input_[i][j][1]
            Smear[i][j][2] = input_[i][j][2]
    return Smear


# e = Smear(a,d)


def Run_Entire_Spike(events, p, eta, phi):

    p = p
    eta = eta
    phi = phi

    Real_ = np.zeros((events, 1000, 3))
    Fake_ = np.zeros((events, 1000, 3))
    Real_ = Real_Spike(events, p, eta, phi, 1000)
    Fake_ = Smear(
        Real_, Resolution_Trimming(Resolution(Tracking_Efficiency_Electron(Real_)))
    )

    return Real_, Fake_


# b,c = Run_Entire_Spike(5,5,1.5,2.5)

# Plot Delphes output

# plt.title('Delphes Momentum Distributions of an Incoming Electron')
# plt.hist(c[0,:,0],histtype='step', bins=100,label='first')
# plt.hist(c[1,:,0],histtype='step',bins=100,label='second')
# plt.hist(c[2,:,0],histtype='step',bins=100,label='third')
# plt.hist(c[3,:,0],histtype='step',bins=100,label='fourth')
# plt.hist(c[4,:,0],histtype='step',bins=100,label='fifth')
# plt.axvline(x=5,color='black')
# plt.legend(loc='upper right')
# plt.xlabel('Momentum / GeV')
# plt.ylabel('Number')
# plt.savefig("...")

# plt.title('Comparison between Delphes and conditional GAN')
# plt.hist(a[0,:,0],histtype='step',bins=50,label='cGAN')
# plt.hist(c[0,:,0],histtype='step',bins=50,label='delphes')
# plt.axvline(x=5,color='black')
# plt.legend(loc='upper right')
# plt.xlabel('Momentum / GeV')
# plt.ylabel('Number')
# plt.savefig("...")


def run_comparison(events, p, eta, phi):

    pt = p / np.cosh(eta)

    a = run_trained(events, pt, eta, phi)
    _, b = Run_Entire_Spike(events, p, eta, phi)

    bins = np.linspace(0, 1, 50)
    bins = 100
    plt.hist(a[0, :, 0], histtype="step", bins=bins, label="first")
    plt.hist(a[1, :, 0], histtype="step", bins=bins, label="second")
    plt.hist(a[2, :, 0], histtype="step", bins=bins, label="third")
    plt.hist(a[3, :, 0], histtype="step", bins=bins, label="fourth")
    plt.hist(a[4, :, 0], histtype="step", bins=bins, label="fifth")
    plt.axvline(x=pt, color="black")
    plt.legend(loc="upper right")
    plt.xlabel("Momentum / GeV")
    plt.ylabel("Number")
    plt.savefig("..." + str(pt) + ".png")
    plt.close()

    plt.title("Delphes Momentum Distributions of an Incoming Electron")
    plt.hist(b[0, :, 0], histtype="step", bins=bins, label="first")
    plt.hist(b[1, :, 0], histtype="step", bins=bins, label="second")
    plt.hist(b[2, :, 0], histtype="step", bins=bins, label="third")
    plt.hist(b[3, :, 0], histtype="step", bins=bins, label="fourth")
    plt.hist(b[4, :, 0], histtype="step", bins=bins, label="fifth")
    plt.axvline(x=pt, color="black")
    plt.legend(loc="upper right")
    plt.xlabel("Momentum / GeV")
    plt.ylabel("Number")
    plt.savefig("..." + str(pt) + "_" + str(eta) + "_" + str(phi) + ".png")
    plt.close()

    binsa = np.linspace(
        min(a[0, :, 0]), max(a[0, :, 0]), int(max((a[0, :, 0]) - min(a[0, :, 0]))) * 2
    )
    binsb = np.linspace(
        min(b[0, :, 0]), max(b[0, :, 0]), int(max((b[0, :, 0]) - min(b[0, :, 0]))) * 2
    )
    # bins=100
    plt.title("Comparison between Delphes and cWGAN-GP")
    plt.hist(a[0, :, 0], histtype="step", bins=binsa, label="cWGAN-GP", color="red")
    plt.hist(b[0, :, 0], histtype="step", bins=binsb, label="delphes", color="green")
    plt.axvline(x=pt, color="black")
    plt.legend(loc="upper right")
    plt.xlabel("Momentum / GeV")
    plt.ylabel("Number")
    plt.savefig("..." + str(pt) + "_" + str(eta) + "_" + str(phi) + ".png")
    plt.close()


if __name__ == "__main__":
    import time

    start_time = time.time()
    run_comparison(1, 10, 1.0, 2.5)
    print("--- %s seconds ---" % (time.time() - start_time))
