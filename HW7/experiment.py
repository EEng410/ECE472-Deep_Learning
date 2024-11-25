# The inspiration from this experiment comes from a failed attempt at my ECE435-Medical Imaging final project. 
# We were instructed to implement the Fast Iterative Soft Thresholding Algorithm on MRI brain scans. The objective
# was to undersample an MRI scan and reconstruct it in the image domain while assuming sparsity in the wavelet
# detail coefficients. The idea behind this was that if we can downsample MRI images, we can simulate a compressed
# sensing process. This is valuable in the sense that MRI scans can go faster and then cost less. 
# 
# I was able to achieve this with downsampling in the image domain and get pretty good results. However, MRI scans 
# are not taken in the actual image domain. Rather, roughly speaking, the MRI machine is sampling the Fourier 
# transform of the image, which suggests that the downsampling should be done on the Fourier Transform of the image. 
# For whatever reason, I couldn't get this to work when downsampling into the Fourier domain, since the artifacts 
# introduced by random pixel masking was ruining the scan in the image domain. 

import pickle
import tensorflow as tf
from sklearn.inspection import DecisionBoundaryDisplay


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, omega_0 = 30, bias=True):
        rng = tf.random.get_global_generator()
        self.omega_0 = omega_0
        self.w = tf.Variable(
            rng.uniform(shape=[num_inputs, num_outputs], minval = -np.sqrt(6/num_inputs)/self.omega_0 , maxval = np.sqrt(6/num_inputs)/self.omega_0),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                rng.uniform(
                    shape=[1, num_outputs], minval = -1/num_inputs , maxval = 1/num_inputs),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z
    
class SineLayer(tf.Module):
    def __init__(self, num_inputs, num_outputs, first_layer = False, bias=True, omega_0 = 30):
        rng = tf.random.get_global_generator()
        self.omega_0 = omega_0
        self.first_layer = first_layer
        if self.first_layer:
            self.w = tf.Variable(
                rng.uniform(shape=[num_inputs, num_outputs], minval = -1/num_inputs, maxval = 1/num_inputs),
                trainable=True,
                name="SineLayer/w",
            )
        else:
            self.w = tf.Variable(
                rng.uniform(shape=[num_inputs, num_outputs], minval = -tf.math.sqrt(6/num_inputs)/self.omega_0 , maxval = tf.math.sqrt(6/num_inputs)/self.omega_0),
                trainable=True,
                name="SineLayer/w",
            )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                rng.uniform(
                    shape=[1, num_outputs], minval = -1/num_inputs , maxval = 1/num_inputs),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b
        z = z * self.omega_0
        z = tf.math.sin(z)
        return z

class SIREN(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, hidden_activation = tf.nn.relu, output_activation = tf.identity, bias=True): # ReLU just to clip negative outputs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.bias = bias
        self.input_layer = SineLayer(num_inputs, hidden_layer_width, first_layer=True)

        self.hidden_layers = [SineLayer(hidden_layer_width, hidden_layer_width) for i in range(int(num_hidden_layers))]

        self.output_layer = Linear(hidden_layer_width, num_outputs)


    def __call__(self, x):
        z = tf.cast(x, 'float32')
        
        z = self.input_layer(z)
       
        for layer in range(self.num_hidden_layers):
            z = self.hidden_layers[layer](z)
        p = self.output_activation(self.output_layer(z))
        return p


def save_model(model, name):
    with open(name, "wb") as file: # file is a variable for storing the newly created file, it can be anything.
        pickle.dump(model, file) # Dump function is used to write the object into the created file in byte format.

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":
    import argparse
    import numpy as np
    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange
    from PIL import Image
    import nibabel as nib
    from scipy.fftpack import fftshift, fft2, ifft2
    num_iters = 1
    step_size = 1e-4
    decay_rate = 0.999
    gamma = 1
    num_hidden_layers = 6
    hidden_layer_width = 512
    refresh_rate = 10

    optimizer = tf.keras.optimizers.Adam(learning_rate = step_size)
    bar = trange(num_iters)

    img = nib.load("sub-CSI1_ses-16_run-01_T1w.nii.gz")
    im1 = img.get_fdata()
    im = im1[87, :, :]
    # Normalize to 0, 1
    # im = im/np.max(im)
    
    F_im = fftshift(fft2(im))
    img_tensor_log_mag = tf.convert_to_tensor(20*np.log10(np.abs(F_im)))
    img_tensor_angle = tf.convert_to_tensor(np.unwrap(np.unwrap(np.angle(F_im))))

    # Normalize Tensor for training
    # img_tensor = img_tensor/tf.math.reduce_max(img_tensor)
    
    img_tensor = tf.convert_to_tensor(F_im)
    num_channels = 1
    # Generate the coordinate 
    X, Y = tf.meshgrid(np.linspace(-1, 1, img_tensor.shape[1]), np.linspace(-1, 1, img_tensor.shape[0]))
    coord_tensor = tf.transpose(tf.squeeze(tf.Variable([X, Y])))
    coord_tensor = tf.reshape(coord_tensor, shape = [-1, 2])

    # Flatten image tensor
    img_tensor_log_mag = tf.reshape(img_tensor_log_mag, shape = [-1, num_channels])

    img_tensor_log_mag = tf.cast(img_tensor_log_mag, dtype = tf.float32)

    # Some normalization 
    img_tensor_log_mag_max = tf.math.reduce_max(img_tensor_log_mag)
    # img_tensor_log_mag = img_tensor_log_mag / img_tensor_log_mag_max

    img_tensor_angle = tf.reshape(img_tensor_angle, shape = [-1, num_channels])

    img_tensor_angle = tf.cast(img_tensor_angle, dtype = tf.float32)
    img_tensor_angle_max = tf.math.reduce_max(img_tensor_angle)
    # img_tensor_angle = img_tensor_angle / img_tensor_angle_max


    # img_tensor = tf.concat([img_tensor_log_mag, img_tensor_angle], 1)

    img_tensor = tf.concat([img_tensor_log_mag, img_tensor_angle], axis = 1)
    img_mean = tf.math.reduce_mean(img_tensor, axis = 0)
    img_std = tf.math.sqrt(tf.math.reduce_variance(img_tensor, axis = 0))

    img_tensor = (img_tensor - img_mean)/img_std
    # Initialize SIREN Model, 2 channel output to model log magnitude and phase
    # siren = SIREN(2, 2, num_hidden_layers, hidden_layer_width)
    siren = unpickle("siren.pkl")
    bar = trange(num_iters)
    for i in bar:
        x_batch = coord_tensor
        y_batch = img_tensor
        with tf.GradientTape(persistent = True) as tape:
            y_hat = siren(x_batch)
            # loss =  tf.math.reduce_mean((y_batch - y_hat) ** 2) + gamma * tf.image.total_variation(tf.reshape(y_hat, shape = [1, 256, 256, 1]))
            loss =  tf.math.reduce_mean((y_batch - y_hat) ** 2)
            # for var in siren.trainable_variables:
            #     loss += gamma* tf.math.reduce_sum(tf.multiply(tf.reshape(var, [1, -1]), tf.reshape(var, [1, -1])))
        grads = tape.gradient(loss, siren.trainable_variables)
        optimizer.apply_gradients(zip(grads, siren.trainable_variables))
        
        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy().squeeze():0.4f}, step_size => {step_size:0.4f}"
            )

            bar.refresh()
    breakpoint()
    save_model(siren, "siren.pkl")
    breakpoint()
    # im = tf.reshape(y_hat[:, 0], shape = [256, 256])
    # plt.imshow(im.numpy())
    # plt.savefig("im_est.png")
    # 
    y_hat = y_hat*img_std
    y_hat = y_hat+img_mean
    logmag = tf.reshape(y_hat[:, 0], shape = [256, 256])
    phase = tf.reshape(y_hat[:, 1], shape = [256, 256])
    plt.imshow(logmag.numpy())
    plt.savefig("mag_est.png")

    plt.imshow(phase.numpy())
    plt.savefig("phase_est.png")
    breakpoint()
    img_out = Image.fromarray(tf.reshape(y_hat*255, shape = [256, 256]).numpy().astype(np.uint8))
    img_out.save("experiment.png")
    breakpoint()