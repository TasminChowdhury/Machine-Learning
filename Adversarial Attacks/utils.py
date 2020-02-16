import numpy as np
import Augmentor
import os


def _permute_index(l, seed):
    """
    Creates a permutation of np.array([0, ..., l-1]) and its inverse
    :param l: length of the array to permute
    :param seed: permutation seed
    :return: (s, s_inverse) where s is permutation of np.array([0, ..., l-1]) and s_inverse is its inverse
    """
    st0 = np.random.get_state()
    s = np.arange(l)
    np.random.seed(seed)
    np.random.shuffle(s)
    s_inverse = np.argsort(s)
    np.random.set_state(st0)
    return s, s_inverse


def permute(data, seed):
    """
    Permutes images in the data with given seed for each channel.
    :param data: numpy array with shape (nb_images, img_rows, img_cols, nb_channels)
    :param seed: permutation seed. If seed=None returns data without permutation
    :return: numpy array with shape (nb_images, img_rows, img_cols, nb_channels) of permuted images
    """
    """
    Permutes images in the data with given seed. If seed=None, returns data without permutation.
    Assumes data has shape (nb_images, img_rows, img_cols, nb_channels)
    """
    nb_images, img_rows, img_cols, nb_channels = data.shape
    if seed is None:
        return data
    l = img_rows * img_cols  # length of the permutation array
    s, _ = _permute_index(l, seed)
    output = np.zeros(data.shape)
    for ch in range(nb_channels):
        output[:, :, :, ch] = data[:, :, :, ch].reshape(-1, l)[:, s].reshape(-1, img_rows, img_cols)
    return output


def ipermute(data, seed):
    """
    inverse of permute
    :param data: numpy array with shape (nb_images, img_rows, img_cols, nb_channels)
    :param seed:  permutation seed. If seed=None returns data without permutation
    :return: numpy array with shape (nb_images, img_rows, img_cols, nb_channels) of inverse permuted images
    """
    nb_images, img_rows, img_cols, nb_channels = data.shape
    if seed is None:
        return data
    l = img_rows * img_cols  # length of the permutation array
    _, s_inverse = _permute_index(l, seed)
    output = np.zeros(data.shape)
    for ch in range(nb_channels):
        output[:, :, :, ch] = data[:, :, :, ch].reshape(-1, l)[:, s_inverse].reshape(-1, img_rows, img_cols)
    return output


def fourier(data):
    """
    converts each channel of images in the data to its 2-dimensional discrete Fourier transform.
    :param data: numpy array with shape (nb_images, img_rows, img_cols, nb_channels)
    :return: numpy array with shape (nb_images, img_rows, img_cols, 2*nb_channels)
    The first half of output channels are magnitude information, the second half are phase info in range (-pi, pi)
    """
    channels = data.shape[-1]
    output_shape = list(data.shape)
    output_shape[-1] = channels*2
    data_f = np.zeros(output_shape)
    for i in range(data.shape[0]):
        for ch in range(channels):
            f = np.fft.fft2(data[i, :, :, ch])
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            phase = np.angle(fshift)
            data_f[i, :, :, ch] = magnitude
            data_f[i, :, :, ch + channels] = phase
    return data_f


def pol2cart(r, theta):
    """
    Convert polar representation of a complex number to cartesian representation
    :param r: scalar or numpy array denoting magnitude component of the complex number
    :param theta: scalar or numpy array denoting phase of the complex number in radians.
    """
    return r * np.exp(1j * theta)


def ifourier(data_f):
    """
    inverse of fourier function
    :param data_f: numpy array with shape (nb_images, img_rows, img_cols, 2*nb_channels)
    The first half of output channels are magnitude information, the second half are phase info in range (-pi, pi)
    :return: numpy array with shape (nb_images, img_rows, img_cols, nb_channels) denoting data in pixel domain.
    """
    channels = int(data_f.shape[-1]/2)
    output_shape = list(data_f.shape)
    output_shape[-1] = channels
    data = np.zeros(output_shape, dtype='complex')  # The dtype is now changed to 'complex' not to lose any information.
    for i in range(data_f.shape[0]):
        for ch in range(channels):
            fshift = pol2cart(data_f[i, :, :, ch], data_f[i, :, :, ch + channels])
            f = np.fft.ifftshift(fshift)
            data[i, :, :, ch] = np.fft.ifft2(f)
    return data


def phase2pixel(phase):
    """
    reconstruct pixel domain from phase by adding unity magnitude.
    :param phase: numpy array with shape (nb_images, img_rows, img_cols, nb_channels) containing phase component
    of two dimensional discrete Fourier transform.
    :return: numpy array with same shape as phase denoting pixel reconstruction from phase only
    while setting magnitude=1
    """
    magnitude = np.ones(phase.shape)
    data_f = np.concatenate((magnitude, phase), axis=3)
    return ifourier(data_f)


def pixel2phase(data):
    """
    converts each channel of images in the data to phase component of its 2-dimensional discrete Fourier transform.
    :param data: numpy array with shape (nb_images, img_rows, img_cols, nb_channels)
    :return: numpy array with same shape as data
    """
    channels = data.shape[-1]
    return fourier(data)[:, :, :, channels:]


def augment(path_to_training_data, nb_samples):
    if os.path.exists(os.path.join(path_to_training_data, 'output')):
        print('Augmented data is already saved to {0}'.format(os.path.join(path_to_training_data, 'output')))
        return
    p = Augmentor.Pipeline(path_to_training_data)

    # augmentation pipeline
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=.5, min_factor=0.8, max_factor=1.2)
    p.random_distortion(probability=.5, grid_width=6, grid_height=6, magnitude=1)

    print(p.status())
    print('{0} samples generated and saved to {1}'.format(nb_samples, os.path.join(path_to_training_data, 'output')))
    p.sample(nb_samples)


def load_images_from_folder(folder):
    """
    loads png images and labels from folder. The folder must contain subfolders of images for different labels.
    For example, it should contain subfolders 0, 1, 2, ...  where each subfolder contains images of the
    corresponding label. Note that the first time this function is called, it saves images and labels as npy
    files in the path of folder for later reference.
    :param folder: string of path to the folder.
    :return: a tuple (images, labels) of numpy arrays
    """
    images = []
    labels = []
    if 'images.npy' in os.listdir(folder) and 'labels.npy' in os.listdir(folder):
        images = np.load(os.path.join(folder, 'images.npy'))
        labels = np.load(os.path.join(folder, 'labels.npy'))
    else:
        from PIL import Image
        for subfolder in os.listdir(folder):
            if subfolder.isdigit():
                for filename in os.listdir(os.path.join(folder, subfolder)):
                    img = Image.open(os.path.join(folder, subfolder, filename))
                    img_arr = np.array(img, dtype='uint8')
                    images.append(img_arr)
                    labels.append(int(subfolder))
        perm = np.random.permutation(len(labels))
        images = np.array(images)[perm]
        labels = np.array(labels)[perm]
        np.save(os.path.join(folder, 'images'), images)
        np.save(os.path.join(folder, 'labels'), labels)
    return images, labels


def log_attack(attack_name, adv_x, perturbation_strength, attack_params):
    """
    saves adv_x with name perturbation_strength in folder with attack_name
    :param attack_name: string name of attack
    :param adv_x: numpy array of adversarial images
    :param perturbation_strength: scalar showing perturbation strength of the adversarial images.
    used for filename of adv_x
    :param attack_params: dictionary of parameters of the attack
    """
    directory = os.path.join('Attack Logs', attack_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    import json
    with open(os.path.join(directory, 'params' + str(perturbation_strength) + '.txt'), 'w') as file:
        file.write(json.dumps(attack_params))  # use `json.loads` to do the reverse
    np.save(os.path.join(directory, str(perturbation_strength)), adv_x)


def _read_attack(attack_name, perturbation_strength):
    """
    loads adv_x with perturbation_strength from folder with attack_name
    :param attack_name: string of attack name used for folder to save
    :param perturbation_strength: a float or string of attack file
    """
    filename = os.path.join('Attack Logs', attack_name, str(perturbation_strength) + '.npy')
    return np.load(filename)


def measure_perturbation(x, adv_x, order):
    """
    average perturbation between x and adv_x. Note that each image is converted to
    a vector of size (img_rows*img_cols*nb_channels) and then norm is calculated.
    :param x: numpy array with shape (nb_images, img_rows, img_cols, nb_channels)
    :param adv_x: numpy array with same shape as x
    :param order: order of the norm (mimics numpy) possible values are np.inf, 1 or 2
    :return: a scalar denoting perturbation between x and adv_x averaged over images.
    """
    nb_images, _, _, _ = x.shape
    dev = (x-adv_x).reshape(nb_images, -1)
    dev_norms = np.linalg.norm(dev, order, axis=1)
    return np.mean(dev_norms)


def random_perturb(x, perturbation_strength, order):
    """
    randomly perturbes pixels of x with perturbation_strength such that
    measure_perturbation(x, random_perturb(x, perturbation_strength, order), order) = perturbation_strength.
    For order=np.inf each pixel is perturbed with either -perturbation_strenth or perturbation_strength.
    For order = 1 and order = 2, images of the pixel are perturbed with a uniform random noise with mean zero.
    :param x: numpy array with shape (nb_images, img_rows, img_cols, nb_channels)
    :param perturbation_strength: a scalar that is strength of noise.
    :param order: order of the norm (mimics numpy) possible values are np.inf, 1 or 2
    :return: numpy array with same shape as x denoting random perturbation of pixels of x with perturbation_strength
    """
    nb_images, img_rows, img_cols, nb_channels = x.shape
    if order == np.inf:
        dev = (np.random.randint(0, 2, size=nb_images*img_rows*img_cols*nb_channels) * 2 * perturbation_strength - perturbation_strength)
    elif order == 1:
        tmp = np.random.rand(nb_images, img_rows*img_cols*nb_channels) - 0.5
        coef = perturbation_strength / np.sum(np.abs(tmp), axis=1)
        dev = tmp * np.expand_dims(coef, axis=1)
    elif order == 2:
        tmp = np.random.rand(nb_images, img_rows*img_cols*nb_channels) - 0.5
        coef = perturbation_strength / np.linalg.norm(tmp, 2, axis=1)
        dev = tmp * np.expand_dims(coef, axis=1)
    else:
        raise(ValueError('order should be np.inf, 1 or 2'))
    return x + dev.reshape(x.shape)


def read_attack(attack_name):
    """
    reads a dictionary whose keys are perturbation strength and values are numpy array of adversarial test images
    :param attack_name: string of attack name (the folder containing adversarial images)
    :return: a dictionary with (key, value) as (scalar of perturbation strength, numpy array of adversarial images)
    """
    directory = os.path.join('Attack Logs', attack_name)
    out = dict()
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            path_to_file = os.path.join(directory, filename)
            out[np.float(os.path.splitext(filename)[0])] = np.load(path_to_file)
    return out


def log_plot_data(attack_name, header, arr):
    """
    concatenates numpy arrays in arr and saves them as 'plot_data.csv'.
    :param attack_name: string of attack name (the folder in which data is to be logged)
    :param header: list of strings denoting header name for element of arr
    :param arr: list of numpy arrays to be logged. For example: [strength, adv_acc, ...]
    """
    import pandas as pd
    directory = os.path.join('Attack Logs', attack_name)
    tmp = np.concatenate(tuple([np.array(a).reshape(-1, 1) for a in arr]), axis=1)
    df = pd.DataFrame(tmp, columns=header)
    df.to_csv(os.path.join(directory, 'plot_data'), index=False)


def load_plot_data(attack_name):
    """
    reads data saved with log_plot_data
    :param attack_name: string of attack name (the folder to read from)
    :return: a pandas dataFrame containing plot data.
    """
    import pandas as pd
    path = os.path.join('Attack Logs', attack_name, 'plot_data')
    df = pd.read_csv(path)
    return df


def mnist_denoise(data):
    """
    denoise MNIST data by making background black.
    :param data: numpy array of shape (nb_images, img_rows, img_cols, nb_channels)
    :return: numpy array of denoised data with the same shape as input
    """
    threshold = .45
    data[data < threshold] = 0
    return data
