import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns

def get_kernels():
    kernels = []
    kernels.append(('Identity',
                     np.array([[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]])))
    kernels.append(('Edge Detection1', 
                     np.array([[1, 0, -1],
                               [0, 0, 0],
                               [-1, 0, 1]])))
    kernels.append(('Edge Detection2', 
                     np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])))
    kernels.append(('Edge Detection3', 
                     np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]])))
    kernels.append(('Sharpen', 
                     np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])))
    kernels.append(('Box Blur', 
                     np.array([[1/9, 1/9, 1/9],
                             [1/9, 1/9, 1/9],
                             [1/9, 1/9, 1/9]])))
    kernels.append(('Gaussian Blur', 
                     np.array([[1/16, 1/8, 1/16],
                               [1/8, 1/4, 1/8],
                               [1/16, 1/8, 1/16]])))
    return kernels

def pad_img(img, kernel):
    pad_height = (kernel.shape[0] - 1) // 2
    pad_width = (kernel.shape[1] - 1) // 2

    if len(img.shape) == 2:
        return np.pad(img, ((pad_height, pad_height),
                            (pad_width, pad_width)), 'constant')
    return np.pad(img, ((pad_height, pad_height),
                        (pad_width, pad_width),
                        (0, 0)), 'constant')

def convolve2D(img, kernel):
    is_gray_scale = len(img.shape) == 2
    
    if is_gray_scale:
        img = np.expand_dims(img, axis=2)

    img = pad_img(img, kernel)

    height, width = img.shape[:2]

    new_img = []

    for i in range(height - kernel.shape[0] + 1):
        row = []
        for j in range(width - kernel.shape[1] + 1):
            channels = []
            for k in range(img.shape[2]):
                slice = img[i:i+kernel.shape[0], j:j+kernel.shape[1], k]
                channels.append(np.expand_dims(np.sum(slice * kernel, keepdims=True), axis=0))
            row.append(np.concatenate(channels, axis=2))
        new_img.append(np.concatenate(row, axis=1))

    res = np.maximum(np.concatenate(new_img, axis=0), 0).astype('uint8')

    if is_gray_scale:
        return res[:, :, 0]
    return res

def plot_with_kernels(img):
    kernels = get_kernels()
    n_sub_plots = len(kernels)
    
    plt.figure('kernels', figsize=(20, 20))

    for i, kernel in enumerate(kernels):
        plt.subplot(n_sub_plots, 3, (i*3) + 1)
        plt.text(0.5, 0.5, kernel[0],
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=15)
        plt.axis('off')

        plt.subplot(n_sub_plots, 3, (i * 3) + 2)
        sns.heatmap(kernel[1], annot=True, cmap='YlGnBu')
        plt.axis('off')

        plt.subplot(n_sub_plots, 3, (i+1) * 3)
        
        img_ = convolve2D(img, kernel[1])

        if len(img_.shape) == 2:
            plt.imshow(img_, cmap='gray')
        else:
            plt.imshow(img_)

        plt.axis('off')
    plt.show()

def imshow(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def get_activations():
    acts = []
    acts.append(('Sigmoid', lambda x: 1/ (1 + np.exp(-x))))
    acts.append(('Hyperbolic Tangent', lambda x: np.tanh(x)))
    acts.append(('Rectified Linear Unit', lambda x: np.maximum(x, 0)))
    return acts

def plot_activations():
    acts = get_activations()
    x = np.arange(-20, 20, 0.01)

    n_sub_plots = len(acts)
    plt.figure('activations', figsize=(20, 15))
    for i in range(n_sub_plots):
        plt.subplot(n_sub_plots, 3, (i*3) + 1)
        plt.text(0.5, 0.5, acts[i][0],
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=15)
        plt.axis('off')

        plt.subplot(n_sub_plots, 3, (i * 3) + 2)
        eq = np.array(Image.open('images/' + acts[i][0] + '.jpg'))
        plt.imshow(eq)
        plt.axis('off')

        plt.subplot(n_sub_plots, 3, (i+1) * 3)
        plt.plot(x, acts[i][1](x))
        plt.xlabel('x')
        plt.ylabel('y')
    plt.show()
