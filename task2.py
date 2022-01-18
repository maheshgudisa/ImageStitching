"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used 
image processing techniques: image denoising and edge detection. 
Specifically, you are given a grayscale image with salt-and-pepper noise, 
which is named 'task2.png' for your code testing. 
Note that different image might be used when grading your code. 

You are required to write programs to: 
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint: 
• Zero-padding is needed before filtering or convolution. 
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolution or correlation. 
You should NOT use any other libraries, which provide APIs for convolution/correlation or median filtering.
Please write the convolution code ON YOUR OWN. 
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)

sobel_m = np.array([[1, 2, 0], [2, 0, -2], [0, -2, -1]]).astype(int)
sobel_n = np.array([[0, 2, 1], [-2, 0, 2], [-1, -2, 0]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """
    denoise_img = np.zeros_like(img)
    padded_image = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    padded_image[1:img.shape[0] + 1, 1:img.shape[1] + 1] = img
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            denoise_img[i, j] = np.median(padded_image[i:i + 3, j:j + 3].reshape(9, ), axis=0)
    return denoise_img.astype(int)


def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """
    conv_img = np.zeros_like(img)
    padded_image = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    padded_image[1:img.shape[0] + 1, 1:img.shape[1] + 1] = img
    flipped_kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            conv_img[i, j] = np.sum(np.multiply(flipped_kernel, padded_image[i:i + 3, j:j + 3]))
    conv_img = conv_img.astype(int)
    return conv_img


def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """
    edge_x = convolve2d(img, kernel=sobel_x)
    edge_y = convolve2d(img, kernel=sobel_y)
    norm_edge_x = 255 * (edge_x - np.min(edge_x)) / (np.max(edge_x) - np.min(edge_x))
    norm_edge_y = 255 * (edge_y - np.min(edge_y)) / (np.max(edge_y) - np.min(edge_y))
    edge_mag = np.sqrt(edge_x ** 2 + edge_y ** 2)
    norm_edge_x = norm_edge_x.astype(np.uint8)
    norm_edge_y = norm_edge_y.astype(np.uint8)
    edge_mag = edge_mag.astype(int)
    return norm_edge_x, norm_edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    edge_45 = convolve2d(denoise_img, kernel=sobel_m)
    edge_135 = convolve2d(denoise_img, kernel=sobel_n)
    edge_45 = 255 * (edge_45 - np.min(edge_45)) / (np.max(edge_45) - np.min(edge_45))
    edge_135 = 255 * (edge_135 - np.min(edge_135)) / (np.max(edge_135) - np.min(edge_135))
    edge_45 = edge_45.astype(np.uint8)
    edge_135 = edge_135.astype(np.uint8)
    print("Kernel to detect edges along 45 deg ", sobel_m)
    print("Kernel to detect edges along 135 deg ", sobel_n)
    return edge_45, edge_135


if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)
