from exp1.bitmap import Bitmap
from exp1.converter import Converter
from exp2.dct import compute_psnr
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.measure import compare_ssim
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter

class Operations:
    def __init__(self):
        pass

    @staticmethod
    def hist(gray, can=256):
        # gray：值域为[0.255]的灰度图像；can：直方图的区间个数
        step = int(np.ceil(256 / can))
        prob = [0 for _ in range(can)]
        num = [0 for _ in range(can)]
        h, w = gray.shape[0], gray.shape[1]
        for i in range(h):
            for j in range(w):
                pix_val = gray[i, j]
                num[pix_val//step] += 1
        for i in range(len(prob)):
            prob[i] = num[i] / (h*w)
        return prob, num

    @staticmethod
    def hist_normalization(gray, min_max=[0, 255], can_num=256):
        data = gray.copy()
        pix_min, pix_max = np.amin(gray), np.amax(gray)
        tar_min, tar_max = min_max[0], min_max[1]
        h, w = gray.shape[0], gray.shape[1]
        for i in range(h):
            for j in range(w):
                data[i, j] = (tar_max - tar_min) / (pix_max - pix_min) * (data[i, j] - pix_min) + tar_min
        prob_before, _ = Operations.hist(gray, can_num)
        prob_after, _ = Operations.hist(data, can_num)
        plt.subplot(2, 2, 1)
        plt.bar(range(len(prob_before)), prob_before)
        plt.xlabel("pixel value")
        plt.ylabel("probability")
        plt.title("Original Hist")
        plt.subplot(2, 2, 2)
        plt.imshow(gray, cmap=plt.cm.gray)
        plt.title("Original Image")
        plt.subplot(2, 2, 3)
        plt.bar(range(len(prob_after)), prob_after)
        plt.xlabel("pixel value")
        plt.ylabel("probability")
        plt.title("Normalized Hist")
        plt.subplot(2, 2, 4)
        plt.imshow(data, cmap=plt.cm.gray)
        plt.title("Normalized Image")
        plt.show()
        return data

    @staticmethod
    def hist_equalization(gray, can_num=256):
        data = gray.copy()
        h, w = gray.shape[0], gray.shape[1]
        hist_before, _ = Operations.hist(gray, can=can_num)
        step = int(np.ceil(256 / can_num))
        acc_prob = [0 for _ in range(can_num)]
        acc_prob[0] = hist_before[0]
        for i in range(1, can_num):
            acc_prob[i] = acc_prob[i-1] + hist_before[i]
        map = [0 for _ in range(can_num)]
        for i in range(can_num):
            map[i] = int(np.floor((can_num-1)*acc_prob[i]+0.5))
        hist_after = [0 for _ in range(can_num)]
        for i in range(can_num):
            hist_after[map[i]] += hist_before[i]
        for i in range(h):
            for j in range(w):
                can = gray[i, j] // step
                new_can = map[can]
                data[i, j] = new_can * step + step//2
        plt.subplot(2, 2, 1)
        plt.bar(range(len(hist_before)), hist_before)
        plt.xlabel("pixel value")
        plt.ylabel("probability")
        plt.title("Original Hist")
        plt.subplot(2, 2, 2)
        plt.imshow(gray, cmap=plt.cm.gray)
        plt.title("Original Image")
        plt.subplot(2, 2, 3)
        plt.bar(range(len(hist_after)), hist_after)
        plt.xlabel("pixel value")
        plt.ylabel("probability")
        plt.title("Equalized Hist")
        plt.subplot(2, 2, 4)
        plt.imshow(data, cmap=plt.cm.gray)
        plt.title("Normalized Image")
        plt.show()
        return data

    @staticmethod
    def add_guassian_noise(img, sigma):
        noise = np.random.normal(0.0, sigma, img.shape)
        img_ = (img + noise).round().clip(0, 255).astype(np.uint8)
        return img_

    @staticmethod
    def add_salty_noise(img, SNR):
        img_ = img.copy()
        h, w = img_.shape
        mask = np.random.choice((0, 1, 2), size=(h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
        img_[mask == 1] = 255
        img_[mask == 2] = 0
        return img_.round().clip(0, 255).astype(np.uint8)

    @staticmethod
    def convolve2d(img, kernel, mode='default'):
        kh, kw = kernel.shape[0], kernel.shape[1]
        ih, iw = img.shape[0], img.shape[1]
        if len(img.shape) == 2:
            ic = 1
            img_ = np.expand_dims(img.copy(), 2)
        else:
            ic = img.shape[2]
            img_ = img.copy()
        if len(kernel.shape) == 2:
            kernel = np.repeat(np.expand_dims(kernel, 2), ic, axis=2)
        elif kernel.shape[2] != ic:
            raise Exception("Channel Mismatch")
        if mode == 'same':
            img_ = np.pad(img_, ((kh//2, kh//2), (kw//2, kw//2), (0, 0)), 'constant')
            res = np.zeros((ih, iw))
        else:
            res = np.zeros((ih-kh+1, iw-kw+1, ic))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                patch = img_[i:i+kh, j:j+kh]
                res[i, j] = np.sum(patch * kernel)
        return res.round().clip(0, 255).astype(np.uint8)

    @staticmethod
    def mean_filter(img, kernel_size):
        kernel = (1/kernel_size**2) * np.ones((kernel_size, kernel_size))
        return Operations.convolve2d(img, kernel, 'same')

    @staticmethod
    def median_filter(img, kernel_size):
        img_ = np.pad(img.copy(), ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), 'constant')
        res = np.zeros(img.shape)
        for i in range(res.shape[0]):
            patch = img_[i:i + kernel_size, 0:kernel_size]
            _, hist = Operations.hist(patch)
            mdn = np.sum([i*hist[i] for i in range(256)]) / (kernel_size**2)
            mdn = int(np.floor(mdn))
            Ltmdn = int(np.sum(patch<mdn))
            res[i, 0] = mdn
            for j in range(1, res.shape[1]):
                left = img_[i:i+kernel_size, j-1]
                right = img_[i:i+kernel_size, j+kernel_size-1]
                for pix in left:
                    hist[pix] -= 1
                    if pix < mdn:
                        Ltmdn -= 1
                for pix in right:
                    hist[pix] += 1
                    if pix < mdn:
                        Ltmdn += 1
                th = int((kernel_size**2-1)/2)
                while Ltmdn > th:
                    mdn = mdn - 1
                    Ltmdn = Ltmdn - hist[mdn]
                while Ltmdn + hist[mdn] <= th:
                    Ltmdn += hist[mdn]
                    mdn += 1
                res[i, j] = mdn
        return res

    @staticmethod
    def edge_detect(gray, patten='Sobel', mode='v'):
        patten_collections = {'Sobel': {'v': np.array([[-1,  0,  1],
                                                       [-2,  0,  2],
                                                       [-1,  0,  1]]),
                                        'h': np.array([[-1, -2, -1],
                                                       [ 0,  0,  0],
                                                       [ 1,  2,  1]])}
                              }
        return Operations.convolve2d(gray, patten_collections[patten][mode], 'same')

    @staticmethod
    def dwt_denoising(gray):
        coeffs = pywt.wavedec2(gray, 'db1', level=2)
        ths = [23.38, 10.12]
        for i in range(1, len(coeffs)):
            coeffs[i] = tuple([pywt.threshold(v, ths[i-1], 'hard') for v in coeffs[i]])
        return pywt.waverec2(coeffs, 'db1')

    @staticmethod
    def thresholing_binarize(gray, threshold):
        img_ = gray.copy()
        new_im = np.zeros_like(img_, dtype=bool)

        for i in range(img_.shape[0]):
            for j in range(img_.shape[1]):
                new_im[i, j] = True if img_[i, j] >= threshold else False

        return new_im

    @staticmethod
    def half_tone(gray):
        m = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                      [[0, 1, 0], [0, 0, 0], [0, 0, 1]],
                      [[1, 1, 0], [0, 0, 0], [0, 0, 1]],
                      [[1, 1, 0], [0, 0, 0], [1, 0, 1]],
                      [[1, 1, 1], [0, 0, 0], [1, 0, 1]],
                      [[1, 1, 1], [0, 0, 1], [1, 0, 1]],
                      [[1, 1, 1], [0, 0, 1], [1, 1, 1]],
                      [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
                      [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
        m = m * 256
        h, w = gray.shape
        half_img = np.zeros((3*h, 3*w))
        step = int(np.ceil(256 / 10))
        img_ten = np.fix(gray / step).astype(np.uint8)
        for i in range(h):
            for j in range(w):
                gray_level = img_ten[i, j]
                half_img[3*i:3*(i+1), 3*j:3*(j+1)] = m[gray_level]
        return half_img


def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = compare_ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s


def test(file_path):
    img = Bitmap(file_path)
    rgb_data = img.get_data()
    test_data = np.mean(rgb_data, axis=-1).astype(np.uint8)
    test_data_1 = np.ceil(test_data / 255.0 * 100).astype(np.uint8)
    Operations.hist_normalization(test_data_1)
    Operations.hist_equalization(test_data_1)

    img_g = Operations.add_guassian_noise(test_data, 15)
    img_g_filtered = Operations.mean_filter(img_g, 3)
    # img_g_filtered = Operations.median_filter(img_g, 3)
    plt.subplot(1, 2, 1)
    plt.imshow(img_g, cmap=plt.cm.gray)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(img_g_filtered, cmap=plt.cm.gray)
    plt.title("Mean-Filtered Image")
    plt.show()
    print("Mean Filter: PSNR={}, SSIM={}".format(compute_psnr(test_data, img_g_filtered),
                                                 compare_ssim(img_g_filtered, test_data)))

    img_s = Operations.add_salty_noise(test_data, 0.9)
    img_s_filtered = Operations.median_filter(img_s, 3)
    # img_s_filtered = Operations.mean_filter(img_s, 3)
    plt.subplot(1, 2, 1)
    plt.imshow(img_s, cmap=plt.cm.gray)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(img_s_filtered, cmap=plt.cm.gray)
    plt.title("Median-Filtered Image")
    plt.show()
    print("Mean Filter: PSNR={}, SSIM={}".format(compute_psnr(img_s_filtered, test_data),
                                                 compare_ssim(img_s_filtered, test_data)))
    #
    img_wavelet = Operations.dwt_denoising(img_g)
    plt.subplot(1, 2, 1)
    plt.imshow(img_g, cmap=plt.cm.gray)
    plt.title("Noisy Image")
    plt.subplot(1, 2, 2)
    plt.imshow(img_wavelet, cmap=plt.cm.gray)
    plt.title("Reconstructed Image")
    plt.show()

    edge_v = Operations.edge_detect(test_data, 'Sobel', 'v')
    edge_h = Operations.edge_detect(test_data, 'Sobel', 'h')
    plt.subplot(1, 3, 1)
    plt.imshow(test_data, cmap=plt.cm.gray)
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(edge_h, cmap=plt.cm.gray)
    plt.title("H Edges")
    plt.subplot(1, 3, 3)
    plt.imshow(edge_v, cmap=plt.cm.gray)
    plt.title("V Edges")
    plt.show()

    bi_img = Operations.thresholing_binarize(test_data, 125)
    plt.subplot(1, 2, 1)
    plt.imshow(test_data, cmap=plt.cm.gray)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(bi_img, cmap=plt.cm.gray)
    plt.title("Binary Image")
    plt.show()

    half_img = Operations.half_tone(test_data)
    plt.subplot(1, 2, 1)
    plt.imshow(test_data, cmap=plt.cm.gray)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(half_img, cmap=plt.cm.gray)
    plt.title("Half-tone Image")
    plt.show()


test("../exp1/lena512color.bmp")