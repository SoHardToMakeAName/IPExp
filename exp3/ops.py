from exp1.bitmap import Bitmap
from exp1.converter import Converter
import numpy as np
import matplotlib.pyplot as plt

class Operations:
    def __init__(self):
        pass

    @staticmethod
    def hist(gray, can=256):
        # gray：值域为[0.255]的灰度图像；can：直方图的区间个数
        step = int(np.ceil(256 / can))
        prob = [0 for _ in range(can)]
        h, w = gray.shape[0], gray.shape[1]
        for i in range(h):
            for j in range(w):
                pix_val = gray[i, j]
                prob[pix_val//step] += 1
        for i in range(len(prob)):
            prob[i] = prob[i] / (h*w)
        return prob

    @staticmethod
    def hist_normalization(gray, min_max=[0, 255], can_num=256):
        data = gray.copy()
        pix_min, pix_max = np.amin(gray), np.amax(gray)
        tar_min, tar_max = min_max[0], min_max[1]
        h, w = gray.shape[0], gray.shape[1]
        for i in range(h):
            for j in range(w):
                data[i, j] = (tar_max - tar_min) / (pix_max - pix_min) * (data[i, j] - pix_min) + tar_min
        prob_before = Operations.hist(gray, can_num)
        prob_after = Operations.hist(data, can_num)
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
        hist_before = Operations.hist(gray, can=can_num)
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


def test(file_path):
    img = Bitmap(file_path)
    test_data = img.get_data()[:, :, 0]
    test_data_1 = np.ceil(test_data / 255.0 * 100).astype(np.uint8)
    # Operations.hist_normalization(test_data_1)
    Operations.hist_equalization(test_data_1)

test("../exp1/lena512color.bmp")