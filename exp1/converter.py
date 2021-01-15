from exp1.bitmap import Bitmap
import numpy as np
import matplotlib.pyplot as plt

class Converter:
    def __init__(self):
        pass

    @staticmethod
    def rgb2yuv(rgb_data):
        rgb_to_yuv = np.array([[0.299, 0.587, 0.114],
                               [-0.147, -0.289, 0.436],
                               [0.615, -0.515, -0.100]])
        yuv_data = np.matmul(rgb_data, rgb_to_yuv.transpose())
        figure, ax=plt.subplots(2, 2)
        ax[0][0].imshow(rgb_data)
        ax[0][0].set_title("RGB Figure")
        ax[0][1].imshow(yuv_data[:, :, 0], cmap=plt.cm.gray)
        ax[0][1].set_title("Y Channel")
        ax[1][0].imshow(yuv_data[:, :, 1], cmap=plt.cm.gray)
        ax[1][0].set_title("U Channel")
        ax[1][1].imshow(yuv_data[:, :, 2], cmap=plt.cm.gray)
        ax[1][1].set_title("V Channel")
        plt.show()
        return yuv_data

    @staticmethod
    def yuv2rgb(yuv_data):
        yuv_to_rgb = np.array([[1, 0, 1.140],
                               [1, -0.395, -0.581],
                               [1, 2.032, 0]])
        rgb_data = np.matmul(yuv_data, yuv_to_rgb.transpose()).round().astype(np.uint8)
        figure, ax = plt.subplots(2, 2)
        ax[0][0].imshow(yuv_data[:, :, 0], cmap=plt.cm.gray)
        ax[0][0].set_title("Y Channel")
        ax[0][1].imshow(yuv_data[:, :, 1], cmap=plt.cm.gray)
        ax[0][1].set_title("U Channel")
        ax[1][0].imshow(yuv_data[:, :, 2], cmap=plt.cm.gray)
        ax[1][0].set_title("V Channel")
        ax[1][1].imshow(rgb_data)
        ax[1][1].set_title("RGB Figure")
        plt.show()
        return rgb_data

    @staticmethod
    def rgb2yiq(rgb_data):
        rgb_to_yiq = np.array([[0.299, 0.587, 0.114],
                               [0.596, -0.274, -0.322],
                               [0.211, -0.523, 0.312]])
        yiq_data = np.matmul(rgb_data, rgb_to_yiq.transpose())
        figure, ax = plt.subplots(2, 2)
        ax[0][0].imshow(rgb_data)
        ax[0][0].set_title("RGB Figure")
        ax[0][1].imshow(yiq_data[:, :, 0], cmap=plt.cm.gray)
        ax[0][1].set_title("Y Channel")
        ax[1][0].imshow(yiq_data[:, :, 1], cmap=plt.cm.gray)
        ax[1][0].set_title("I Channel")
        ax[1][1].imshow(yiq_data[:, :, 2], cmap=plt.cm.gray)
        ax[1][1].set_title("Q Channel")
        plt.show()
        return yiq_data

    @staticmethod
    def yiq2rgb(yiq_data):
        yiq_to_rgb = np.array([[1, 0.956, 0.621],
                               [1, -0.272, -0.647],
                               [1, -1.106, 1.703]])
        rgb_data = np.matmul(yiq_data, yiq_to_rgb.transpose()).round().astype(np.uint8)
        figure, ax = plt.subplots(2, 2)
        ax[0][0].imshow(yiq_data[:, :, 0], cmap=plt.cm.gray)
        ax[0][0].set_title("Y Channel")
        ax[0][1].imshow(yiq_data[:, :, 1], cmap=plt.cm.gray)
        ax[0][1].set_title("I Channel")
        ax[1][0].imshow(yiq_data[:, :, 2], cmap=plt.cm.gray)
        ax[1][0].set_title("Q Channel")
        ax[1][1].imshow(rgb_data)
        ax[1][1].set_title("RGB Figure")
        plt.show()
        return rgb_data


    @staticmethod
    def rgb2ycbcr(rgb_data):
        rgb_to_ycbcr = np.array([[0.299, 0.857, 0.114],
                                 [0.5, -0.4187, -0.0813],
                                 [-0.1687, -0.3313, 0.5]])
        ycbcr_data = np.matmul(rgb_data, rgb_to_ycbcr.transpose())
        figure, ax = plt.subplots(2, 2)
        ax[0][0].imshow(rgb_data)
        ax[0][0].set_title("RGB Figure")
        ax[0][1].imshow(ycbcr_data[:, :, 0], cmap=plt.cm.gray)
        ax[0][1].set_title("Y Channel")
        ax[1][0].imshow(ycbcr_data[:, :, 1], cmap=plt.cm.gray)
        ax[1][0].set_title("Cb Channel")
        ax[1][1].imshow(ycbcr_data[:, :, 2], cmap=plt.cm.gray)
        ax[1][1].set_title("Cr Channel")
        plt.show()
        return ycbcr_data

    @staticmethod
    def ycbcr2rgb(ycbcr_data):
        rgb_to_ycbcr = np.array([[0.299, 0.857, 0.114],
                                 [0.5, -0.4187, -0.0813],
                                 [-0.1687, -0.3313, 0.5]])
        ycbcr_to_rgb = np.linalg.inv(rgb_to_ycbcr)
        rgb_data = np.matmul(ycbcr_data, ycbcr_to_rgb.transpose()).round().astype(np.uint8)
        figure, ax = plt.subplots(2, 2)
        ax[0][0].imshow(ycbcr_data[:, :, 0], cmap=plt.cm.gray)
        ax[0][0].set_title("Y Channel")
        ax[0][1].imshow(ycbcr_data[:, :, 1], cmap=plt.cm.gray)
        ax[0][1].set_title("Cb Channel")
        ax[1][0].imshow(ycbcr_data[:, :, 2], cmap=plt.cm.gray)
        ax[1][0].set_title("Cr Channel")
        ax[1][1].imshow(rgb_data)
        ax[1][1].set_title("RGB Figure")
        plt.show()
        return rgb_data

    @staticmethod
    def rgb2xyz(rgb_data):
        rgb_to_xyz = np.array([[0.490, 0.310, 0.2],
                               [0.177, 0.813, 0.011],
                               [0.0, 0.01, 0.99]])
        xyz_data = np.matmul(rgb_data, rgb_to_xyz.transpose())
        figure, ax = plt.subplots(2, 2)
        ax[0][0].imshow(rgb_data)
        ax[0][0].set_title("RGB Figure")
        ax[0][1].imshow(xyz_data[:, :, 0], cmap=plt.cm.gray)
        ax[0][1].set_title("X Channel")
        ax[1][0].imshow(xyz_data[:, :, 1], cmap=plt.cm.gray)
        ax[1][0].set_title("Y Channel")
        ax[1][1].imshow(xyz_data[:, :, 2], cmap=plt.cm.gray)
        ax[1][1].set_title("Z Channel")
        plt.show()
        return xyz_data

    @staticmethod
    def xyz2rgb(xyz_data):
        rgb_to_xyz = np.array([[0.490, 0.310, 0.2],
                               [0.177, 0.813, 0.011],
                               [0.0, 0.01, 0.99]])
        xyz_to_rgb = np.linalg.inv(rgb_to_xyz)
        rgb_data = np.matmul(xyz_data, xyz_to_rgb.transpose()).round().astype(np.uint8)
        figure, ax = plt.subplots(2, 2)
        ax[0][0].imshow(xyz_data[:, :, 0], cmap=plt.cm.gray)
        ax[0][0].set_title("X Channel")
        ax[0][1].imshow(xyz_data[:, :, 1], cmap=plt.cm.gray)
        ax[0][1].set_title("Y Channel")
        ax[1][0].imshow(xyz_data[:, :, 2], cmap=plt.cm.gray)
        ax[1][0].set_title("Z Channel")
        ax[1][1].imshow(rgb_data)
        ax[1][1].set_title("RGB Figure")
        plt.show()
        return rgb_data

    @staticmethod
    def rgb2hsi(rgb_data):
        height, width, channels = rgb_data.shape
        hsi_data = np.zeros((height, width, channels))
        for i in range(height):
            for j in range(width):
                r, g, b = int(rgb_data[i, j, 0]), int(rgb_data[i, j, 1]), int(rgb_data[i, j, 2])
                hsi_data[i, j, 0] = np.mean(rgb_data[i, j, :]) # I channel
                hsi_data[i, j, 1] = 1 - 3.0/np.sum(rgb_data[i, j, :]) * np.min(rgb_data[i, j, :]) # S channel
                hsi_data[i, j, 2] = np.arccos((2*r-g-b)/2/(np.sqrt((r-g)**2+(r-b)*(g-b)))) # H channel
                if g < b:
                    hsi_data[i, j, 2] = 2*np.pi - hsi_data[i, j, 2]

        figure, ax = plt.subplots(2, 2)
        ax[0][0].imshow(rgb_data)
        ax[0][0].set_title("RGB Figure")
        ax[0][1].imshow(hsi_data[:, :, 2], cmap=plt.cm.gray)
        ax[0][1].set_title("H Channel")
        ax[1][0].imshow(hsi_data[:, :, 1], cmap=plt.cm.gray)
        ax[1][0].set_title("S Channel")
        ax[1][1].imshow(hsi_data[:, :, 0], cmap=plt.cm.gray)
        ax[1][1].set_title("I Channel")
        plt.show()
        return hsi_data

    @staticmethod
    def hsi2rgb(hsi_data):
        height, width, channels = hsi_data.shape
        rgb_data = np.zeros((height, width, channels))
        for k in range(height):
            for j in range(width):
                i, s, h = hsi_data[k, j, :]
                if h <= 2/3*np.pi:
                    rgb_data[k, j, 2] = i*(1-s)
                    rgb_data[k, j, 0] = i*(1+s*np.cos(h)/np.cos(np.pi/3-h))
                    rgb_data[k, j, 1] = 3*i-rgb_data[k, j, 0]-rgb_data[k, j, 2]
                elif h > 2/3*np.pi and h <= 4/3*np.pi:
                    rgb_data[k, j, 0] = i * (1 - s)
                    rgb_data[k, j, 1] = i * (1 + s * np.cos(h-2/3*np.pi) / np.cos(np.pi - h))
                    rgb_data[k, j, 2] = 3 * i - rgb_data[k, j, 1] - rgb_data[k, j, 0]
                else:
                    rgb_data[k, j, 1] = i * (1 - s)
                    rgb_data[k, j, 2] = i * (1 + s * np.cos(h-4/3*np.pi) / np.cos(5/3*np.pi - h))
                    rgb_data[k, j, 0] = 3 * i - rgb_data[k, j, 1] - rgb_data[k, j, 2]

        rgb_data = rgb_data.round().astype(np.uint8)
        figure, ax = plt.subplots(2, 2)
        ax[0][0].imshow(hsi_data[:, :, 2], cmap=plt.cm.gray)
        ax[0][0].set_title("H Channel")
        ax[0][1].imshow(hsi_data[:, :, 1], cmap=plt.cm.gray)
        ax[0][1].set_title("S Channel")
        ax[1][0].imshow(hsi_data[:, :, 0], cmap=plt.cm.gray)
        ax[1][0].set_title("I Channel")
        ax[1][1].imshow(rgb_data)
        ax[1][1].set_title("RGB Figure")
        plt.show()
        return rgb_data

def test(file_path):
    bmp = Bitmap(file_path)
    # bmp.show()
    rgb_data = bmp.get_data()
    yuv_data = Converter.rgb2yuv(rgb_data)
    Converter.yuv2rgb(yuv_data)
    yiq_data = Converter.rgb2yiq(rgb_data)
    Converter.yiq2rgb(yiq_data)
    ycbcr_data = Converter.rgb2ycbcr(rgb_data)
    Converter.ycbcr2rgb(ycbcr_data)
    xyz_data = Converter.rgb2xyz(rgb_data)
    Converter.xyz2rgb(xyz_data)
    hsi_data = Converter.rgb2hsi(rgb_data)
    Converter.hsi2rgb(hsi_data)

# test("lena512color.bmp")