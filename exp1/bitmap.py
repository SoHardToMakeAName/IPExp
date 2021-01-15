import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

class Bitmap:
    def __init__(self, file_path):
        self.file_header = OrderedDict()
        self.info_header = OrderedDict()
        self.quad = None
        self.data = None
        self.open(file_path)

    def open(self, file_path):
        with open(file_path, 'rb') as f:

            # 读文件头
            self.file_header['bfType'] = f.read(2)
            self.file_header['bfSize'] = f.read(4)
            self.file_header['bfReserved1'] = f.read(2)
            self.file_header['bfReserved2'] = f.read(2)
            self.file_header['bfOffBits'] = f.read(4)

            # 读信息头
            self.info_header['biSize'] = f.read(4)
            self.info_header['biWidth'] = f.read(4)
            self.info_header['biHeight'] = f.read(4)
            self.info_header['biPlanes'] = f.read(2)
            self.info_header['biBitCount'] = f.read(2)
            self.info_header['biCompression'] = f.read(4)
            self.info_header['biSizeImage'] = f.read(4)
            self.info_header['biXPelsPerMeter'] = f.read(4)
            self.info_header['biYPelsPerMeter'] = f.read(4)
            self.info_header['biClrUsed'] = f.read(4)
            self.info_header['biClrImportant'] = f.read(4)

            # 读调色盘
            if bytes_to_int(self.info_header['biBitCount']) == 8:
                self.quad = f.read(1024)

            # 读数据
            if bytes_to_int(self.info_header['biBitCount']) == 8:
                channels = 1
            elif bytes_to_int(self.info_header['biBitCount']) == 24:
                channels = 3
            else:
                raise Exception("File type not supported")
            width = bytes_to_int(self.info_header['biWidth'])
            height = bytes_to_int(self.info_header['biHeight'])
            offset = (channels * width) % 4
            if offset != 0:
                offset = 4 - offset
            self.data = np.zeros((height, width, channels), dtype=np.uint8)
            line = height - 1
            while line >= 0:
                try:
                    buf = f.read(width*channels)
                    _ = f.read(offset)
                    if not buf:
                        break
                    if channels == 1:
                        self.data[line, :, 0] = np.frombuffer(buf, dtype=np.uint8)
                    else:
                        line_data = np.frombuffer(buf, dtype=np.uint8) # 读入一行
                        self.data[line, :, 0] = line_data[[i * 3 + 2 for i in range(width)]] # R通道
                        self.data[line, :, 1] = line_data[[i * 3 + 1 for i in range(width)]] # G通道
                        self.data[line, :, 2] = line_data[[i * 3 for i in range(width)]] # B通道
                    line -= 1
                except:
                    break

    def show(self):
        figure, ax = plt.subplots(2, 2)
        plt.title("Bitmap")
        ax[0][0].imshow(self.data)
        ax[0][0].set_title("RGB Figure")
        ax[0][1].imshow(self.data[:, :, 0], cmap=plt.cm.gray)
        ax[0][1].set_title("R Channel")
        ax[1][0].imshow(self.data[:, :, 1], cmap=plt.cm.gray)
        ax[1][0].set_title("G Channel")
        ax[1][1].imshow(self.data[:, :, 2], cmap=plt.cm.gray)
        ax[1][1].set_title("B Channel")
        plt.show()

    def save(self, file_path):
        height, width, channels = self.data.shape
        self.info_header['biWidth'] = int_to_bytes(width, 4)
        self.info_header['biHeight'] = int_to_bytes(height, 4)
        offset = (channels * width) % 4
        if offset != 0:
            offset = 4 - offset
        with open(file_path, 'wb') as f:
            for v in self.file_header.values():
                f.write(v)
            for v in self.info_header.values():
                f.write(v)
            if channels != 3:
                f.write(self.quad)
            if channels == 1: # 单通道灰度图
                for i in range(height-1, -1, -1):
                    f.write(self.data[i, :, :].tobytes())
                    if offset != 0:
                        for j in range(offset):
                            f.write(bytes('\0'))
            else: # 三通道彩色图
                data = self.data[..., ::-1] # RGB转BGR
                for i in range(height - 1, -1, -1):
                    f.write(data[i, :, :].tobytes())
                    if offset != 0:
                        for j in range(offset):
                            f.write(bytes('\0'))

    def get_pixel(self, x, y):
        if self.data is None:
            return -1
        width, height, channels = self.data.shape
        if x < 0 or x >= width or y < 0 or y >= height:
            return -1
        return self.data[x, y]

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def height(self):
        return self.data.shape[0]

    @property
    def channels(self):
        return self.data.shape[2]

    def get_data(self):
        return self.data


def int_to_bytes(number, length, byteorder='little'):
    return number.to_bytes(length, byteorder)

def bytes_to_int(bytes, byteorder='little'):
    return int.from_bytes(bytes, byteorder)

def test(file_path):
    bmp = Bitmap(file_path)
    bmp.show()
    # bmp.save('lena.bmp')
    # print(bmp.get_pixel(0, 0))

# test("lena512color.bmp")