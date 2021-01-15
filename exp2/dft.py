from exp1.bitmap import Bitmap
from exp1.converter import Converter
import numpy as np
import matplotlib.pyplot as plt
import time


class DFT:
    def __init__(self):
        pass

    @staticmethod
    def forward(mat, patch_size=8):
        N = patch_size
        A = np.zeros((N, N), dtype=np.complex_)
        for i in range(N):
            for j in range(N):
                A[i, j] = np.exp(-2j*np.pi*(i*j/N))
        A = np.asmatrix(A)
        ycbcr = Converter.rgb2ycbcr(mat)
        f = ycbcr[:, :, 0]
        h, w = f.shape[0], f.shape[1]
        f = f[0:int(np.floor(h/N)*N), 0:int(np.floor(w/N)*N)]
        F = np.zeros(f.shape, dtype=np.complex_)
        start = time.time()
        for i in range(0, h, N):
            for j in range(0, w, N):
                patch = f[i:i+N, j:j+N]
                # 对patch做傅里叶变换
                F[i:i+N, j:j+N] = (1/N)*np.matmul(np.matmul(A, patch), A.H)
        end = time.time()
        print("DFT costs %.2f s"%(end-start))
        return F

    @staticmethod
    def show_magnitude_phase(F):
        magnitude = np.absolute(F)
        phase = np.arctan2(np.imag(F), np.real(F))
        magnitude_normalized = (magnitude - np.amin(magnitude)) / (np.amax(magnitude) - np.amin(magnitude)) # 归一化以便于显示
        phase_normalized = (phase - np.min(phase)) / (np.amax(phase) - np.amin(phase)) # 归一化以便于显示
        figure, ax = plt.subplots(1, 2)
        ax[0].imshow(magnitude_normalized, cmap=plt.cm.gray)
        ax[0].set_title("Magnitude")
        ax[1].imshow(phase_normalized, cmap=plt.cm.gray)
        ax[1].set_title("Phase")
        plt.show()
        return None

    @staticmethod
    def backward(F, patch_size=8):
        N = patch_size
        A = np.zeros((N, N), dtype=np.complex_)
        for i in range(N):
            for j in range(N):
                A[i, j] = np.exp(-2j * np.pi * (i * j / N))
        A = np.asmatrix(A)
        h, w = F.shape[0], F.shape[1]
        F = F[0:int(np.floor(h/N)*N), 0:int(np.floor(w/N)*N)]
        f = np.zeros(F.shape)
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = F[i:i + N, j:j + N]
                # 对patch做傅里叶逆变换
                f[i:i + N, j:j + N] = (1/N) * np.matmul(np.matmul(A.H, patch), A)
        return f.round()


    @staticmethod
    def reconstruct_from_magnitude_and_phase(F):
        magnitude = np.absolute(F)
        phase = np.arctan2(np.imag(F), np.real(F))
        phase_ = np.exp(1j*phase)
        re_mag = DFT.backward(magnitude)
        re_pha = DFT.backward(phase_)
        figure, ax = plt.subplots(1, 2)
        ax[0].imshow(re_mag, cmap=plt.cm.gray)
        ax[0].set_title("Magnitude Reconstruction")
        ax[1].imshow(re_pha, cmap=plt.cm.gray)
        ax[1].set_title("Phase Reconstruction")
        plt.show()


def test(file_path):
    img = Bitmap(file_path)
    F = DFT.forward(img.get_data())
    DFT.show_magnitude_phase(F)
    DFT.reconstruct_from_magnitude_and_phase(F)

test("../exp1/lena512color.bmp")