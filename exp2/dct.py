from exp1.bitmap import Bitmap
from exp1.converter import Converter
import numpy as np
import matplotlib.pyplot as plt
import time

class DCT:
    def __init__(self):
        pass

    @staticmethod
    def forward(mat, patch_size=8):
        N = patch_size
        A = np.zeros((N, N)) # 变换矩阵A
        for j in range(N):
            A[0, j] = 1/np.sqrt(N)
        for i in range(1, N):
            for j in range(N):
                A[i, j] = np.sqrt(2.0/N) * np.cos(i*np.pi*(j+0.5)/N)
        ycbcr = Converter.rgb2ycbcr(mat)
        f = ycbcr[:, :, 0]
        h, w = f.shape[0], f.shape[1]
        f = f[0:int(np.floor(h / N) * N), 0:int(np.floor(w / N) * N)]
        F = np.zeros(f.shape)
        for i in range(0, h, N):
            for j in range(0, w, N):
                patch = f[i:i+N, j:j+N]
                # 对patch做DCT
                F[i:i+N, j:j+N] = np.matmul(np.matmul(A, patch), A.transpose())
        return F

    @staticmethod
    def backward(F, patch_size=8, top_k=1):
        N = patch_size
        A = np.zeros((N, N))  # 变换矩阵A
        for j in range(N):
            A[0, j] = 1 / np.sqrt(N)
        for i in range(1, N):
            for j in range(N):
                A[i, j] = np.sqrt(2.0 / N) * np.cos(i * np.pi * (j + 0.5) / N)
        h, w = F.shape
        F = F[0:int(np.floor(h / N) * N), 0:int(np.floor(w / N) * N)]
        f = np.zeros(F.shape)
        for i in range(0, h, N):
            for j in range(0, w, N):
                patch = F[i:i+N, j:j+N]
                # 取top_k个系数
                patch_ = np.zeros(patch.shape)
                row, col, s, dir = 0, 0, 0, 0 # (row,col)坐标，s计数，dir四个行进方向0（right）,1（left_down）,2（down）,3（right_up）
                while s < top_k: # zigzag扫描
                    patch_[row, col] = patch[row, col]
                    if dir == 0:
                        col += 1
                        if row == 0:
                            dir = 1
                        if row == N-1:
                            dir = 3
                    elif dir == 1:
                        col -= 1
                        row += 1
                        if row == N-1:
                            dir = 0
                        elif col == 0:
                            dir = 2
                    elif dir == 2:
                        row += 1
                        if col == 0:
                            dir = 3
                        if col == N-1:
                            dir = 1
                    elif dir == 3:
                        col += 1
                        row -= 1
                        if col == N-1:
                            dir = 2
                        elif row == 0:
                            dir = 0
                    else:
                        break
                    s += 1
                # 对保留前k个系数的patch做逆DCT
                f[i:i + N, j:j + N] = np.matmul(np.matmul(A.transpose(), patch_), A)
        f = f.round()
        figure, ax = plt.subplots(1, 2)
        basis = np.matmul(A.transpose().reshape((-1, 1), order='F'), A.reshape((1, -1)))
        basis_normalized = (basis - np.amin(basis)) / (np.amax(basis) - np.amin(basis))
        ax[0].imshow(basis_normalized, cmap=plt.cm.gray)
        ax[0].set_title("basis functions")
        ax[1].imshow(f, cmap=plt.cm.gray)
        ax[1].set_title("top-{} coefficients".format(top_k))
        plt.show()
        return f

def compute_psnr(img1, img2):
    # img1与img2的灰度值范围均为[0, 255]
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 10*np.log10(255.0*255.0/mse)

def test(file_path):
    img = Bitmap(file_path)
    F = DCT.forward(img.get_data())
    y_data = Converter.rgb2ycbcr(img.get_data())[:, :, 0]
    # DCT.backward(F, top_k=1)
    for k in [1, 2, 4, 6, 8, 10]:
        recon = DCT.backward(F, top_k=k)
        psnr = compute_psnr(recon, y_data[0:recon.shape[0], 0:recon.shape[1]])
        print("top-{}, psnr={}".format(k, psnr))

# test("../exp1/lena512color.bmp")