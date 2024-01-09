import numpy as np
import matplotlib.pyplot as plt


def func_delta2(N, step_1mm):
    delta = np.zeros((N, N))
    delta[N//2 + 1 + step_1mm, :] = 1
    return delta


def func_circ(x, y, N, r):
    f_circ = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if x[i, j]**2 + y[i, j]**2 <= r**2:
                f_circ[i, j] = 1
    return f_circ


def func_h(x, y, N, lambda_val, z, n, k):
    f_h = np.zeros((N, N), dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            f_h[j, i] = (1/(1j*lambda_val*z)) * np.exp(1j*k*n*z) * np.exp(1j*k*n*z) * np.exp(1j*k*n/(2*z)*(x[i, j]**2 + y[i, j]**2))
    return f_h


def func_rect(x, N):
    f_rect = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if -0.5 < x[i, j] < 0.5:
                f_rect[j, i] = 1
    return f_rect


def plot2(f_func, intensity, N, x, y, x_max, title1, title2):
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.pcolor(x, y, f_func, cmap='gray')
    plt.axis('equal')
    plt.axis([-x_max, x_max, -x_max, x_max])
    plt.title(title1)

    plt.subplot(232)
    plt.plot(x[N//2 + 1, :], f_func[N//2 + 1, :], y[:, N//2 + 1], f_func[:, N//2 + 1])
    plt.title("Two coordinate slise")

    plt.subplot(234)
    plt.pcolor(x, y, intensity, cmap='gray')
    plt.axis('equal')
    plt.axis([-x_max, x_max, -x_max, x_max])
    plt.title(title2)

    plt.subplot(235)
    plt.plot(x[N//2 + 1, :], intensity[N // 2 + 1, :], y[:, N // 2 + 1], intensity[:, N // 2 + 1])
    plt.title("Two coordinate slise")

    plt.show()


def func124(f_func, N, x, y, x_max, title1, title2):
    f_func_shift = np.fft.fftshift(f_func)
    f_func_fft = np.fft.fft2(f_func_shift)
    f_func_fft_shift = np.fft.fftshift(f_func_fft)

    f_func_fft_shift = f_func_fft_shift / N
    intensity = np.abs(f_func_fft_shift) ** 2
    plot2(f_func, intensity, N, x, y, x_max, title1, title2)


def func3(f_func1, f_func2, N, x, y, x_max, title1, title2):
    f_func1_shift = np.fft.fftshift(f_func1)
    f_func1_fft = np.fft.fft2(f_func1_shift)
    f_func2_shift = np.fft.fftshift(f_func2)
    f_func2_fft = np.fft.fft2(f_func2_shift)

    f_funcs_fft = f_func1_fft * f_func2_fft
    f_funcs_fft_shift_ifft = np.fft.ifft2(f_funcs_fft)
    f_funcs_fft_shift_ifft_shift = np.fft.fftshift(f_funcs_fft_shift_ifft)
    f_funcs_fft_shift_ifft_shift = f_funcs_fft_shift_ifft_shift / N
    intensity = np.abs(f_funcs_fft_shift_ifft_shift) ** 2
    plot2(f_func1, intensity, N, x, y, x_max, title1, title2)


def func_rect2(x, y, N):
    f_rect = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if -0.5 < x[i, j] < 0.5 and -0.5 < y[i, j] < 0.5:
                f_rect[j, i] = 1
            else:
                f_rect[j, i] = 0
    return f_rect


# Basic data
N = 512
step = np.sqrt(1/N)
x_max = step * N/2

# filling x, y
x, y = np.meshgrid(np.arange(-x_max, x_max, step), np.arange(-x_max, x_max, step))

# First part
f_func = func_rect2(x, 2 * y, N)
title1 = "Gap by Rect function"
title2 = "Intensity by Fraunhofer diffraction"
func124(f_func, N, x, y, x_max, title1, title2)  # for direct Fourier

# Second part
step_1mm = int(np.ceil(1 / step))   # number of counts for +/- 1 mm
f_func = func_delta2(N, step_1mm) + func_delta2(N, -step_1mm)
title1 = "Two narrow gaps"
title2 = "Intensity by Fraunhofer diffraction"
func124(f_func, N, x, y, x_max, title1, title2)

# Third part
r = 0.051
z = 1000
n = 1
lambda_val = 0.5e-6
x_max = 0.256

k = 2 * np.pi / lambda_val
step = x_max * 2 / N
x, y = np.meshgrid(np.arange(-x_max, x_max, step), np.arange(-x_max, x_max, step))
f_func = func_circ(x, y, N, r)
f_h = func_h(x, y, N, lambda_val, z, n, k)
num_zones_Fresnel = np.power(r, 2) / (lambda_val * z)  # number of Fresnel zones
title1 = f"Function circ with r = 0.051 м,\nNumber of Fresnel zones: {num_zones_Fresnel:.0f}"
title2 = "Fresnel diffraction on a circular screen"
func3(f_func, f_h, N, x, y, x_max, title1, title2)

# Fourth part
step = np.sqrt(1/N)
x_max = step * N/2
x, y = np.meshgrid(np.arange(-x_max, x_max, step), np.arange(-x_max, x_max, step))
for i in range(1, 6):
    b = i * 1e-01  
    f_func = func_rect((x + 1) / b, N) + func_rect((x - 1) / b, N)
    title1 = f"Two gaps with size of b={b:.1f}"
    title2 = "Intensity by Fraunhofer diffraction"
    func124(f_func, N, x, y, x_max, title1, title2)