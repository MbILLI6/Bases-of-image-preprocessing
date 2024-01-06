import matplotlib.pyplot as plt   #to run this program libraries must be installed
from scipy.fft import fft, ifft, fftshift
from scipy.signal.windows import triang
import numpy as np


def delta(n):  # needs length N
    delta_function = [0 for i in range(N)]  # array of zeros
    delta_function[len(delta_function) // 2] = 1
    delta_function = np.array(delta_function)
    return delta_function


def comb(x, n):   #needs list of x and length N
    f_comb = np.zeros(n)
    for i in range(n):
        if x[i] % 1 == 0:
            f_comb[i] = x[i]
    return f_comb


def rect(x):  # needs list x
    return np.where(abs(x) <= 0.5, 1, 0)


def tr(n):
    return triang(n)


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


N = 1024
step = (1 / N) ** (1 / 2)
x_max = step * (N / 2)
x = np.arange(-x_max, x_max, step)
x_tr = np.arange(-1, 1, 1 / 512)

delta_function = delta(N)
y_delta = fftshift(fft(fftshift(delta_function)))/(N ** 0.5)

comb = comb(x, N)
y_comb = fft(comb)/(N ** 0.5)

rect = rect(x)
rect = fftshift(rect)
y_rect = fftshift(fft(rect))/(N ** 0.5)

tr = tr(N)
y_tr = fftshift(fft(tr))/(N ** 0.5)

sin = sin(x)
y_sin = fftshift(fft(sin))/(N ** 0.5)

cos = cos(x)
y_cos = fftshift(fft(cos))/(N ** 0.5)

plt.figure(1)
plt.subplot(211)
plt.plot(x, delta_function)
plt.grid(True)
plt.subplot(212)
plt.plot(x, y_delta, label='Function')
plt.plot(x, np.real(y_delta), label='Real')
plt.plot(x, np.imag(y_delta), label='Imaginary')
plt.grid(True)
plt.legend()

plt.figure(2)
plt.subplot(211)
plt.plot(x, comb)
plt.grid(True)
plt.subplot(212)
plt.plot(x, y_comb, label='Function')
plt.plot(x, np.real(y_comb), label='Real')
plt.plot(x, np.imag(y_comb), label='Imaginary')
plt.grid(True)
plt.legend()


plt.figure(3)
plt.subplot(211)
plt.plot(x, rect)
plt.grid(True)
plt.subplot(212)
plt.plot(x, y_rect, label='Function')
plt.plot(x, np.real(y_rect), label='Real')
plt.plot(x, np.imag(y_rect), label='Imaginary')
plt.grid(True)
plt.legend()

plt.figure(4)
plt.subplot(211)
plt.plot(x_tr, tr)
plt.grid(True)
plt.subplot(212)
plt.plot(x_tr, y_tr, label='Function')
plt.plot(x, np.real(y_tr), label='Real')
plt.plot(x, np.imag(y_tr), label='Imaginary')
plt.grid(True)
plt.legend()

plt.figure(5)
plt.subplot(211)
plt.plot(x, sin)
plt.grid(True)
plt.subplot(212)
plt.plot(x, y_sin, label='Function')
plt.plot(x, np.real(y_sin), label='Real')
plt.plot(x, np.imag(y_sin), label='Imaginary')
plt.grid(True)
plt.legend()

plt.figure(6)
plt.subplot(211)
plt.plot(x, cos)
plt.grid(True)
plt.subplot(212)
plt.plot(x, y_cos, label='Function')
plt.plot(x, np.real(y_cos), label='Real')
plt.plot(x, np.imag(y_cos), label='Imaginary')
plt.grid(True)
plt.legend()
plt.show()


