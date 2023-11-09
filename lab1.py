import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from scipy.signal import triang
import numpy as np

def delta(n): #needs length N
    delta_function = [0 for i in range(N)]  #array of zeros
    delta_function[len(delta_function)//2] = 1
    delta_function = np.array(delta_function)
    return delta_function

def rect(x): #needs list x
    return np.where(abs(x)<=0.5, 1, 0)
def tr(n):
    return triang(n)

N = 1024
step = (1/N)**(1/2)
x_max = step * (N/2)
x = np.arange(-x_max, x_max, step)

delta_function = delta(N)
y_delta = fftshift(fft(fftshift(delta_function)))

rect = rect(x)
rect = fftshift(rect)
y_rect = fftshift(fft(rect))

tr = tr(N)
y_tr = fftshift(fft(tr))

plt.figure(1)
plt.subplot(211)
plt.plot(x, delta_function)
plt.subplot(212)
plt.plot(x,y_delta)

plt.figure(2)
plt.subplot(211)
plt.plot(x, rect)
plt.subplot(212)
plt.plot(x,y_rect)

plt.figure(3)
plt.subplot(211)
plt.plot(x, tr)
plt.subplot(212)
plt.plot(x,y_tr)

