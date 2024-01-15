import numpy as np
import matplotlib.pyplot as plt


def make_abb(f, px, py, C40):
    rho = np.sqrt(px**2 + py**2)
    rho_normalized = rho / np.max(rho)
    R40 = 6 * rho_normalized**4 - 6 * rho_normalized**2 + 1
    f_abb = f * np.exp(2 * np.pi * 1j * C40 * R40)
    return f_abb

# Function to create a circular aperture
def func_Circ(x, y, N, r):
    f_circ = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if x[j, i]**2 + y[j, i]**2 <= r**2:
                f_circ[j, i] = 1
    return f_circ


N = 512
lmbda = 0.5
Dzr = 4
A = 0.5
R_zr = N / Dzr
dp = Dzr / N
dn = 1 / (N * dp)
dx = dn * lmbda / A

n_max = dn * N / 2
p_max = dp * N / 2
x_max = dx * N / 2

nx, ny = np.meshgrid(np.arange(-n_max, n_max, dn), np.arange(-n_max, n_max, dn))
px, py = np.meshgrid(np.arange(-p_max, p_max, dp), np.arange(-p_max, p_max, dp))
x, y = np.meshgrid(np.arange(-x_max, x_max, dx), np.arange(-x_max, x_max, dx))

for C40 in [0, 0.39, 0.67, 0.535, 0.64]:
    zr = func_Circ(px, py, N, 1)
    zr_abb = make_abb(zr, px, py, C40)

    psf_ = (dp / dn) * np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(zr))) * N
    psf_abs = (np.abs(psf_) * np.abs(psf_)) / (np.pi**2)
    D = (dn / dp) * np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf_abs))) / N
    D_norm = D * np.pi
    M_abs = np.abs(D_norm)

    psf_abb = (dp / dn) * np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(zr_abb))) * N
    psf_abb_abs = (np.abs(psf_abb) * np.abs(psf_abb)) / (np.pi**2)
    D_abb = (dn / dp) * np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf_abb_abs))) / N
    D_abb_norm = D_abb * np.pi
    M_abb_abs = np.abs(D_abb_norm)
    st = np.abs(np.max(psf_abb) / np.max(psf_))

    psf = (np.abs(psf_) * np.abs(psf_)) / (np.pi**2)
    psf_abb = (np.abs(psf_abb) * np.abs(psf_abb)) / (np.pi**2)

    # Plot PSF
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.cla()
    plt.plot(x[N // 2, :], psf[N // 2, :], 'b')
    plt.plot(x[N // 2, :], psf_abb[N // 2, :], 'r')
    plt.xlim([-x_max / 4, x_max / 4])
    plt.grid(True)
    plt.xlabel("x', µm")
    plt.title('PSF cross-section')
    text = f'c40 = {C40:.2f}'
    plt.legend(['Without aberrations', text])

    plt.subplot(1, 2, 2)
    plt.pcolormesh(x, y, psf_abb, cmap='gray')
    plt.axis('equal')
    plt.axis([-x_max / 4, x_max / 4, -x_max / 4, x_max / 4])
    plt.xlabel("x', µm")
    plt.ylabel("y', µm")
    text = f'PSF, c40 = {C40:.2f}'
    plt.title(text)

    # Plot MTF
    plt.figure(2)
    plt.subplot(1, 1, 1)
    plt.cla()
    plt.plot(px[N // 2, :], M_abb_abs[N // 2, :])
    plt.xlim([0, 2])
    plt.grid(True)
    plt.xlabel('sx, sy, cycles/µm')
    text = f'MTF, c40 = {C40:.2f}'
    plt.title(text)
    plt.pause(0.1)

plt.show()




