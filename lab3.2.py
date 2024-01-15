import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def make_abb(f, px, py, C40):
    rho = np.sqrt(px**2 + py**2)
    rho_normalized = rho / np.max(rho)
    R40 = 6 * rho_normalized**4 - 6 * rho_normalized**2 + 1
    f_abb = f * np.exp(2 * np.pi * 1j * C40 * R40)
    return f_abb

# Load the image
image = plt.imread('cat.jpeg')

# Determine the image dimensions
height, width, _ = image.shape

# Determine the minimum size of the image
minSize = min(height, width)

# Crop the image to a square size
croppedImage = image[:minSize-1, :minSize-1, :]

# Resize the image to [512, 512]
resizedImage = np.array(Image.fromarray(croppedImage).resize((512, 512)))
item = np.mean(resizedImage, axis=-1)  # Convert to grayscale

N = 512
A = 0.5
lambda_val = 0.5
D_zr = 20
step_zr = D_zr / N
step_it = 1 / (N * step_zr)
step_im = step_it * lambda_val / A
[im_axis_X, im_axis_Y] = np.meshgrid(
    np.arange(-(N/2) * step_im, (N/2) * step_im, step_im),
    np.arange(-(N/2) * step_im, (N/2) * step_im, step_im)
)

pupil = np.zeros((N, N))
x, y = np.meshgrid(
    np.arange(-(N/2) * step_zr, (N/2) * step_zr, step_zr),
    np.arange(-(N/2) * step_zr, (N/2) * step_zr, step_zr)
)
a = 1
for i in range(N-1):
    for j in range(N-1):
        if (x[i, j]**2)/(a**2) + y[i, j]**2 < 1:
            pupil[i, j] = 1


# Incoherent image
fft_intensity = 1 / N * np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.abs(item))))
fft_func_rasp = 0.25 * N * np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(pupil)))) ** 2)))

func_rasp_img = fft_intensity * fft_func_rasp
intensity_rasp_img = N * np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(func_rasp_img)))
intensity_rasp_img = np.abs(intensity_rasp_img) ** 2
intensity_rasp_img = intensity_rasp_img[::-1, :]

abb = make_abb(intensity_rasp_img, im_axis_X, im_axis_Y, 0.5)

# Plot results
#plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.pcolormesh(im_axis_X, im_axis_Y, intensity_rasp_img, cmap='gray')
plt.axis('equal')
plt.title("Incoherent Image")

plt.subplot(1, 2, 2)
plt.pcolormesh(im_axis_X, im_axis_Y, np.abs(abb), cmap='gray')
plt.axis('equal')
plt.title('Spherical aberration')
plt.show()
