import numpy as np
from scipy.stats import poisson
from skimage.util import random_noise
import scipy.io as sio
import matplotlib.pyplot as plt
import glymur
#glymur.config.set_library_search_paths(['D:/Program Files/openjpeg-v2.5.2-windows-x64/bin'])


def add_noise(img_clean, which_case):
    row, column, bands = img_clean.shape
    N = row * column  # img_clean has dimensions [row, column, band]
    img_noisy = None

    if which_case == 'case1':
        # --------------------- Case 1 --------------------------------------
        # Zero-mean Gaussian noise is added to all the bands of the Washington DC Mall
        # and Pavia city center data
        noise_type = 'additive'
        iid = 1
        sigma = 0.1
        np.random.seed(0)

        # Generate noisy image
        noise = sigma * np.random.randn(*img_clean.shape)
        img_noisy = img_clean + noise

    elif which_case == 'case2':
        # --------------------- Case 2 ---------------------
        # Different variance zero-mean Gaussian noise is added to each band
        # The std values are randomly selected from 0 to 0.1.
        noise_type = 'additive'
        iid = 0
        np.random.seed(0)

        sigma = np.random.rand(bands) * 0.1
        noise = np.random.randn(*img_clean.shape)

        for cb in range(bands):
            noise[:, :, cb] = sigma[cb] * noise[:, :, cb]

        img_noisy = img_clean + noise

    elif which_case == 'case3':
        #  ---------------------  Case 3: Poisson Noise ---------------------
        noise_type = 'poisson'
        iid = np.nan  # noise_type is set to 'poisson'
        img_wN = np.copy(img_clean)

        snr_db = 15
        snr_set = np.exp(snr_db * np.log(10) / 10)

        img_wN_scale = np.zeros((bands, N))
        img_wN_noisy = np.zeros((bands, N))

        for i in range(bands):
            img_wNtmp = img_wN[:, :, i].reshape(1, N)
            img_wNtmp = np.maximum(img_wNtmp, 0)
            factor = snr_set / (np.sum(img_wNtmp ** 2) / np.sum(img_wNtmp))
            img_wN_scale[i, :] = factor * img_wNtmp
            # Generates random samples from a Poisson distribution
            img_wN_noisy[i, :] = poisson.rvs(factor * img_wNtmp)

        img_noisy = img_wN_noisy.T.reshape((row, column, bands))
        img_clean = img_wN_scale.T.reshape((row, column, bands))

    elif which_case == 'case4':
        # --------------------- Case 4: Salt & Pepper Noise ---------------------
        noise_type = 'salt & pepper'
        amount = 0.05  # the amount of noise

        img_noisy = np.zeros_like(img_clean)
        for cb in range(bands):
            img_noisy[:, :, cb] = random_noise(img_clean[:, :, cb], mode='s&p', amount=amount)

    elif which_case == 'case5':
        # --------------------- Case 5: Stripes Noise ---------------------
        noise_type = 'stripes'
        img_noisy = np.copy(img_clean)

        # Define stripe noise parameters
        stripenum = np.random.randint(6, 16)  # Randomly generate 6-15 stripes
        for cb in range(bands):
            locolumn = np.random.choice(column, size=stripenum, replace=False)  # Random stripe positions
            img_noisy[:, locolumn, cb] = 0.2 * np.random.rand(1) + 0.6  # Add stripes noise

    elif which_case == 'case6':
        # --------------------- Case 6: Deadlines Noise ---------------------
        noise_type = 'deadlines'
        img_noisy = np.copy(img_clean)

        # Define deadline noise parameters
        deadlinenum = np.random.randint(6, 11)  # Randomly generate 6-10 deadlines
        for cb in range(bands):
            locolumn = np.random.choice(column - 2, size=deadlinenum, replace=False)  # Deadline positions
            an = np.random.randint(1, 4, size=deadlinenum)  # Randomly select width (1-3)
            for idx in range(deadlinenum):
                if an[idx] == 1:
                    img_noisy[:, locolumn[idx], cb] = 0
                elif an[idx] == 2:
                    img_noisy[:, locolumn[idx]:locolumn[idx] + 2, cb] = 0
                else:
                    img_noisy[:, locolumn[idx]:locolumn[idx] + 3, cb] = 0

    elif which_case == 'case7':
        compression_ratios = [50]
        output_file = 'output_lossy.jp2'
        jp2k = glymur.Jp2k(output_file, img_clean, cratios=compression_ratios)
        img_noisy = jp2k[:]

    elif which_case == 'case8':
        # Define the indices of the visible and infrared bands
        visible_band_indices = list(range(10, 30))  # Assuming these are the visible bands
        infrared_band_indices = list(range(80, 103))  # Assuming these are the infrared bands 

        # Step 1: Segment the hyperspectral image into visible and infrared groups
        I1 = img_clean[:, :, visible_band_indices]  # Visible bands
        I2 = img_clean[:, :, infrared_band_indices]  # Infrared bands

        # Step 2: Compute the average of visible bands
        I1_avg = np.mean(I1, axis=2)  # Average across the band dimension, resulting in (90, 90)

        # Step 3: Compute the average of infrared bands
        I2_avg = np.mean(I2, axis=2)  # Average across the band dimension, resulting in (90, 90)

        # Step 4: Calculate the fog intensity map F
        F = I1_avg - I2_avg

        # Select the pixel coordinates for the fog corrupted object and the clean object
        x1, y1 = 10, 20  # Coordinates for the fog corrupted pixel
        x2, y2 = 30, 40  # Coordinates for the clean pixel

        # Step 5: Extract values from the hyperspectral image and the fog intensity map
        I_i = img_clean[x1, y1, :]  # Spectral reflectance for the fog corrupted pixel
        I_j = img_clean[x2, y2, :]  # Spectral reflectance for the clean pixel
        F_i = F[x1, y1]  # Fog intensity for the fog corrupted pixel
        F_j = F[x2, y2]  # Fog intensity for the clean pixel

        # Step 6: Calculate the fog abundance A
        # Ensure to handle cases where F_i and F_j are equal to avoid division by zero
        A = np.zeros_like(I_i)  # Initialize A with the same shape as the spectral bands

        # Calculate A only for channels where F_i and F_j are not equal
        mask = F_i != F_j
        A[mask] = (I_i[mask] - I_j[mask]) / (F_i - F_j)
        img_noisy = img_clean + A * F

    return img_noisy, img_clean


def display_hyperspectral_images(img_clean, img_noisy, bands=(30, 20, 10)):
    # Normalize the images for display
    def normalize(image):
        image_min = image.min()
        image_max = image.max()
        return (image - image_min) / (image_max - image_min)

    # Select the bands for RGB representation (R, G, B)
    img_clean_rgb = normalize(img_clean[:, :, bands])
    img_noisy_rgb = normalize(img_noisy[:, :, bands])

    # Plot the clean and noisy images
    plt.figure(figsize=(12, 6))

    # Clean image
    plt.subplot(1, 2, 1)
    plt.imshow(img_clean_rgb)
    plt.title("Clean Hyperspectral Image (False Color)")

    # Noisy image
    plt.subplot(1, 2, 2)
    plt.imshow(img_noisy_rgb)
    plt.title("Noisy Hyperspectral Image (False Color)")

    plt.show()


which_case = 'case8'
img_clean = sio.loadmat("./data/PaviaU.mat")
img_clean = img_clean['paviaU']
img_noisy, _ = add_noise(img_clean, which_case)
display_hyperspectral_images(img_clean, img_noisy, bands=(30, 20, 10))



