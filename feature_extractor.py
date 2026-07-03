"""
=============================================================
Feature Extractor
-------------------------------------------------------------
Real Photo vs Screen Recapture Detection

Part 1
    ✔ Imports
    ✔ FeatureExtractor Class
    ✔ Image Loading
    ✔ Image Preprocessing
    ✔ FFT Features
    ✔ Multi-scale FFT
    ✔ Brightness
    ✔ Contrast
    ✔ Sharpness

Author : Yashi
=============================================================
"""

import cv2
import numpy as np

from scipy.fft import fft2, fftshift
from scipy.stats import skew

from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops
)

from skimage.filters.rank import entropy
from skimage.morphology import disk


class FeatureExtractor:

    def __init__(self, image_size=256):

        self.image_size = image_size

        # LBP Parameters
        self.lbp_radius = 2
        self.lbp_points = 16

        # GLCM Parameters
        self.glcm_distances = [1, 2, 4]

        self.glcm_angles = [
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ]

    # ==========================================================
    # Load Image
    # ==========================================================

    def load_image(self, image):

        """
        Parameters
        ----------
        image : str or ndarray

        Returns
        -------
        RGB Image
        """

        if isinstance(image, str):

            img = cv2.imread(image)

            if img is None:

                raise ValueError(
                    f"Unable to load image : {image}"
                )

        else:

            img = image.copy()

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        return img

    # ==========================================================
    # Preprocessing
    # ==========================================================

    def preprocess(self, image):

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size
            )
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2HSV
        )

        return image, gray, hsv

    # ==========================================================
    # Radial FFT Profile
    # ==========================================================

    def radial_profile(self, magnitude):

        y, x = np.indices(magnitude.shape)

        center = np.array(magnitude.shape) // 2

        r = np.sqrt(
            (x - center[1]) ** 2 +
            (y - center[0]) ** 2
        )

        r = r.astype(np.int32)

        tbin = np.bincount(
            r.ravel(),
            magnitude.ravel()
        )

        nr = np.bincount(
            r.ravel()
        )

        return tbin / (nr + 1e-6)

    # ==========================================================
    # FFT Features
    # ==========================================================

    def fft_features(self, gray):

        fft = fft2(gray)

        fft = fftshift(fft)

        magnitude = np.log(
            np.abs(fft) + 1
        )

        h, w = magnitude.shape

        cy = h // 2
        cx = w // 2

        radius = 30

        low = magnitude[
            cy-radius:cy+radius,
            cx-radius:cx+radius
        ]

        high = magnitude.copy()

        high[
            cy-radius:cy+radius,
            cx-radius:cx+radius
        ] = 0

        low_energy = np.sum(low)

        high_energy = np.sum(high)

        ratio = high_energy / (
            low_energy + 1e-6
        )

        profile = self.radial_profile(
            magnitude
        )

        return [

            low_energy,

            high_energy,

            ratio,

            np.mean(profile),

            np.std(profile),

            np.max(profile),

            np.min(profile)

        ]

    # ==========================================================
    # Multi Scale FFT
    # ==========================================================

    def multiscale_fft(self, gray):

        feats = []

        for scale in [256, 128, 64]:

            resized = cv2.resize(
                gray,
                (
                    scale,
                    scale
                )
            )

            fft = fft2(resized)

            fft = fftshift(fft)

            magnitude = np.log(
                np.abs(fft) + 1
            )

            profile = self.radial_profile(
                magnitude
            )

            feats.extend([

                np.mean(profile),

                np.std(profile),

                np.max(profile),

                np.min(profile)

            ])

        return feats
            # ==========================================================
# Advanced FFT Features
# ==========================================================

# ==========================================================
# Advanced FFT Features
# ==========================================================

    def advanced_fft_features(self, gray):

        fft = fft2(gray)

        fft = fftshift(fft)

        magnitude = np.log(np.abs(fft) + 1)

        h, w = magnitude.shape

        cy = h // 2
        cx = w // 2

        Y, X = np.ogrid[:h, :w]

        radius = np.sqrt((X-cx)**2 + (Y-cy)**2)

        # -----------------------------
        # High Frequency Region
        # -----------------------------

        high_mask = radius > 80

        high = magnitude[high_mask]

        # -----------------------------
        # Peak Detection
        # -----------------------------

        threshold = np.mean(high) + 2*np.std(high)

        peaks = high > threshold

        peak_count = np.sum(peaks)

        peak_strength = np.sum(high[peaks])

        # -----------------------------
        # Spectral Entropy
        # -----------------------------

        norm = magnitude / (magnitude.sum() + 1e-10)

        spectral_entropy = -np.sum(
            norm * np.log(norm + 1e-10)
        )

        # -----------------------------
        # Radial Statistics
        # -----------------------------

        radial_std = np.std(high)

        radial_mean = np.mean(high)

        radial_max = np.max(high)

        radial_min = np.min(high)

        return [

            peak_count,

            peak_strength,

            spectral_entropy,

            radial_mean,

            radial_std,

            radial_max,

            radial_min

        ]
   
    
    
        
    # ==========================================================
    # Brightness Features
    # ==========================================================

    def brightness_features(self, gray):

        return [

            np.mean(gray),

            np.std(gray),

            np.percentile(gray, 5),

            np.percentile(gray, 95)

        ]

    # ==========================================================
    # Contrast Features
    # ==========================================================

    def contrast_features(self, gray):

        return [

            gray.std(),

            gray.var()

        ]

    # ==========================================================
    # Sharpness Features
    # ==========================================================

    def sharpness_features(self, gray):

        lap = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        return [

            lap.var(),

            lap.mean(),

            lap.std()

        ]
        # ==========================================================
    # LBP Features
    # ==========================================================

    def lbp_features(self, gray):
        """
        Standard Uniform Local Binary Pattern
        """

        lbp = local_binary_pattern(
            gray,
            P=self.lbp_points,
            R=self.lbp_radius,
            method="uniform"
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.lbp_points + 3),
            range=(0, self.lbp_points + 2)
        )

        hist = hist.astype(np.float32)

        hist /= (hist.sum() + 1e-6)

        return hist.tolist()

    # ==========================================================
    # GLCM Features
    # ==========================================================

    def glcm_features(self, gray):
        """
        Gray Level Co-occurrence Matrix Features
        """

        gray = gray.astype(np.uint8)

        glcm = graycomatrix(
            gray,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        features = []

        properties = [
            "contrast",
            "correlation",
            "energy",
            "homogeneity"
        ]

        for prop in properties:

            values = graycoprops(glcm, prop)

            features.extend([
                values.mean(),
                values.std(),
                values.max(),
                values.min()
            ])

        return features

    # ==========================================================
    # Canny Edge Features
    # ==========================================================

    def canny_features(self, gray):

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_density = np.count_nonzero(edges) / edges.size

        return [

            edge_density,

            np.mean(edges),

            np.std(edges)

        ]

    # ==========================================================
    # Sobel Gradient Features
    # ==========================================================

    def sobel_features(self, gray):

        gx = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        magnitude = np.sqrt(
            gx**2 + gy**2
        )

        return [

            gx.mean(),
            gx.std(),

            gy.mean(),
            gy.std(),

            magnitude.mean(),
            magnitude.std(),

            magnitude.max()

        ]

    # ==========================================================
    # Edge Orientation Histogram
    # ==========================================================

    def edge_orientation_features(self, gray):

        gx = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0
        )

        gy = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1
        )

        angle = np.arctan2(
            gy,
            gx
        )

        hist, _ = np.histogram(
            angle,
            bins=16,
            range=(-np.pi, np.pi)
        )

        hist = hist.astype(np.float32)

        hist /= (hist.sum() + 1e-6)

        return hist.tolist()

    # ==========================================================
    # Noise Features
    # ==========================================================

    def noise_features(self, gray):

        blur = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        noise = gray.astype(np.float32) - blur.astype(np.float32)

        return [

            np.mean(noise),

            np.std(noise),

            np.var(noise)

        ]
        # ==========================================================
    # HSV Statistics
    # ==========================================================

    def hsv_features(self, hsv):

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        features = []

        for channel in [h, s, v]:

            features.extend([

                np.mean(channel),

                np.std(channel),

                np.min(channel),

                np.max(channel)

            ])

        return features


    # ==========================================================
    # RGB Statistics
    # ==========================================================

    def rgb_statistics(self, image):

        features = []

        for i in range(3):

            channel = image[:, :, i]

            features.extend([

                np.mean(channel),

                np.std(channel)

            ])

        return features


    # ==========================================================
    # Color Moments
    # ==========================================================

    def color_moments(self, image):

        features = []

        for i in range(3):

            channel = image[:, :, i].astype(np.float32)

            features.extend([

                np.mean(channel),

                np.std(channel),

                skew(channel.flatten())

            ])

        return features


    # ==========================================================
    # Global Entropy
    # ==========================================================

    def entropy_features(self, gray):

        hist = cv2.calcHist(
            [gray],
            [0],
            None,
            [256],
            [0, 256]
        )

        hist = hist.flatten()

        hist /= (hist.sum() + 1e-6)

        entropy = -np.sum(

            hist * np.log2(hist + 1e-10)

        )

        return [entropy]


    # ==========================================================
    # Local Entropy
    # ==========================================================

    def local_entropy_features(self, gray):

        ent = entropy(
            gray,
            disk(5)
        )

        return [

            np.mean(ent),

            np.std(ent),

            np.max(ent),

            np.min(ent)

        ]


    # ==========================================================
    # Glare Features
    # ==========================================================

    def glare_features(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        _, mask = cv2.threshold(

            gray,

            240,

            255,

            cv2.THRESH_BINARY

        )

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )

        bright_pixels = np.count_nonzero(mask)

        glare_ratio = bright_pixels / mask.size

        largest_area = 0

        if len(contours):

            largest_area = max(

                cv2.contourArea(c)

                for c in contours

            )

        return [

            glare_ratio,

            len(contours),

            largest_area

        ]


    # ==========================================================
    # JPEG Blocking Features
    # ==========================================================

    def jpeg_features(self, gray):

        h, w = gray.shape

        vertical = []

        horizontal = []

        for x in range(8, w, 8):

            diff = np.abs(

                gray[:, x-1].astype(np.float32)

                -

                gray[:, x].astype(np.float32)

            )

            vertical.append(diff.mean())

        for y in range(8, h, 8):

            diff = np.abs(

                gray[y-1, :].astype(np.float32)

                -

                gray[y, :].astype(np.float32)

            )

            horizontal.append(diff.mean())

        return [

            np.mean(vertical),

            np.std(vertical),

            np.mean(horizontal),

            np.std(horizontal)

        ]


    # ==========================================================
    # Dynamic Range
    # ==========================================================

    def dynamic_range_features(self, gray):

        return [

            gray.max() - gray.min(),

            np.percentile(gray, 95),

            np.percentile(gray, 5)

        ]
        # ==========================================================
    # Main Feature Extraction Function
    # ==========================================================

    def extract(self, image):

        # Load image if path is provided
        image = self.load_image(image)

        # Preprocess
        image, gray, hsv = self.preprocess(image)

        features = []

        # Frequency Features
        features.extend(self.fft_features(gray))
        features.extend(self.multiscale_fft(gray))
        features.extend(
    self.advanced_fft_features(gray)
)

        # Image Statistics
        features.extend(self.brightness_features(gray))
        features.extend(self.contrast_features(gray))
        features.extend(self.sharpness_features(gray))

        # Texture Features
        features.extend(self.lbp_features(gray))
        features.extend(self.glcm_features(gray))

        # Edge Features
        features.extend(self.canny_features(gray))
        features.extend(self.sobel_features(gray))
        features.extend(self.edge_orientation_features(gray))

        # Noise
        features.extend(self.noise_features(gray))

        # Color Features
        features.extend(self.hsv_features(hsv))
        features.extend(self.rgb_statistics(image))
        features.extend(self.color_moments(image))

        # Entropy
        features.extend(self.entropy_features(gray))
        features.extend(self.local_entropy_features(gray))

        # Screen Specific Features
        features.extend(self.glare_features(image))
        features.extend(self.jpeg_features(gray))
        features.extend(self.dynamic_range_features(gray))

        return np.array(features, dtype=np.float32)
    def get_feature_names(self):

        names = []

        # ---------------- FFT ----------------
        names.extend([
            "fft_low_energy",
            "fft_high_energy",
            "fft_energy_ratio",
            "fft_profile_mean",
            "fft_profile_std",
            "fft_profile_max",
            "fft_profile_min"
        ])

        # ---------------- Multi-scale FFT ----------------
        
        names.extend([

    "fft_peak_count",

    "fft_peak_strength",

    "fft_spectral_entropy",

    "fft_radial_mean",

    "fft_radial_std",

    "fft_radial_max",

    "fft_radial_min"

])
        
        
        for scale in [256, 128, 64]:

            names.extend([
                f"fft_{scale}_profile_mean",
                f"fft_{scale}_profile_std",
                f"fft_{scale}_profile_max",
                f"fft_{scale}_profile_min"
            ])

        # ---------------- Brightness ----------------
        names.extend([
            "brightness_mean",
            "brightness_std",
            "brightness_p5",
            "brightness_p95"
        ])

        # ---------------- Contrast ----------------
        names.extend([
            "contrast_std",
            "contrast_var"
        ])

        # ---------------- Sharpness ----------------
        names.extend([
            "laplacian_variance",
            "laplacian_mean",
            "laplacian_std"
        ])

        # ---------------- LBP ----------------
        for i in range(self.lbp_points + 2):
            names.append(f"lbp_bin_{i}")

        # ---------------- GLCM ----------------
        properties = [
            "contrast",
            "correlation",
            "energy",
            "homogeneity"
        ]

        stats = [
            "mean",
            "std",
            "max",
            "min"
        ]

        for prop in properties:
            for stat in stats:
                names.append(f"glcm_{prop}_{stat}")

        # ---------------- Canny ----------------
        names.extend([
            "edge_density",
            "edge_mean",
            "edge_std"
        ])

        # ---------------- Sobel ----------------
        names.extend([
            "sobel_gx_mean",
            "sobel_gx_std",
            "sobel_gy_mean",
            "sobel_gy_std",
            "sobel_magnitude_mean",
            "sobel_magnitude_std",
            "sobel_magnitude_max"
        ])

        # ---------------- Edge Orientation ----------------
        for i in range(16):
            names.append(f"edge_orientation_bin_{i}")

        # ---------------- Noise ----------------
        names.extend([
            "noise_mean",
            "noise_std",
            "noise_variance"
        ])

        # ---------------- HSV ----------------
        for c in ["H", "S", "V"]:

            names.extend([
                f"{c}_mean",
                f"{c}_std",
                f"{c}_min",
                f"{c}_max"
            ])

        # ---------------- RGB Statistics ----------------
        for c in ["R", "G", "B"]:

            names.extend([
                f"{c}_mean",
                f"{c}_std"
            ])

        # ---------------- Color Moments ----------------
        for c in ["R", "G", "B"]:

            names.extend([
                f"{c}_moment_mean",
                f"{c}_moment_std",
                f"{c}_moment_skew"
            ])

        # ---------------- Entropy ----------------
        names.append("global_entropy")

        names.extend([
            "local_entropy_mean",
            "local_entropy_std",
            "local_entropy_max",
            "local_entropy_min"
        ])

        # ---------------- Glare ----------------
        names.extend([
            "glare_ratio",
            "glare_count",
            "largest_glare_area"
        ])

        # ---------------- JPEG ----------------
        names.extend([
            "jpeg_vertical_mean",
            "jpeg_vertical_std",
            "jpeg_horizontal_mean",
            "jpeg_horizontal_std"
        ])

        # ---------------- Dynamic Range ----------------
        names.extend([
            "dynamic_range",
            "dynamic_p95",
            "dynamic_p5"
        ])

       

        return names
        
        

# ==========================================================
# Test Feature Extractor
# ==========================================================

if __name__ == "__main__":

    extractor = FeatureExtractor()

    img_path = "dataset/real/1.jpeg"      

    features = extractor.extract(img_path)

    print("="*60)
    print("Feature Extraction Successful")
    print("="*60)

    print("Total Features :", len(features))
    print("Total Feature Names :", len(extractor.get_feature_names()))

    print("\nFirst 20 Features\n")

    print(features[:20])
       # ==========================================================
    # Feature Names
    # ==========================================================

    