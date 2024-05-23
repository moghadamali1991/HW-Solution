import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
import warnings
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

warnings.filterwarnings('ignore')


class WaveletDenoiser:
    def __init__(self, image_path, mode='haar', level=1):
        self.image_path = image_path
        self.mode = mode
        self.level = level
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.denoised_image = None
        self.thresholded_image = None

    def denoise(self):
        """Perform wavelet denoising and thresholding."""
        im_array = np.float32(self.gray_image) / 255
        coeffs = pywt.wavedec2(im_array, self.mode, level=self.level)
        coeffs_H = list(coeffs)
        coeffs_H[0] *= 0
        im_array_H = pywt.waverec2(coeffs_H, self.mode)
        im_array_H = np.uint8(im_array_H * 255)
        _, self.thresholded_image = cv2.threshold(im_array_H, 5, 255, cv2.THRESH_BINARY)
        return self.thresholded_image


class BlobDetector:
    def __init__(self):
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.minThreshold = 0
        self.params.maxThreshold = 255
        self.params.filterByArea = True
        self.params.minArea = 200
        self.params.maxArea = 20000
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.5
        self.params.filterByConvexity = False
        self.params.minConvexity = 0.5
        self.params.filterByInertia = False
        self.params.minInertiaRatio = 0.01
        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def detect(self, image):
        """Detect blobs in the given image."""
        keypoints = self.detector.detect(image)
        return keypoints

    @staticmethod
    def draw_blobs(image, keypoints):
        """Draw detected blobs on the image."""
        for kp in keypoints:
            x, y = np.int0(kp.pt)
            cv2.circle(image, (x, y), np.int0(kp.size / 2), (255, 0, 0), 1)
            cv2.circle(image, (x, y), 1, (255, 0, 0), 1)
        img_with_blobs = cv2.drawKeypoints(
            image, keypoints, np.array([]), (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        return img_with_blobs


def main():
    image_directory = "../images/"

    for i in range(10):
        image_path = os.path.join(image_directory, f'hw_image_{i}.png')
        output_path = os.path.join(image_directory, f'output_image_wl{i}.png')

        # Denoise the image using wavelet transform
        denoiser = WaveletDenoiser(image_path, mode='db1', level=9)
        thresholded_image = denoiser.denoise()
        thresholded_image = cv2.bitwise_not(thresholded_image)

        # Detect blobs in the denoised image
        blob_detector = BlobDetector()
        keypoints = blob_detector.detect(thresholded_image)
        print(f"Number of detected circles in img{i}.jpg:", len(keypoints))

        # Draw detected blobs on the original image
        original_image = cv2.imread(image_path)
        image_with_blobs = blob_detector.draw_blobs(original_image, keypoints)

        # Save the result
        cv2.imwrite(output_path, image_with_blobs)

        # Optionally display the result
        plt.imshow(cv2.cvtColor(image_with_blobs, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Blobs in img{i}.jpg")
        plt.show()


if __name__ == "__main__":
    main()
