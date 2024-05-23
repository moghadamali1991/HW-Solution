import cv2
import numpy as np
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

class CircleDetector:
    def __init__(self, image_path, num_clusters=4):
        self.image_path = image_path
        self.num_clusters = num_clusters
        self.image = cv2.imread(image_path)
        self.segmented_image = None
        self.labels = None
        self.masks = []
        self.centers = None

    def preprocess_image(self):
        """Converts the image to a 2D array of pixels."""
        pixel_values = self.image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        return pixel_values

    def apply_kmeans(self, pixel_values):
        """Applies K-means clustering to segment the image."""
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixel_values, self.num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        self.labels = labels.reshape(self.image.shape[:2])
        self.centers = np.uint8(centers)
        self.segmented_image = self.centers[labels.flatten()].reshape(self.image.shape)

    def create_masks(self):
        """Creates masks for each color cluster."""
        for i in range(self.num_clusters):
            mask = np.zeros_like(self.image)
            mask[self.labels == i] = self.image[self.labels == i]
            self.masks.append(mask)

    def detect_and_draw_circles(self):
        """Detects circles in each mask and draws them on the original image."""
        for i, mask in enumerate(self.masks):
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                param1=50, param2=30, minRadius=10, maxRadius=50
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    cv2.circle(self.image, (x, y), r, (255, 0, 0), 1)
                    cv2.circle(self.image, (x, y), 1, (255, 0, 0), 1)
                    

    def display_result(self, output_path):
        """Write the overlay image into the path."""
        cv2.imwrite(output_path, self.image)
        

    def run(self,output_path):
        """Runs the complete process to detect and draw circles."""
        pixel_values = self.preprocess_image()
        self.apply_kmeans(pixel_values)
        self.create_masks()
        self.detect_and_draw_circles()
        self.display_result(output_path)

def process_images(directory, num_clusters=4):
    for i in range(10):
        image_path = os.path.join(directory, f'hw_image_{i}.png')
        output_path = os.path.join(directory, f'output_image_km{i}.png')
        detector = CircleDetector(image_path, num_clusters)
        detector.run(output_path)
        # detector.display_result(output_path)        


if __name__ == "__main__":
    image_directory = "../images/"
    process_images(image_directory, num_clusters=8)
