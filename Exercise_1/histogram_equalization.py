import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    # Load the image using CV2 and return it.
    loaded_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if loaded_image is None:
        raise FileNotFoundError(f"Cannot load image at {path}")
    return loaded_image


def compute_histogram(image: np.ndarray) -> np.ndarray:
    # 256-bin histogram without np.histogram
    hist = np.zeros(256, dtype=np.int64)
    flat = image.flatten()
    for px in flat:
        hist[int(px)] += 1
    return hist



def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    # Manual cumulative sum + normalization
    cdf = np.zeros_like(histogram, dtype=np.float64)
    total = float(np.sum(histogram))
    running = 0.0
    for i in range(256):
        running += float(histogram[i])
        cdf[i] = running / total
    return cdf


def equalize_image(image: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    # Apply: floor((C(v)-Cmin)/(1-Cmin)*255)
    flat = image.flatten()

    # Cmin = first non-zero cdf value
    cdf_min = cdf[cdf > 0][0]

    out = np.zeros_like(flat, dtype=np.uint8)
    for i, v in enumerate(flat):
        new_v = (cdf[int(v)] - cdf_min) / (1.0 - cdf_min) * 255.0
        out[i] = np.floor(new_v).astype(np.uint8)

    return out.reshape(image.shape)



def save_image(image: np.ndarray, path: str) -> None:
    # Save the image to the given folder.
    cv2.imwrite(path, image)



def show_images(original_image: np.ndarray, equalized_image: np.ndarray) -> None:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def histogram_equalization(input_path: str, output_path: str) -> None:
    # ToDo: Combine the different functions into one.
    loaded_image = load_image(input_path)
    histogram = compute_histogram(loaded_image)
    cdf = compute_cdf(histogram)
    equalized_image = equalize_image(loaded_image, cdf)
    if equalized_image.size != 0:
        save_image(equalized_image, output_path)


if __name__ == '__main__':
    # Load the images and perform histogram equalization.
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    input_image_path = str(data_dir / 'hello.png')
    output_image_path = str(data_dir / 'kitty.png')
    histogram_equalization(input_image_path, output_image_path)

    # Show the images next to each other.
    original = load_image(input_image_path)
    if Path(output_image_path).exists():
        equalized = load_image(output_image_path)
        show_images(original, equalized)
