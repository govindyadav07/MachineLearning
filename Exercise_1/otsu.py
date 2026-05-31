import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    loaded_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if loaded_image is None:
        raise FileNotFoundError(f"Cannot load image at {path}")
    return loaded_image


def compute_histogram(image: np.ndarray) -> np.ndarray:
    """Compute a grayscale histogram with 256 bins."""
    histogram = np.zeros(256, dtype=np.int64)
    flat = image.flatten()
    for px in flat:
        histogram[int(px)] += 1
    return histogram


def p_helper(prob: np.ndarray, theta: int) -> tuple[float, float]:
    """Compute class probabilities p0 and p1 for threshold theta."""
    n = len(prob)
    theta = min(theta, n - 1)

    p0 = 0.0
    for i in range(theta + 1):
        p0 += float(prob[i])

    p1 = 1.0 - p0
    return p0, p1


def mu_helper(prob: np.ndarray, theta: int, p0: float, p1: float) -> tuple[float, float]:
    """Compute class means mu0 and mu1 for threshold theta."""
    n = len(prob)
    theta = min(theta, n - 1)

    mu0 = 0.0
    mu1 = 0.0

    if p0 > 0.0:
        s0 = 0.0
        for i in range(theta + 1):
            s0 += i * float(prob[i])
        mu0 = s0 / p0

    if p1 > 0.0:
        s1 = 0.0
        for i in range(theta + 1, n):
            s1 += i * float(prob[i])
        mu1 = s1 / p1

    return mu0, mu1


def otsu_threshold(histogram: np.ndarray) -> int:
    """Compute Otsu's threshold from a histogram."""
    total = int(np.sum(histogram))
    if total == 0:
        return 0

    prob = histogram.astype(np.float64) / total
    n = len(prob)

    max_variance = -1.0
    best_threshold = 0

    for theta in range(n):
        p0, p1 = p_helper(prob, theta)

        if p0 <= 0.0 or p1 <= 0.0:
            continue

        mu0, mu1 = mu_helper(prob, theta, p0, p1)
        variance = p0 * p1 * (mu1 - mu0) ** 2

        if variance > max_variance:
            max_variance = variance
            best_threshold = theta

    return int(best_threshold)


def otsu_binarize(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Binarize an image using Otsu's threshold."""
    histogram = compute_histogram(image)
    theta = otsu_threshold(histogram)
    binarized = np.where(image > theta, 255, 0).astype(np.uint8)
    return binarized, theta


def custom_binarization(image: np.ndarray, theta: int) -> tuple[np.ndarray, int]:
    new_image = np.where(image > theta, 255, 0).astype(np.uint8)
    return new_image, theta


# Compatibility aliases for unit tests
create_greyscale_histogram = compute_histogram
calculate_otsu_threshold = otsu_threshold


def otsu(image: np.ndarray) -> np.ndarray:
    """Return binarized image using Otsu's method (wrapper for tests)."""
    bin_img, _ = otsu_binarize(image)
    return bin_img


def binarize_threshold(image: np.ndarray, theta: int) -> np.ndarray:
    """Simple thresholding utility used in tests."""
    return np.where(image > theta, 255, 0).astype(np.uint8)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    loaded_image = load_image(str(base_dir / "data" / "runes.png"))

    binarized_image, threshold = otsu_binarize(loaded_image)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(loaded_image, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(binarized_image, cmap="gray")
    plt.title(f"Otsu Binarization (t={threshold})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()