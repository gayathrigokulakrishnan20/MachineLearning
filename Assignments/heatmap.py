"""Heatmap example."""
import matplotlib.pyplot as plt
import numpy as np


class Heatmap:
    @staticmethod
    def plot() -> None:
        image = plt.imshow(np.array([[1, 3, 2], [4, 6, 5], [7, 8, 9]]), cmap="viridis")
        plt.colorbar(image, label="Value")
        plt.title("Heatmap")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.show()
