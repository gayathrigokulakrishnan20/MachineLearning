"""Line plot examples."""
import matplotlib.pyplot as plt
import numpy as np


class LinePlot:
    @staticmethod
    def plot() -> None:
        x = np.arange(0, 10, 0.1)
        fig, ax = plt.subplots()
        ax.plot(x, 5 * x + 6.5)
        ax.set(title="Line Plot", xlabel="X", ylabel="Y")
        plt.show()

        x = np.arange(0, 10)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(x, 2 * x + 5, linewidth=4, linestyle=":", color="r", alpha=0.5, marker="2", label="line1")
        ax.set_facecolor("pink")
        ax.set(title="Customized Line Plot", xlabel="X-Axis", ylabel="Y-Axis")
        ax.legend()
        ax.grid(True)
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(x, 2 * x + 5)
        ax1.set_title("Graph1")
        ax2.plot(x, 3 * x + 10)
        ax2.set_title("Graph2")
        fig.tight_layout()
        plt.show()
