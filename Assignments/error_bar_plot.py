"""Error bar plot example."""
import matplotlib.pyplot as plt
import numpy as np


class ErrorBarPlot:
    @staticmethod
    def plot() -> None:
        x = np.arange(1, 6)
        y = np.array([2.1, 2.8, 3.2, 4.1, 4.8])
        errors = np.array([0.3, 0.2, 0.4, 0.25, 0.35])
        plt.errorbar(x, y, yerr=errors, fmt="o-", capsize=5)
        plt.title("Error-Bar Plot")
        plt.xlabel("Measurement")
        plt.ylabel("Observed Value")
        plt.grid(True)
        plt.show()
