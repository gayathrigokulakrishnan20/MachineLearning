"""Scatter plot example."""
import matplotlib.pyplot as plt


class ScatterPlot:
    @staticmethod
    def plot() -> None:
        plt.scatter([1, 2, 3, 4, 5, 6], [2, 5, 3, 7, 4, 8], color="teal", s=70)
        plt.title("Scatter Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()
