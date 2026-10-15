"""Stem plot example."""
import matplotlib.pyplot as plt


class StemPlot:
    @staticmethod
    def plot() -> None:
        plt.stem(range(1, 8), [3, 5, 2, 6, 4, 7, 3])
        plt.title("Stem Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
