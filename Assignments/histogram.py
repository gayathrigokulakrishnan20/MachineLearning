"""Histogram example."""
import matplotlib.pyplot as plt


class Histogram:
    @staticmethod
    def plot() -> None:
        values = [12, 15, 15, 18, 19, 20, 21, 21, 22, 25, 28, 30, 32]
        plt.hist(values, bins=6, color="cornflowerblue", edgecolor="black")
        plt.title("Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
