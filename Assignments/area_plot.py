"""Stacked area plot example."""
import matplotlib.pyplot as plt


class AreaPlot:
    @staticmethod
    def plot() -> None:
        plt.stackplot([1, 2, 3, 4, 5], [2, 3, 4, 3, 5], [1, 2, 2, 3, 2], labels=["Series A", "Series B"], alpha=0.8)
        plt.title("Stacked Area Plot")
        plt.xlabel("Period")
        plt.ylabel("Value")
        plt.legend(loc="upper left")
        plt.show()
