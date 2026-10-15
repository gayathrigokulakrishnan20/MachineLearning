"""Box plot example."""
import matplotlib.pyplot as plt


class BoxPlot:
    @staticmethod
    def plot() -> None:
        groups = [[12, 14, 15, 15, 16, 18, 22], [10, 11, 13, 14, 16, 19, 25]]
        plt.boxplot(groups, labels=["Group A", "Group B"])
        plt.title("Box Plot")
        plt.ylabel("Value")
        plt.show()
