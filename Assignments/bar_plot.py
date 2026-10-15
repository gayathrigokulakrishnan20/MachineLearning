"""Vertical and horizontal bar plot examples."""
import matplotlib.pyplot as plt


class BarPlot:
    @staticmethod
    def plot() -> None:
        names = ["apples", "Mangoes", "Lemon", "Oranges"]
        values = [20, 15, 30, 10]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.bar(names, values)
        ax1.set_title("Vertical Bar Plot")
        ax2.barh(names, values, color=["orange", "blue", "green", "yellow"])
        ax2.set(title="Horizontal Bar Plot", xlabel="Quantity", ylabel="Fruit")
        fig.tight_layout()
        plt.show()
