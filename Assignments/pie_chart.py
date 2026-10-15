"""Pie chart example."""
import matplotlib.pyplot as plt


class PieChart:
    @staticmethod
    def plot() -> None:
        plt.pie([20, 15, 30, 10], labels=["Apples", "Mangoes", "Lemon", "Oranges"], autopct="%1.1f%%", startangle=90)
        plt.title("Pie Chart")
        plt.axis("equal")
        plt.show()
