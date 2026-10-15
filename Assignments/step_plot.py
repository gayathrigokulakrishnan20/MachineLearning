"""Step plot example."""
import matplotlib.pyplot as plt


class StepPlot:
    @staticmethod
    def plot() -> None:
        plt.step(range(6), [1, 3, 2, 5, 4, 6], where="mid")
        plt.title("Step Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()
