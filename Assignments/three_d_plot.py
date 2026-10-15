"""3D scatter plot example."""
import matplotlib.pyplot as plt


class ThreeDPlot:
    @staticmethod
    def plot() -> None:
        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")
        ax.scatter([1, 2, 3, 4, 5], [2, 4, 1, 5, 3], [3, 1, 4, 2, 5], color="purple", s=60)
        ax.set(title="3D Scatter Plot", xlabel="X", ylabel="Y", zlabel="Z")
        plt.show()
