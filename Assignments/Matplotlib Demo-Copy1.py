"""Run the Matplotlib examples defined in the separate chart modules."""

from area_plot import AreaPlot
from bar_plot import BarPlot
from box_plot import BoxPlot
from error_bar_plot import ErrorBarPlot
from heatmap import Heatmap
from histogram import Histogram
from line_plot import LinePlot
from pie_chart import PieChart
from scatter_plot import ScatterPlot
from stem_plot import StemPlot
from step_plot import StepPlot
from three_d_plot import ThreeDPlot


def main() -> None:
    for chart in (
        LinePlot, BarPlot, ScatterPlot, Histogram, PieChart, BoxPlot,
        AreaPlot, Heatmap, ErrorBarPlot, StemPlot, StepPlot, ThreeDPlot,
    ):
        chart.plot()


if __name__ == "__main__":
    main()
