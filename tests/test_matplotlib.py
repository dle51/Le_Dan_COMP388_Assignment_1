import utils.utils as utils
import matplotlib.pyplot as plt
import click

@click.command()
@click.option('--test', default=1, help='Which test plot to execute')

def plot_test(test):
    """Plots TEST:\n
        1. Empty Histogram\n
        2. Sample Histogram\n
        3. Empty AOC Graph\n
        4. Sample AOC Graph
    """
    match test:
        case 1:
            plot_empty_histogram()
        case 2:
            plot_sample_histogram()
        case 3:
            plot_empty_aoc_graph()
        case 4:
            plot_sample_aoc_graph()

def plot_empty_histogram():
    utils.plot_hist([])

def plot_sample_histogram():
    utils.plot_hist([(0, 0.2), (0, 0.3), (0, 0.4), (1, 0.5), (1, 0.6), (1, 0.7)])

def plot_empty_aoc_graph():
    utils.plot_sim_fmr_tmr_auc([])

def plot_sample_aoc_graph():
    utils.plot_sim_fmr_tmr_auc([(0, 0.2), (0, 0.3), (0, 0.4), (1, 0.5), (1, 0.6), (1, 0.7)])

if __name__ == "__main__":
    plot_test()
