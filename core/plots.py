
import matplotlib.pyplot as plt


def plot_equity(equity, bh_equity):
    """Return a matplotlib figure comparing strategy equity versus buy & hold."""

    fig, ax = plt.subplots(figsize=(9, 4))
    equity.plot(ax=ax, label="Estrategia")
    bh_equity.plot(ax=ax, label="Buy & Hold")
    ax.legend()
    ax.set_title("Equity")
    return fig


def plot_drawdown(equity):
    """Return a matplotlib figure illustrating running drawdown."""

    fig, ax = plt.subplots(figsize=(9, 2.5))
    (equity / equity.cummax() - 1.0).plot(ax=ax, label="Drawdown")
    ax.legend()
    ax.set_title("Drawdown")
    return fig
