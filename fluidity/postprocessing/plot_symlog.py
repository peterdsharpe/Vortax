
import numpy as np
import matplotlib.pyplot as plt


def get_symlog_bins(
        min_val: float, max_val: float, n_bins: int = 50, linear_threshold: float = 1.0
    ) -> np.ndarray:
        # Handle negative values
        if min_val < -linear_threshold:
            neg_bins = -np.logspace(
                np.log10(max(abs(min_val), linear_threshold)), np.log10(linear_threshold), n_bins // 3
            )[::-1]
        else:
            neg_bins = []

        # Handle small values around zero (linear region)
        if min_val < linear_threshold and max_val > -linear_threshold:
            lin_bins = np.linspace(-linear_threshold, linear_threshold, n_bins // 3)
        else:
            lin_bins = []

        # Handle positive values
        if max_val > linear_threshold:
            pos_bins = np.logspace(
                np.log10(linear_threshold), np.log10(max(max_val, linear_threshold)), n_bins // 3
            )
        else:
            pos_bins = []

        # Combine all bins and remove duplicates
        all_bins = np.unique(np.concatenate([neg_bins, lin_bins, pos_bins]))
        return all_bins


def plot_symlog_distribution(data: np.ndarray, ax: plt.Axes=None, linear_threshold: float=1.0, add_reference_lines: bool = True, show: bool = True):
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    bins = get_symlog_bins(np.min(data), np.max(data), linear_threshold=linear_threshold)
    ax.hist(data, bins=bins, density=True, alpha=0.7)
    ax.set_ylabel("Frequency")
    ax.set_xscale("symlog", linthresh=linear_threshold)

    if add_reference_lines:
        mean_val = np.mean(data)
        median_val = np.median(data)
        ax.axvline(x=mean_val, color="r", linestyle="--", label=f"Mean: {mean_val:.3e}")
        ax.axvline(x=median_val, color="g", linestyle="--", label=f"Median: {median_val:.3e}")
        
        
        percentiles = [5, 95]
        for p in percentiles:
            ax.axvline(x=np.percentile(data, p), color="orange", linestyle=":", label=f"{p}% percentile: {np.percentile(data, p):.3e}")

        ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if show:
        plt.show()
    return ax
