"""Map outputs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_overview_maps(path, allowed, suit, energy_cell, cand):
    """Three panels: suitability, energy per cell, and candidate areas."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    im = ax[0].imshow(np.where(allowed, suit, np.nan), cmap="YlGn", vmin=0, vmax=1)
    ax[0].set_title("Suitability (excluded = blank)")
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    im = ax[1].imshow(np.where(allowed, energy_cell / 1e3, np.nan), cmap="plasma")
    ax[1].set_title("Annual PV energy per cell (MWh)")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(np.where(allowed, suit, np.nan), cmap="Greys", alpha=0.35)
    cm = ax[2].imshow(np.where(cand > 0, energy_cell / 1e3, np.nan), cmap="plasma")
    ax[2].set_title("Decision map: candidate areas")
    plt.colorbar(cm, ax=ax[2], fraction=0.046, label="MWh/cell/yr")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def save_ranked_map(path, allowed, suit, energy_cell, cand, table, top=4):
    """Single panel: candidate areas coloured by energy, top areas numbered by rank."""
    fig, a = plt.subplots(figsize=(5.4, 4.6))
    a.imshow(np.where(allowed, suit, np.nan), cmap="Greys", alpha=0.30, vmin=0, vmax=1)
    a.imshow(np.where(~allowed, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3, alpha=0.55)   # excluded cells
    cm = a.imshow(np.where(cand > 0, energy_cell / 1e3, np.nan), cmap="YlOrRd")
    for rank, row in table.head(top).iterrows():
        a.text(row.centroid_col, row.centroid_row, str(rank + 1), ha="center", va="center", fontsize=11,
               fontweight="bold", color="white", bbox=dict(boxstyle="circle,pad=0.2", fc="#1F3864", ec="white"))
    plt.colorbar(cm, ax=a, fraction=0.046, label="Annual PV energy (MWh per cell)")
    a.set_title("Candidate areas ranked by annual energy", fontsize=11)
    a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)
