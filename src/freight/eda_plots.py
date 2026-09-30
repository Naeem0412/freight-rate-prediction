"""Figures used in the report."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data import load_labeled, load_unlabeled


def main() -> None:
    C.REPORTS.mkdir(exist_ok=True)
    t, v = load_labeled(), load_unlabeled()
    t["rpm"] = t.posted_rate / t.distance

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(np.log10(t.rpm), bins=120, color="#064A56")
    ax[0].set_title("log10(rate per mile): ~1.4% corrupted labels\n(rate x/÷ 2-5)")
    ax[0].set_xlabel("log10(rate/mile)")
    ax[1].hist(t.weight.dropna(), bins=100, color="#064A56")
    ax[1].set_title("weight: negatives + pile-up at +/-47,500")
    ax[1].set_xlabel("weight (lb)")
    allf = pd.concat([t.drop(columns="posted_rate"), v])
    q = allf.groupby("date")["quote_signal"].mean()
    ax[2].plot(q.index, q.values, color="#064A56")
    ax[2].axvline(pd.Timestamp("2025-11-01"), color="r", ls="--", label="validation starts")
    ax[2].set_title("quote_signal daily mean: regime switches")
    ax[2].legend()
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.REPORTS / "eda.png", dpi=130)

    cv = pd.read_csv(C.OUT / "cv_folds.csv")
    cv = cv[(cv.scope == "clean_labels") & cv.config.isin(["final (features as chosen)", "with quote_signal", "with market_index"])]
    p = cv.pivot(index="fold", columns="config", values="MAPE")
    p.plot.bar(figsize=(8, 4), color=["#064A56", "#E08A1E", "#9DAFB3"])
    plt.ylabel("MAPE (clean labels)")
    plt.title("Rolling temporal folds: feature-set stability")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(C.REPORTS / "cv_folds.png", dpi=130)


if __name__ == "__main__":
    main()
