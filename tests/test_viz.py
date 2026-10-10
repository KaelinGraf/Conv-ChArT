"""Per-panel figure export.  PYTHONPATH= python -m pytest tests/test_viz.py -q
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcc.viz import save_panels


def test_save_panels_one_file_per_data_panel_and_restores_visibility(tmp_path):
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].plot([0, 1], [0, 1], label="a"); ax[0].set_title("Alpha: one!")
    ax[0].set_ylabel("a y-label much longer than the 3-inch panel is tall, which must not be clipped")
    ax[1].bar([0, 1], [1, 2], label="b"); ax[1].set_title("Beta")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax[2].axis("off")
    fig.suptitle("shared banner"); fig.tight_layout()
    out = tmp_path / "fig.png"; fig.savefig(out, dpi=150)
    paths = save_panels(fig, out)
    assert [p.name for p in paths] == ["fig_p1_alpha_one.png", "fig_p2_beta.png"]
    w = plt.imread(out).shape[1]
    assert all(plt.imread(p).shape[1] < w / 2 for p in paths)
    assert all(o.get_visible() for o in [*fig.axes, *fig.texts])
    assert plt.imread(paths[0]).shape[0] > plt.imread(out).shape[0]
    plt.close(fig)
