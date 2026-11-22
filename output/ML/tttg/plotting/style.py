"""Shared matplotlib style: CMS colour palette and axis styling."""

# named palette (used by the stacked NN-output plots)
CMS_NAMED = {
    "blue": "#3f90da",
    "orange": "#ffa90e",
    "red": "#bd1f01",
    "gray": "#94a4a2",
    "purple": "#832db6",
    "brown": "#a96b59",
    "dark_orange": "#e76300",
    "beige": "#b9ac70",
    "dark_gray": "#717581",
    "light_blue": "#92dadd",
    "cyan": "#17becf",
    "yellow-green": "#bcbd22",
}

# ordered palette (one colour per category in the limit plots)
CMS_COLORS = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59",
              "#e76300", "#b9ac70", "#717581", "#92dadd", "#17becf", "#bcbd22"]


def style_axes(ax):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=14, width=2, length=8)
    ax.tick_params(axis="both", which="minor", labelsize=12, width=1.5, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(True, alpha=0.3)
