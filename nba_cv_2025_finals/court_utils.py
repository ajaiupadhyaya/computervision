# court_utils.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_halfcourt(ax, title="NBA Halfcourt"):
    # Court layout
    hoop = patches.Circle((25, 5.25), radius=0.75, linewidth=2, fill=False)
    backboard = patches.Rectangle((22, 4), 6, 0.1, linewidth=2, fill=False)
    paint = patches.Rectangle((17, 0), 16, 19, linewidth=2, fill=False)
    ft_circle = patches.Circle((25, 19), radius=6, linewidth=2, fill=False)
    three_arc = patches.Arc((25, 5.25), 47.5, 47.5, theta1=22, theta2=158, linewidth=2)

    for patch in [hoop, backboard, paint, ft_circle, three_arc]:
        ax.add_patch(patch)

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 47)
    ax.set_aspect(1)
    ax.axis('off')
    ax.set_title(title, fontsize=14)