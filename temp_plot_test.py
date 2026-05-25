from pathlib import Path
import numpy as np
from utils.plotting import plot_learning_curves

# create synthetic avg_curves with 4 algos, two very close to each other
N = 2000
k = np.arange(N)
# LMS nearly flat
lms = -18.5 + np.zeros(N)
# WL-LMS moderate improvement
wl = -18.8 - 10.0 * (1 - np.exp(-k/800.0))
# GH-WL-LMS fast improvement
gh = -18.9 - 25.0 * (1 - np.exp(-k/300.0))
# GH2D-WL-LMS almost identical to gh but tiny difference
gh2d = gh + 0.01 * np.sin(k/50.0)

avg_curves = {
    'LMS': lms,
    'WL-LMS': wl,
    'GH-WL-LMS': gh,
    'GH2D-WL-LMS': gh2d,
}

out = Path('d:/pyProject/Idea_1/results/temp_test_fig.png')
plot_learning_curves(results=avg_curves, title='Test IMD', save_path=out, smooth_window=10, y_lim=(-60, 10))
print('wrote', out)

