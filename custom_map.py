import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from scipy.stats import t


def cmap_pearson(features, n, alpha, name="g-bajon"):
    K = (features * (features - 1)) / 2
    alpha_eff = alpha / K  # Bonferroni
    df = n - 2

    t_crit = t.ppf(1 - alpha_eff / 2, df)
    r_crit = t_crit / np.sqrt(n + (t_crit ** 2) - 2)  # hypothesis H(0) for the Pearson test

    dt = 1 / 128
    steps = int(np.ceil(r_crit / dt)*2)

    if name=="h-bajon":
        cmap = mpl.colormaps["RdBu_r"]
        cmap_r = mpl.colormaps["RdBu_r"]
    if name=="g-bajon":
        cmap = mpl.colormaps["Greys"]
        cmap_r = mpl.colormaps["Greys_r"]
    filter = mpl.colormaps["Greys"]

    rgba1 = cmap_r(np.linspace(0, 0.7, 128 - steps//2))
    rgba3 = cmap(np.linspace(0.3, 1, 128 - steps//2))
    rgba2 = filter(np.linspace(0, 0, steps))
    rgba = np.concatenate([rgba1, rgba2, rgba3], axis=0)[:256]

    hex256 = [mcolors.to_hex(c) for c in rgba]

    my_cmap = mcolors.ListedColormap(hex256, name=name)
    mpl.colormaps.register(my_cmap, force=True)

    return {
        "name": name,
        "alpha_eff": alpha_eff,
        "df": df,
        "t_crit": float(t_crit),
        "r_crit": float(r_crit),
        "steps": steps,
        "side_left": 128 - steps//2,
        "side_right": 128 - steps//2,
        "N": n,
    }