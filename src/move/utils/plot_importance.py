#!/usr/bin/env python

import warnings
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as pl
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

red_blue = LinearSegmentedColormap('red_blue', { # #1E88E5 -> #ff0052
    'red': ((0.0, 30./255, 30./255),
            (1.0, 255./255, 255./255)),

    'green': ((0.0, 136./255, 136./255),
              (1.0, 13./255, 13./255)),

    'blue': ((0.0, 229./255, 229./255),
             (1.0, 87./255, 87./255)),

    'alpha': ((0.0, 1, 1),
              (0.5, 0.3, 0.3),
              (1.0, 1, 1))
})

red_blue_solid = LinearSegmentedColormap('red_blue_solid', {
    'red': ((0.0, 30./255, 30./255),
            (1.0, 255./255, 255./255)),

    'green': ((0.0, 136./255, 136./255),
              (1.0, 13./255, 13./255)),

    'blue': ((0.0, 229./255, 229./255),
             (1.0, 87./255, 87./255)),

    'alpha': ((0.0, 1, 1),
              (0.5, 1, 1),
              (1.0, 1, 1))
})

# colors = []
# for l in np.linspace(1, 0, 100):
#     colors.append((30./255, 136./255, 229./255,l))
# for l in np.linspace(0, 1, 100):
#     colors.append((255./255, 13./255, 87./255,l))
# red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

default_colors = ["#1E88E5", "#ff0d57", "#13B755", "#7C52FF", "#FFC000", "#00AEEF"]

#blue_rgba = np.array([0.11764705882352941, 0.5333333333333333, 0.8980392156862745, 1.0])
blue_rgba = np.array([30, 136, 229, 255]) / 255
blue_rgb = np.array([30, 136, 229]) / 255
red_rgb = np.array([255, 13, 87]) / 255

default_blue_colors = []
tmp = blue_rgba.copy()
for i in range(10):
    default_blue_colors.append(tmp.copy())
    if tmp[-1] > 0.1:
        tmp[-1] *= 0.7

def summary_plot(shap_values, features=None, feature_names=None, max_display=None, plot_type="dot",
                 color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                 color_bar=True, auto_size_plot=True, layered_violin_max_num_bins=50, class_names=None, size = 10):
    """
    Create a SHAP summary plot, colored by feature values when they are provided.

    Args:
        shap_values (np.array): Matrix of SHAP values (# samples x # features)
        features (np.array or pd.DataFrame or list, optional): Matrix of feature values (# samples x # features) or a feature_names list as shorthand. Defaults to None.
        feature_names (list, optional): Names of the features (length # features). Defaults to None.
        max_display (int, optional): How many top features to include in the plot (default is 20, or 7 for interaction plots). Defaults to None.
        plot_type (str, optional): What type of summary plot to produce. Defaults to "dot".   
        color (str, optional): color for a plot. Defaults to None.
        axis_color (str, optional): color for axis. Defaults to "#333333".
        title (str, optional): #Not used in the function. Defaults to None. 
        alpha (float, optional): The alpha blending value, between 0 (transparent) and 1 (opaque). Defaults to 1.
        show (bool, optional): whether to show plot. Defaults to True.
        sort (bool, optional): #Not used in the function. Defaults to True.
        color_bar (bool, optional): Draw color bar. Defaults to True.
        auto_size_plot (bool, optional): autosize plot. Defaults to True.
        layered_violin_max_num_bins (int, optional): maximum number of bins for layered violin. Defaults to 50.
        class_names (list, optional): features names. Defaults to None.
        size (int, optional): size for the plot (if auto_size_plot=True). Defaults to 10.
    """    
    
    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        plot_type = "bar" # only type supported for now
    else:
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."
    
    # default color:
    if color is None:
        color = "coolwarm" if plot_type == 'layered_violin' else "#1E88E5" #"#ff0052"
    
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None
    
    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])
    
    if feature_names is None:
        feature_names = np.array([Feature %s% str(i) for i in range(num_features)])
    
    # plotting SHAP interaction values
    if not multi_class and len(shap_values.shape) == 3:
        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)
        
        sort_inds = np.argsort(-np.abs(shap_values.sum(1)).sum(0))
        
        # get plotting limits
        delta = 1.0 / (shap_values.shape[1] ** 2)
        slow = np.nanpercentile(shap_values, delta)
        shigh = np.nanpercentile(shap_values, 100 - delta)
        v = max(abs(slow), abs(shigh))
        slow = -v
        shigh = v
        
        pl.figure(figsize=(1.5 * max_display + 1, 1 * max_display + 1))
        pl.subplot(1, max_display, 1)
        proj_shap_values = shap_values[:, sort_inds[0], sort_inds]
        proj_shap_values[:, 1:] *= 2  # because off diag effects are split in half
        summary_plot(
            proj_shap_values, features[:, sort_inds] if features is not None else None,
            feature_names=feature_names[sort_inds],
            sort=False, show=False, color_bar=False,
            auto_size_plot=False,
            max_display=max_display
        )
        
        pl.xlim((slow, shigh))
        pl.xlabel("")
        title_length_limit = 11
        pl.title(shorten_text(feature_names[sort_inds[0]], title_length_limit))
        for i in range(1, min(len(sort_inds), max_display)):
            ind = sort_inds[i]
            pl.subplot(1, max_display, i + 1)
            proj_shap_values = shap_values[:, ind, sort_inds]
            proj_shap_values *= 2
            proj_shap_values[:, i] /= 2  # because only off diag effects are split in half
            summary_plot(
                proj_shap_values, features[:, sort_inds] if features is not None else None,
                sort=False,
                feature_names=["" for i in range(len(feature_names))],
                show=False,
                color_bar=False,
                auto_size_plot=False,
                max_display=max_display
            )
            pl.xlim((slow, shigh))
            pl.xlabel("")
            if i == min(len(sort_inds), max_display) // 2:
                pl.xlabel("Interaction value")
            pl.title(shorten_text(feature_names[ind], title_length_limit))
        
        
        pl.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        pl.subplots_adjust(hspace=0, wspace=0.1)
        
        if show:
            pl.show()
        return
    
    if max_display is None:
        max_display = 20
    
    if sort:
        # order features by the sum of their effect magnitudes
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=0), axis=0))
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)
    
    row_height = 0.4
    if auto_size_plot:
        pl.gcf().set_size_inches(size, len(feature_order) * row_height + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)
    
    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = shap_values[:, i]
            values = None if features is None else features[:, i]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
            except:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            
            ys *= 0.9 * (row_height / np.max(ys + 1))
            
            if features is not None and colored_feature:
                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                
                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"
                nan_mask = np.isnan(values)
                pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                           vmax=vmax, s=16, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
                pl.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                           cmap=red_blue, vmin=vmin, vmax=vmax, s=16,
                           c=values[np.invert(nan_mask)], alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            else:
                
                pl.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                           color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)
    
    elif plot_type == "violin":
        for pos, i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        
        if features is not None:
            global_low = np.nanpercentile(shap_values[:, :len(feature_names)].flatten(), 1)
            global_high = np.nanpercentile(shap_values[:, :len(feature_names)].flatten(), 99)
            for pos, i in enumerate(feature_order):
                shaps = shap_values[:, i]
                shap_min, shap_max = np.min(shaps), np.max(shaps)
                rng = shap_max - shap_min
                xs = np.linspace(np.min(shaps) - rng * 0.2, np.max(shaps) + rng * 0.2, 100)
                if np.std(shaps) < (global_high - global_low) / 100:
                    ds = gaussian_kde(shaps + np.random.randn(len(shaps)) * (global_high - global_low) / 100)(xs)
                else:
                    ds = gaussian_kde(shaps)(xs)
                
                ds /= np.max(ds) * 3
                
                values = features[:, i]
                window_size = max(10, len(values) // 20)
                smooth_values = np.zeros(len(xs) - 1)
                sort_inds = np.argsort(shaps)
                trailing_pos = 0
                leading_pos = 0
                running_sum = 0
                back_fill = 0
                for j in range(len(xs) - 1):
                    
                    while leading_pos < len(shaps) and xs[j] >= shaps[sort_inds[leading_pos]]:
                        running_sum += values[sort_inds[leading_pos]]
                        leading_pos += 1
                        if leading_pos - trailing_pos > 20:
                            running_sum -= values[sort_inds[trailing_pos]]
                            trailing_pos += 1
                    if leading_pos - trailing_pos > 0:
                        smooth_values[j] = running_sum / (leading_pos - trailing_pos)
                        for k in range(back_fill):
                            smooth_values[j - k - 1] = smooth_values[j]
                    else:
                        back_fill += 1
                
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                
                pl.scatter(shaps, np.ones(shap_values.shape[0]) * pos, s=9, cmap=red_blue_solid, vmin=vmin, vmax=vmax,
                           c=values, alpha=alpha, linewidth=0, zorder=1)
                # smooth_values -= nxp.nanpercentile(smooth_values, 5)
                # smooth_values /= np.nanpercentile(smooth_values, 95)
                smooth_values -= vmin
                if vmax - vmin > 0:
                    smooth_values /= vmax - vmin
                for i in range(len(xs) - 1):
                    if ds[i] > 0.05 or ds[i + 1] > 0.05:
                        pl.fill_between([xs[i], xs[i + 1]], [pos + ds[i], pos + ds[i + 1]],
                                        [pos - ds[i], pos - ds[i + 1]], color=red_blue_solid(smooth_values[i]),
                                        zorder=2)
        
        else:
            parts = pl.violinplot(shap_values[:, feature_order], range(len(feature_order)), points=200, vert=False,
                                  widths=0.7,
                                  showmeans=False, showextrema=False, showmedians=False)
            
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('none')
                pc.set_alpha(alpha)
    
    elif plot_type == "layered_violin":  # courtesy of @kodonnell
        num_x_points = 200
        bins = np.linspace(0, features.shape[0], layered_violin_max_num_bins + 1).round(0).astype(
            'int')  # the indices of the feature data corresponding to each bin
        shap_min, shap_max = np.min(shap_values), np.max(shap_values)
        x_points = np.linspace(shap_min, shap_max, num_x_points)
        
        # loop through each feature and plot:
        for pos, ind in enumerate(feature_order):
            # decide how to handle: if #unique < layered_violin_max_num_bins then split by unique value, otherwise use bins/percentiles.
            # to keep simpler code, in the case of uniques, we just adjust the bins to align with the unique counts.
            feature = features[:, ind]
            unique, counts = np.unique(feature, return_counts=True)
            if unique.shape[0] <= layered_violin_max_num_bins:
                order = np.argsort(unique)
                thesebins = np.cumsum(counts[order])
                thesebins = np.insert(thesebins, 0, 0)
            else:
                thesebins = bins
            
            nbins = thesebins.shape[0] - 1
            # order the feature data so we can apply percentiling
            order = np.argsort(feature)
            # x axis is located at y0 = pos, with pos being there for offset
            y0 = np.ones(num_x_points) * pos
            # calculate kdes:
            ys = np.zeros((nbins, num_x_points))
            for i in range(nbins):
                # get shap values in this bin:
                shaps = shap_values[order[thesebins[i]:thesebins[i + 1]], ind]
                # if there's only one element, then we can't
                if shaps.shape[0] == 1:
                    warnings.warn(
                        "not enough data in bin #%d for feature %s, so it'll be ignored. Try increasing the number of records to plot."
                        % (i, feature_names[ind]))
                    # to ignore it, just set it to the previous y-values (so the area between them will be zero). Not ys is already 0, so there's
                    # nothing to do if i == 0
                    if i > 0:
                        ys[i, :] = ys[i - 1, :]
                    continue
                # save kde of them: note that we add a tiny bit of gaussian noise to avoid singular matrix errors
                ys[i, :] = gaussian_kde(shaps + np.random.normal(loc=0, scale=0.001, size=shaps.shape[0]))(x_points)
                # scale it up so that the 'size' of each y represents the size of the bin. For continuous data this will
                # do nothing, but when we've gone with the unqique option, this will matter - e.g. if 99% are male and 1%
                # female, we want the 1% to appear a lot smaller.
                size = thesebins[i + 1] - thesebins[i]
                bin_size_if_even = features.shape[0] / nbins
                relative_bin_size = size / bin_size_if_even
                ys[i, :] *= relative_bin_size
            
            # now plot 'em. We don't plot the individual strips, as this can leave whitespace between them.
            # instead, we plot the full kde, then remove outer strip and plot over it, etc., to ensure no
            # whitespace
            ys = np.cumsum(ys, axis=0)
            width = 0.8
            scale = ys.max() * 2 / width  # 2 is here as we plot both sides of x axis
            for i in range(nbins - 1, -1, -1):
                y = ys[i, :] / scale
                c = pl.get_cmap(color)(i / (
                        nbins - 1)) if color in pl.cm.datad else color  # if color is a cmap, use it, otherwise use a color
                pl.fill_between(x_points, pos - y, pos + y, facecolor=c)
        
        pl.xlim(shap_min, shap_max)
    
    elif not multi_class and plot_type == "bar":
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        global_shap_values = np.abs(shap_values).mean(0)
        pl.barh(y_pos, global_shap_values[feature_inds], 0.7, align='center', color=color)
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])
    
    elif multi_class and plot_type == "bar":
        if class_names is None:
            class_names = ["Class "+str(i) for i in range(len(shap_values))]
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        left_pos = np.zeros(len(feature_inds))
        
        class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
        for i,ind in enumerate(class_inds):
            global_shap_values = np.abs(shap_values[ind]).mean(0)
            pl.barh(
                y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align='center',
                color=default_blue_colors[min(i, len(default_blue_colors)-1)], label=class_names[ind]
            )
            left_pos += global_shap_values[feature_inds]
        
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])
        pl.legend(frameon=False, fontsize=12)
      
    # draw the color bar
    if color_bar and features is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in pl.cm.datad):
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=red_blue_solid if plot_type != "layered_violin" else pl.get_cmap(color))
        m.set_array([0, 1])
        cb = pl.colorbar(m, ticks=[0, 1], aspect=1000)
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Feature value", size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()
       
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=12)
    if plot_type != "bar":
        pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=11)
    pl.ylim(-1, len(feature_order))
    if plot_type == "bar":
        pl.xlabel("Average impact on latent", fontsize=14)
    else:
        pl.xlabel("Summed impact on all latent dims", fontsize=14)
    if show:
        pl.show()

def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[:length_limit - 3] + "..."
    else:
        return text