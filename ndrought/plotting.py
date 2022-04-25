import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from wrangle import da_summary_stats


def plot_moy_overlay(da: xr.DataArray, ax=None, start_year=None, end_year=None, 
                    cmap=None, cbar=False, color=None, alpha=1, label=True):
    """Overlay each year of da by month of year.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with parameters time and park expected.
    ax : matplotlib axis, (optional)
        What axis to plot on, all years are plotted on the same axis. If
        None is provided, one is created.
    start_year : int-like, (optional)
        Year to start plotting from.
    end_year : int-like, (optional)
        Last year to plot.
    cmap : matplotlib colormap, (optional)
        Color-coding for each year to create a color gradient as plotting
        progresses from start_year to end_year. If None is provided, default
        color is used instead.
    cbar : boolean, (optional)
        Whether to plot the colorbar on ax, defaults as False.
    color : string, (optional)
        If cmap is None, then colors each year the same with this parameter.
        If None is provided here, then matplotlib defaults are used.
    alpha : float, (optional)
        Transparency of each year plotted, uniform for all years, defaults 1.
    label : boolean, (optional)
        Whether to label the axis and title of ax, defaults as True. If you
        are planning ot use figure labels, then consider turning this to 
        False to reduce clutter. Assumes that y is Day Visits and that da
        has a dimension called park.
    
    Returns
    -------
    ax : matplotlib axis
        The axis object plotted on.
    """
    # check if given years to start/end, and set them to the start/end of the da
    # if not given
    if start_year is None:
        start_year = pd.to_datetime(da.time.isel(time=0).values).year
    if end_year is None:
        end_year = pd.to_datetime(da.time.isel(time=-1).values).year

    # see if we need to create an axis object
    if ax is None:
        __, ax = plt.subplots()

    # check if we want to plot the colorbar
    if cbar:
        # double check that they actually gave a cmap
        if cmap is None:
            raise Exception('Please specify a cmap if you wish to plot a colorbar')
        # plot the colorbar
        norm = mpl.colors.Normalize(vmin=start_year, vmax=end_year)
        scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(scalar_mappable, label='Year')

    # plot by slicing years, (don't forget the +1 so we
    # actually end up plottin the last year!)
    for year in np.arange(start_year, end_year+1, 1):
        # quick little slice and grab the months to plot
        year_da = da.sel(time=slice(f'{year}',f'{year}'))
        month = pd.to_datetime(year_da.time).month

        # if we were given a cmap, we need to specify the color via the cmap
        if cmap:
            ax.plot(month, year_da.values, color=cmap((year-start_year)/(end_year-start_year)), alpha=alpha)
        # otherwise we can assign whatever color is given
        else:
            ax.plot(month, year_da.values, color=color, alpha=alpha)

    # check if want to label and do so if so (may not want to if applying suplabels)
    if label:
        ax.set_xlabel('Month of Year')
        ax.set_ylabel('Day Visits')
        ax.set_title(da.park.values);

    return ax

def plot_summary_stats(da:xr.DataArray, ax=None, line_colors=['k','r','b'], std_color='grey', std_alpha=0.2, label=True):
    """Plots summary statistics of da, including mean, max, min, and one standard deviation range.

    Parameters
    ----------
    da : xr.DataArray
        Data to plot.
    ax : matplotlib axis object, (optional)
        What to plot da on. If no ax is provided, one will be created.
    line_colors : list, (optional)
        Colors to plot the line data, (mean, max, and min in that order).
        Defaults as ['k', 'r', 'b'].
    std_color : str, (optional)
        Fill color for 1-std range, defaults as 'grey'.
    std_alpha : float, (optional)
        Transparency for 1-std range filled area, defaults as 0.2.
    label : boolean, (optional)
        Whether to label the axis and title assuming this as a parks
        dataset that is plotting 'Day Visits' on the y-axis, and 
        can call da.park.values to get the name of the park plotted.
        Turn this off for other data or to declutter.
    """
    if ax is None:
        __, ax = plt.subplots()
    
    mean, std, max, min = da_summary_stats(da, ['mean','std','max','min'])
    
    for color, stat, name in zip(line_colors, [mean, max, min], ['Mean', 'Max', 'Min']):
        ax.plot(stat.month, stat.values, color=color, label=name)
    ax.fill_between(mean.month, mean.values-std.values, mean.values+std.values, 
    color=std_color, alpha=std_alpha, label='1-STD')
    
    if label:
        ax.legend()
        ax.set_xlabel('Month of Year')
        ax.set_ylabel('Day Visits')
        ax.set_title(da.park.values)

    return ax

def scatter_linregress(x:np.ndarray, y:np.ndarray, ax=None, title_var=None):
    """Create a scatter between x and y and perform linear regression.

    Parameters
    ----------
    x, y : np.ndarray
        1D data to plot against each other with an equal number of values each.
    ax : matplotlib axis, (optional)
        Where to plot the scatter plot. One will be created with default settings
        if none is provided.
    title_var : str, (optional)
        Additional specification for title label. R-squared value will be displayed
        in the title regardless.

    Returns
    -------
    ax
        Axis object containing created plot.
    """

    if ax is None:
        __, ax = plt.subplots()

    if len(x) != len(y):
        raise Exception('x and y do not have an equal number of points, they cannot be paired for scatter')

    x_reshape = x.reshape((-1, 1))

    model = LinearRegression().fit(x_reshape, y)
    #model.fit(x_reshape.squeeze(), y.squeeze())
    r_sq = model.score(x_reshape, y)

    x_new = np.arange(x.min(), x.max()).reshape((-1,1))
    y_new = model.predict(x_new)

    ax.scatter(x, y)
    ax.plot(x_new, y_new, linestyle='--', color='r')

    if title_var and isinstance(title_var, str):
        ax.set_title(f'{title_var}, '+r'$R^2$'+f' = {r_sq:.3f}')
    else:
        ax.set_title(r'$R^2$'+f' = {r_sq:.3f}')

    return ax

    