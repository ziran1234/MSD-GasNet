import matplotlib.pyplot as plt


def down_sample(array, step):
    """Return every ``step``-th element from an input sequence."""

    new_array = []
    size = len(array)
    for i in range(0, size, step):
        new_array.append(array[i])
    return new_array


def plot_figure(figure_size, xdata, ydata, title="", xlabel='Time (s)', ylabel='Response', labels=None, color="darkred"):
    """Plot a single response curve with the project's publication-style axes."""

    colors = ['darkred', 'darkgoldenrod', 'darkgreen', 'darkslategray', 'darkblue', 'purple', 'gray', 'olive']
    legend_font = {
        'family': 'Arial',  # Font family.
        'style': 'normal',
        'size': 30,  # Font size.
        'weight': "bold",
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    bold_font = {'fontname': 'Arial', 'weight': 'bold'}
    axes.spines['bottom'].set_linewidth(5)
    axes.spines['left'].set_linewidth(5)
    axes.spines['right'].set_linewidth(5)
    axes.spines['top'].set_linewidth(5)
    # axes.set_title(title, fontsize=26, **bold_font)
    axes.set_xlabel(xlabel, labelpad=1, fontsize=40, **bold_font)
    axes.set_ylabel(ylabel, labelpad=5, fontsize=40, **bold_font)
    axes.plot(xdata, ydata, color=color)
    axes.tick_params(axis='x', labelsize=36, direction='out', width=4, length=10)
    axes.tick_params(axis='y', labelsize=36, direction='out', width=4, length=10)
    x_label = axes.get_xticklabels()
    [x_label_temp.set_fontweight('bold') for x_label_temp in x_label]
    y_label = axes.get_yticklabels()
    [y_label_temp.set_fontweight('bold') for y_label_temp in y_label]
    # axes.legend(prop=legend_font, loc="upper left")
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    # img_path = "./Domain Loss Img/" + 'train_test_with_Domain_fit' + str(index) + '.png'
    # plt.savefig(title + ".png")
    plt.show()


def generate_fig(figure_size, title="", xlabel='Time (s)', ylabel='Response'):
    """Create and style a Matplotlib figure, returning ``(fig, axes)``."""

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    bold_font = {'fontname': 'Arial', 'weight': 'bold'}
    axes.spines['bottom'].set_linewidth(5)
    axes.spines['left'].set_linewidth(5)
    axes.spines['right'].set_linewidth(5)
    axes.spines['top'].set_linewidth(5)
    # plt.xlim(60, 100)
    # plt.ylim(0.7, 1)
    # axes.set_title(title, fontsize=26, **bold_font)
    axes.set_xlabel(xlabel, labelpad=1, fontsize=40, **bold_font)
    axes.set_ylabel(ylabel, labelpad=5, fontsize=40, **bold_font)
    axes.tick_params(axis='x', labelsize=36, direction='out', width=4, length=10)
    axes.tick_params(axis='y', labelsize=36, direction='out', width=4, length=10)
    x_label = axes.get_xticklabels()
    [x_label_temp.set_fontweight('bold') for x_label_temp in x_label]
    y_label = axes.get_yticklabels()
    [y_label_temp.set_fontweight('bold') for y_label_temp in y_label]
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    return fig, axes


def plot_figure_mapCAM(
    figure_size,
    xdata,
    ydata,
    final_count_array,
    index,
    title="",
    xlabel='Time (s)',
    ylabel='Response',
    labels=None,
    color="darkred",
):
    """Plot one response curve and overlay sampled CAM segment positions."""

    colors = ['darkred', 'darkgoldenrod', 'darkgreen', 'darkslategray', 'darkblue', 'purple', 'gray', 'olive']
    legend_font = {
        'family': 'Arial',  # Font family.
        'style': 'normal',
        'size': 30,  # Font size.
        'weight': "bold",
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    bold_font = {'fontname': 'Arial', 'weight': 'bold'}
    axes.spines['bottom'].set_linewidth(5)
    axes.spines['left'].set_linewidth(5)
    axes.spines['right'].set_linewidth(5)
    axes.spines['top'].set_linewidth(5)
    # axes.set_title(title, fontsize=26, **bold_font)
    axes.set_xlabel(xlabel, labelpad=1, fontsize=40, **bold_font)
    axes.set_ylabel(ylabel, labelpad=5, fontsize=40, **bold_font)
    axes.plot(xdata, ydata, color=color)
    count_array = final_count_array[index]
    count_array = down_sample(count_array, 10)
    print(len(count_array))
    for i in range(0, len(count_array)):
        print(i)
        point = count_array[i]
        x1 = xdata[point[0]]
        x2 = xdata[point[1]]
        y1 = ydata[point[0]]
        y2 = ydata[point[1]]
        axes.plot([x1, x2], [y1, y2], color='y')
        axes.scatter([x1, x2], [y1, y2], color='b')
    axes.tick_params(axis='x', labelsize=36, direction='out', width=4, length=10)
    axes.tick_params(axis='y', labelsize=36, direction='out', width=4, length=10)
    x_label = axes.get_xticklabels()
    [x_label_temp.set_fontweight('bold') for x_label_temp in x_label]
    y_label = axes.get_yticklabels()
    [y_label_temp.set_fontweight('bold') for y_label_temp in y_label]
    # axes.legend(prop=legend_font, loc="upper left")
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    # img_path = "./Domain Loss Img/" + 'train_test_with_Domain_fit' + str(index) + '.png'
    # plt.savefig(title + ".png")
    plt.show()


def plot_figure_countCAMmap(
    figure_size,
    xdata,
    ydata,
    count_array,
    response_index,
    title="",
    xlabel='Time (s)',
    ylabel='Response',
    labels=None,
    color="darkred",
):
    """Plot one response curve and overlay sampled CAM ranges from ``count_array``."""

    colors = ['darkred', 'darkgoldenrod', 'darkgreen', 'darkslategray', 'darkblue', 'purple', 'gray', 'olive']
    legend_font = {
        'family': 'Arial',  # Font family.
        'style': 'normal',
        'size': 30,  # Font size.
        'weight': "bold",
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    bold_font = {'fontname': 'Arial', 'weight': 'bold'}
    axes.spines['bottom'].set_linewidth(5)
    axes.spines['left'].set_linewidth(5)
    axes.spines['right'].set_linewidth(5)
    axes.spines['top'].set_linewidth(5)
    # axes.set_title(title, fontsize=26, **bold_font)
    axes.set_xlabel(xlabel, labelpad=1, fontsize=40, **bold_font)
    axes.set_ylabel(ylabel, labelpad=5, fontsize=40, **bold_font)
    axes.plot(xdata, ydata, color=color)
    # axes.axvline(xdata[0], color='red')
    # axes.axvline(xdata[0 + response_index * 300], color='red')
    count_array = down_sample(count_array, 10)
    print(count_array)
    print(len(count_array))
    for i in range(0, len(count_array)):
        print(i)
        point = count_array[i]
        x1 = xdata[point[0]]
        x2 = xdata[point[1]]
        y1 = ydata[point[0]]
        y2 = ydata[point[1]]
        axes.plot([x1, x2], [y1, y2], color='y')
        axes.scatter([x1, x2], [y1, y2], color='b')
    axes.tick_params(axis='x', labelsize=36, direction='out', width=4, length=10)
    axes.tick_params(axis='y', labelsize=36, direction='out', width=4, length=10)
    x_label = axes.get_xticklabels()
    [x_label_temp.set_fontweight('bold') for x_label_temp in x_label]
    y_label = axes.get_yticklabels()
    [y_label_temp.set_fontweight('bold') for y_label_temp in y_label]
    # axes.legend(prop=legend_font, loc="upper left")
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    # img_path = "./Domain Loss Img/" + 'train_test_with_Domain_fit' + str(index) + '.png'
    # plt.savefig(title + ".png")
    plt.show()


def plot_comparison_figure(figure_size, origin_data, pre_data, title=""):
    """Plot ground-truth and predicted response curves on the same axes."""

    colors = ['darkblue', 'darkred']
    labels = ["true", "predict"]
    legend_font = {
        'family': 'Arial',  # Font family.
        'style': 'normal',
        'size': 30,  # Font size.
        'weight': "bold",
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    bold_font = {'fontname': 'Arial', 'weight': 'bold'}
    axes.spines['bottom'].set_linewidth(5)
    axes.spines['left'].set_linewidth(5)
    axes.spines['right'].set_linewidth(5)
    axes.spines['top'].set_linewidth(5)
    title = title
    axes.set_xlabel('Time (s)', labelpad=1, fontsize=40, **bold_font)
    axes.set_ylabel('Response', labelpad=5, fontsize=40, **bold_font)
    o_xdata = origin_data[0]
    o_ydata = origin_data[1]
    axes.plot(o_xdata, o_ydata, color=colors[0], label=labels[0])
    p_xdata = pre_data[0]
    p_ydata = pre_data[1]
    axes.plot(p_xdata, p_ydata, color=colors[1], label=labels[1])
    axes.tick_params(axis='x', labelsize=36, direction='out', width=4, length=10)
    axes.tick_params(axis='y', labelsize=36, direction='out', width=4, length=10)
    x_label = axes.get_xticklabels()
    [x_label_temp.set_fontweight('bold') for x_label_temp in x_label]
    y_label = axes.get_yticklabels()
    [y_label_temp.set_fontweight('bold') for y_label_temp in y_label]
    axes.legend(prop=legend_font, loc="upper left")
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    # img_path = "./Domain Loss Img/" + 'train_test_with_Domain_fit' + str(index) + '.png'
    # plt.savefig(img_path)
    plt.show()


def plot_train_test_figure(
    figure_size,
    xdata,
    ydata1,
    ydata2,
    title="",
    xlabel='Time (s)',
    ylabel='Response',
    labels=None,
    colors=None,
):
    """Plot train/test metric curves, usually loss, accuracy, or R2."""

    # colors = ['darkred', 'darkgoldenrod', 'darkgreen', 'darkslategray', 'darkblue', 'purple', 'gray', 'olive']
    legend_font = {
        'family': 'Arial',  # Font family.
        'style': 'normal',
        'size': 30,  # Font size.
        'weight': "bold",
    }
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figure_size)
    bold_font = {'fontname': 'Arial', 'weight': 'bold'}
    axes.spines['bottom'].set_linewidth(5)
    axes.spines['left'].set_linewidth(5)
    axes.spines['right'].set_linewidth(5)
    axes.spines['top'].set_linewidth(5)
    # axes.set_title(title, fontsize=26, **bold_font)
    axes.set_xlabel(xlabel, labelpad=1, fontsize=40, **bold_font)
    axes.set_ylabel(ylabel, labelpad=5, fontsize=40, **bold_font)
    axes.plot(xdata, ydata1, color=colors[0], label=labels[0])
    axes.plot(xdata, ydata2, color=colors[1], label=labels[1])
    axes.tick_params(axis='x', labelsize=36, direction='out', width=4, length=10)
    axes.tick_params(axis='y', labelsize=36, direction='out', width=4, length=10)
    x_label = axes.get_xticklabels()
    [x_label_temp.set_fontweight('bold') for x_label_temp in x_label]
    y_label = axes.get_yticklabels()
    [y_label_temp.set_fontweight('bold') for y_label_temp in y_label]
    axes.legend(prop=legend_font)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    # img_path = "./Domain Loss Img/" + 'train_test_with_Domain_fit' + str(index) + '.png'
    # plt.savefig(title + ".png")

    plt.show()
