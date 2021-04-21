import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def draw_4lines(history, set, results_folder, classes):
    graph_folder = results_folder + "/plots"
    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)

    for i, cur_class in enumerate(classes):
        curr_class_history = pd.Series([el[i] for el in history])
        # Loss plot
        plt.figure()
        plt.hist(curr_class_history)

        plt.title(set + ' distribution ' + cur_class)
        plt.ylabel(cur_class)
        plt.xlabel('images')
        plt.savefig(graph_folder + "/" + set + ' distribution ' + cur_class + ".png")

        layout = go.Layout(
            title=set + ' distribution ' + cur_class,
            plot_bgcolor="#FFFFFF",
            hovermode="x",
            hoverdistance=100,  # Distance to show hover label of data point
            spikedistance=1000,  # Distance to show spike
            xaxis=dict(
                title="images",
                linecolor="#BCCCDC",
                showspikes=True,  # Show spike line for X-axis
                # Format spike
                spikethickness=2,
                spikedash="dot",
                spikecolor="#999999",
                spikemode="across",
            ),
            yaxis=dict(
                title=cur_class,
                linecolor="#BCCCDC"
            )
        )

        fig = px.histogram(curr_class_history, title=set + ' distribution ' + cur_class)
        fig.write_html(graph_folder + "/" + set + ' distribution ' + cur_class + ".html")

    return


def plot_confusion_matrix(cm, class_list, title, normalized=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalized=True`.

    Based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    fig, ax = plt.subplots(figsize=(24, 24))
    im = ax.imshow(cm[1:5], interpolation='nearest', cmap=cmap)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]-2),
           yticks=np.arange(cm.shape[0]-2),
           # ... and label them with the respective list entries
           xticklabels=class_list[1:5], yticklabels=class_list[1:5],
            )
    plt.xlabel('Predicted label', fontsize=40)
    plt.ylabel('True label', fontsize=40)
    ax.set_title(title, pad=30, fontsize=40)

    # Turn spines off.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=40)
    plt.setp(ax.get_yticklabels(), fontsize=40)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalized else 'd'
    thresh = cm[1:5].max() / 2.
    for i in range(cm.shape[0]-2):
        for j in range(cm.shape[1]-2):
            ax.text(j+1, i+1, format(cm[i+1, j+1], fmt),
                    ha="center", va="center",
                    color="white" if cm[i+1, j+1] > thresh or cm[i+1, j+1] < 0.01 else "black", fontsize=36
                    )
    fig.tight_layout()
    return ax


def draw_history_plots(history, experiment_name, results_folder):
    graph_folder = results_folder + "/plots"
    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(experiment_name+' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder+"/loss.png")
    plt.close()

    layout = go.Layout(
        title=experiment_name+' model loss',
        plot_bgcolor="#FFFFFF",
        hovermode="x",
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        xaxis=dict(
            title="epoch",
            linecolor="#BCCCDC",
            showspikes=True,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
        ),
        yaxis=dict(
            title="Loss",
            linecolor="#BCCCDC"
        )
    )

    data = []
    for line in ['loss', 'val_loss']:
        loss = history.history[line]
        line_chart = go.Scatter(
            y=loss,
            name=line
        )
        data.append(line_chart)

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(graph_folder+"/loss.html")

    # Accuracy plot
    plt.figure()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title(experiment_name+' model accuracy')
    plt.ylabel('categorical accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder + "/cat_accuracy.png")
    plt.close()

    layout = go.Layout(
        title=experiment_name+' model accuracy',
        plot_bgcolor="#FFFFFF",
        hovermode="x",
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        xaxis=dict(
            title="epoch",
            linecolor="#BCCCDC",
            showspikes=True,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
        ),
        yaxis=dict(
            title="categorical accuracy",
            linecolor="#BCCCDC"
        )
    )

    data = []
    for line in ['categorical_accuracy', 'val_categorical_accuracy']:
        loss = history.history[line]
        line_chart = go.Scatter(
            y=loss,
            name=line
        )
        data.append(line_chart)

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(graph_folder + "/cat_accuracy.html")

    # IoU plot
    plt.figure()
    plt.plot(history.history['mean_io_u'])
    plt.plot(history.history['val_mean_io_u'])
    plt.title(experiment_name+' model IoU')
    plt.ylabel('Mean IoU')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder + "/meaniou.png")
    plt.close()

    layout = go.Layout(
        title=experiment_name+' model IoU',
        plot_bgcolor="#FFFFFF",
        hovermode="x",
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        xaxis=dict(
            title="epoch",
            linecolor="#BCCCDC",
            showspikes=True,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
        ),
        yaxis=dict(
            title="mean_io_u",
            linecolor="#BCCCDC"
        )
    )

    data = []
    for line in ['mean_io_u', 'val_mean_io_u']:
        loss = history.history[line]
        line_chart = go.Scatter(
            y=loss,
            name=line
        )
        data.append(line_chart)

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(graph_folder + "/mean_io_u.html")

    # Precision
    plt.figure()
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title(experiment_name+' model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder + "/precision.png")
    plt.close()

    layout = go.Layout(
        title=experiment_name + ' model precision',
        plot_bgcolor="#FFFFFF",
        hovermode="x",
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        xaxis=dict(
            title="epoch",
            linecolor="#BCCCDC",
            showspikes=True,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
        ),
        yaxis=dict(
            title="precision",
            linecolor="#BCCCDC"
        )
    )

    data = []
    for line in ['precision', 'val_precision']:
        loss = history.history[line]
        line_chart = go.Scatter(
            y=loss,
            name=line
        )
        data.append(line_chart)

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(graph_folder + "/precision.html")

    # Recall
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title(experiment_name+' model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder + "/recall.png")
    plt.close()

    layout = go.Layout(
        title=experiment_name + ' model recall',
        plot_bgcolor="#FFFFFF",
        hovermode="x",
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        xaxis=dict(
            title="epoch",
            linecolor="#BCCCDC",
            showspikes=True,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
        ),
        yaxis=dict(
            title="recall",
            linecolor="#BCCCDC"
        )
    )

    data = []
    for line in ['recall', 'val_recall']:
        loss = history.history[line]
        line_chart = go.Scatter(
            y=loss,
            name=line
        )
        data.append(line_chart)

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(graph_folder + "/recall.html")

    # Recall
    plt.plot(history.history['custom_f1'])
    plt.plot(history.history['val_custom_f1'])
    plt.title(experiment_name + ' model coefficient')
    plt.ylabel('dice coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder + "/f1.png")
    plt.close()

    layout = go.Layout(
        title=experiment_name + ' model dice coefficient',
        plot_bgcolor="#FFFFFF",
        hovermode="x",
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        xaxis=dict(
            title="epoch",
            linecolor="#BCCCDC",
            showspikes=True,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across",
        ),
        yaxis=dict(
            title="coefficient",
            linecolor="#BCCCDC"
        )
    )

    data = []
    for line in ['custom_f1', 'val_custom_f1']:
        loss = history.history[line]
        line_chart = go.Scatter(
            y=loss,
            name=line
        )
        data.append(line_chart)

    fig = go.Figure(data=data, layout=layout)
    fig.write_html(graph_folder + "/f1.html")
