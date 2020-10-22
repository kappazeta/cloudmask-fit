import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go


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
