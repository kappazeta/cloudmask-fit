import matplotlib.pyplot as plt
import os


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

    # Accuracy plot
    plt.figure()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title(experiment_name+' model accuracy')
    plt.ylabel('categorical accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder + "/cat_accuracy.png")

    # IoU plot
    plt.figure()
    plt.plot(history.history['mean_io_u'])
    plt.plot(history.history['val_mean_io_u'])
    plt.title(experiment_name+' model IoU')
    plt.ylabel('Mean IoU')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder + "/meaniou.png")

    # Precision
    plt.figure()
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title(experiment_name+' model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder + "/precision.png")
    # Recall
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title(experiment_name+' model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(graph_folder + "/recall.png")
