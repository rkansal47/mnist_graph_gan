from torch.utils.data import DataLoader
import utils
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from particlenet import ParticleNet
from jets_dataset import JetsClassifierDataset


dir_path = "/graphganvol/mnist_graph_gan/jets/"


def plot_confusion_matrix(cm, target_names,
                          fname, epoch,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm[np.isnan(cm)] = 0.0

    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + 'at epoch' + str(epoch))
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlim(-1, len(target_names))
    plt.ylim(-1, len(target_names))
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()

    plt.savefig(fname + '.pdf')
    plt.close(fig)

    return fig, ax


args = utils.objectview({'mask': False, 'datasets_path': dir_path + 'datasets/', 'node_feat_size': 3, 'num_hits': 30})

test_dataset = JetsClassifierDataset(args, train=False, lim=1000)
test_loaded = DataLoader(test_dataset)

C = ParticleNet(args.num_hits, args.node_feat_size, num_classes=5)
C.load_state_dict(torch.load(dir_path + 'particlenet/cmodels/5/C_18.pt', map_location=torch.device('cpu')))

C.eval()
correct = 0
y_true = []
y_outs = []
with torch.no_grad():
    for batch_ndx, (x, y) in tqdm(enumerate(test_loaded), total=len(test_loaded)):
        output = C(x)
        pred = output.max(1, keepdim=True)[1]
        y_outs.append(output.numpy())
        y_true.append(y)
        correct += pred.eq(y.view_as(pred)).sum()

y_true = [y.numpy() for y in y_true]
y_true = np.array(y_true).squeeze(1)

np.array(y_outs).squeeze(1).shape
y_outs = np.array(y_outs).squeeze(1)

print(f"accuracy = {correct / 5000}")

cm = confusion_matrix(y_true, y_outs.argmax(axis=1))

plot_confusion_matrix(cm, ['g', 't', 'q', 'W', 'Z'], dir_path + "particlenet/particlenetcm", 18)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_outs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


f = open(dir_path + "particlenet/roc_auc.txt", "w")
f.write(str(roc_auc))
f.close()


f = open(dir_path + "particlenet/tpr.txt", "w")
f.write(str(tpr))
f.close()


f = open(dir_path + "particlenet/fpr.txt", "w")
f.write(str(fpr))
f.close()
