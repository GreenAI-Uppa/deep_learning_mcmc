import torch
from sklearn.metrics import confusion_matrix

def evaluate(dataloader, model, loss_fn):
    device = next(model.parameters()).device
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    #test_loss /= size
    correct /= size
    return test_loss, correct

def get_conf_matrix_imgclass(pairs, classes=None, normalize=False, unknown_class="OTHER"):
    """
    pairs : pairs of detection and ground truths
    """
    y_true = []
    y_pred = []
    if classes is None:
        # getting all the classes
        classes = sorted(set([pair["cls_gr"] for pair in pairs]))

    for pair in pairs:
        cls_gt, cls_pred = pair["cls_gr"], pair["cls_pred"]
        y_true.append(classes.index(cls_gt))
        if cls_pred in classes:
            y_pred.append(classes.index(cls_pred))
        else: # We had one tricky case where the predictions have labels which are not in the groundtruth
            y_pred.append(classes.index(unknown_class))
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        # row normalization
        cm = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100)).astype('int')
    return cm, classes