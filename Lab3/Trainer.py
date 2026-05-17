import torch
from sklearn.metrics import f1_score, confusion_matrix


def train(model, data, optimizer, criterion, clip):
    model.train()

    loss_total = 0.0

    for batch in data:
        x, y, lengths = batch
        model.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loss_total += loss.item()
    
    return loss_total / len(data)


def evaluate(model, data, criterion):
    model.eval()
    loss_total = 0.0

    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in data:
            x, y, lengths = batch
            logits = model(x, lengths)
            loss = criterion(logits, y.float())
            loss_total += loss.item()
            preds = (logits > 0).long()
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(y.cpu().tolist())

    avg_loss = loss_total / len(data)
    acc = (torch.tensor(preds_all) == torch.tensor(labels_all)).float().mean().item()
    f1 = f1_score(labels_all, preds_all, zero_division=0)
    conf = confusion_matrix(labels_all, preds_all)
    return avg_loss, acc, f1, conf