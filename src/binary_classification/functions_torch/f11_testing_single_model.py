import time
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score


def testing(device, model, x_testing, y_testing):
    """
        :param device: CPU or GPU
        :param model: model to test
        :param x_testing: testing set without labels
        :param y_testing: testing set with only labels
    """
    # Testing Model
    set_t0 = time.time()
    model.eval()
    with torch.no_grad():
        outputs = model(x_testing)
        _, predicted = torch.max(outputs, 1)
        class_weights = torch.tensor([len(y_testing) / (2 * torch.sum(y_testing == 0)),
                                      len(y_testing) / (2 * torch.sum(y_testing == 1))], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Calculating Metrics
        final_loss = criterion(outputs, y_testing)
        final_accuracy = (predicted == y_testing).sum().item() / y_testing.size(0)
        precision = precision_score(y_testing.cpu(), predicted.cpu(), average='binary')
        recall = recall_score(y_testing.cpu(), predicted.cpu(), average='binary')
        f1 = f1_score(y_testing.cpu(), predicted.cpu(), average='binary')

        # Printing Metrics
        print(f'\t--> Testing took {time.time() - set_t0:.4f} sec')
        print(f'\t--> Final Accuracy: {final_accuracy:.4f}')
        print(f'\t--> Final Loss: {final_loss.item():.4f}')
        print(f'\t--> Final Precision: {precision:.4f}')
        print(f'\t--> Final Recall: {recall:.4f}')
        print(f'\t--> Final F1 Score: {f1:.4f}')
