import time
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score


def testing(models, x_testing, y_testing):
    """
        :param models: list of models to test
        :param x_testing: testing set without labels
        :param y_testing: testing set with only labels
    """
    # Testing Model
    set_t0 = time.time()
    for model in models:
        model.eval()

    # Predicting outputs
    ensemble_predictions = []
    with torch.no_grad():
        for model in models:
            outputs = model(x_testing)
            _, predicted = torch.max(outputs, 1)
            ensemble_predictions.append(predicted)

    # Managing predictions and voting
    ensemble_predictions = torch.stack(ensemble_predictions)
    final_predicted = torch.mode(ensemble_predictions, dim=0)[0]
    criterion = nn.CrossEntropyLoss()

    # Calculating Metrics
    final_loss = criterion(outputs, y_testing)
    final_accuracy = (final_predicted == y_testing).sum().item() / y_testing.size(0)
    precision = precision_score(y_testing.cpu(), final_predicted.cpu(), average='binary')
    recall = recall_score(y_testing.cpu(), final_predicted.cpu(), average='binary')
    f1 = f1_score(y_testing.cpu(), final_predicted.cpu(), average='binary')

    # Printing Metrics
    print(f'\t--> Testing took {time.time() - set_t0:.4f} sec')
    print(f'\t--> Final Accuracy: {final_accuracy:.4f}')
    print(f'\t--> Final Loss: {final_loss.item():.4f}')
    print(f'\t--> Final Precision: {precision:.4f}')
    print(f'\t--> Final Recall: {recall:.4f}')
    print(f'\t--> Final F1 Score: {f1:.4f}')
