import torch


def testing(model, X_testing, y_testing):
    """
        :param model: model to test
        :param X_testing: testing set without labels
        :param y_testing: testing set with only labels
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_testing)
        _, predicted = torch.max(outputs, 1)
        final_accuracy = (predicted == y_testing).sum().item() / y_testing.size(0)
        print(f'\t--> Final Accuracy: {final_accuracy:.4f}')
