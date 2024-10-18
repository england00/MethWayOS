import time
import torch


def testing(model, x_testing, y_testing):
    """
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
        final_accuracy = (predicted == y_testing).sum().item() / y_testing.size(0)
        print(f'\t--> Testing took {time.time() - set_t0} sec')
        print(f'\t--> Final Accuracy: {final_accuracy:.4f}')
