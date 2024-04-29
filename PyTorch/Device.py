import torch
from torch import nn


# Create linear regression model class
class LinearRegressionModel(nn.Module):  # <- almost everything in PyTorch inherhits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1,  # <- start with a random weight and try to adjust it to the ideal weight
                        requires_grad=True,  # <- can this parameter be updated via gradient descent?
                        dtype=torch.float))  # <- PyTorch loves the datatype torch.float32

        self.bias = nn.Parameter(torch.randn(1,  # <- start with a random bias and try to adjust it to the ideal bias
                                             requires_grad=True,
                                             # <- can this parameter be updated via gradient descent?
                                             dtype=torch.float))  # <- PyTorch loves the datatype torch.float32

    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- "x" is the input data
        return self.weights * x + self.bias
        # this is the linear regression formula

    # Create a random seed
    torch.manual_seed(42)

    # Create an instance of the model (this is a subclass of nn.Module)
    model_0 = LinearRegressionModel()

    # Check out the parameters
    list(model_0.parameters())

    # List named parameters
    model_0.state_dict()

    y_preds = model_0(X_test)
    y_preds

    # Make predictions with model
    with torch.inference_mode():
        y_preds = model_0(X_test)

    # # You can also do something similar with torch.no_grad(), however, torch.inference_mode() is preferred
    # with torch.no_grad():
    #   y_preds = model_0(X_test)

    y_preds

    y_test

    plot_predictions(predictions=y_preds)