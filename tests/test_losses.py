import numpy as np
from src.losses import MSELoss


def test_mse_loss_forward():
    # 1. Arrange (Set up your test data)
    loss_fn = MSELoss()
    y_pred = np.array([[2.0], [4.0], [7.0]])
    y_true = np.array([[3.0], [3.0], [9.0]])
    
    # 2. Act (Run the target method)
    calculated_loss = loss_fn.forward(y_pred, y_true)
    
    # 3. Assert (Check if the result matches the expected hand-calculated math)
    # Expected: ((3-2)^2 + (3-4)^2 + (9-7)^2) / 3 = (1 + 1 + 4) / 3 = 2.0
    assert np.isclose(calculated_loss, 2.0), f"Expected 2.0 but got {calculated_loss}"
