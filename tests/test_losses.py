import numpy as np
from src.losses import MSELoss, BCELoss


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


def test_bse_loss_forward():
    loss_fn = BCELoss()
    y_pred = np.array([[0.5], [1.0], [0.8]])
    y_true = np.array([[1.0], [0.0], [1.0]])

    calculated_loss = loss_fn.forward(y_pred, y_true)
    expected_loss = 11.8186
    assert np.isclose(calculated_loss, expected_loss), f"Expected {expected_loss} but got {calculated_loss}"