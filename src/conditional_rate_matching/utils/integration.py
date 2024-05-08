import torch


def integrate_quad_tensor_vec(func, a, b, segments=100):
    """
    Integrate a function using the trapezoidal rule with tensor limits, vectorized.

    Args:
    func (callable): The function to integrate. Must accept and return PyTorch tensors.
    a (torch.Tensor): The lower limit tensor of integration.
    b (torch.Tensor): The upper limit tensor of integration.
    segments (int): The number of segments to divide each interval into.

    Returns:
    torch.Tensor: The result of the integration for each interval.
    """
    device = a.device
    assert b.device == device

    # Ensure a and b are tensors and have the same size
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    if a.size() != b.size():
        raise ValueError("a and b must have the same size")

    # Create a grid of points for integration
    x = torch.linspace(0, 1, steps=segments + 1).view(-1, 1)
    x = x.to(device)
    x = x * (b - a) + a  # Broadcasting to create the grid

    # Compute the function values at these points
    y = func(x)

    # Compute the weights for the trapezoidal rule
    h = (b - a) / segments
    weights = torch.full((segments + 1, 1), 2.0)  # Adjusted shape for broadcasting
    weights = weights.to(device)
    weights[0, 0] = weights[-1, 0] = 1.0

    # Perform the integration
    results = h / 2 * torch.sum(weights * y, dim=0)

    return results


# Example usage
def example_func(x):
    return torch.sin(x)


if __name__=="__main__":
    # Tensor limits of integration
    a = torch.tensor([0, 1])
    b = torch.tensor([torch.pi, torch.pi * .25])

    result = integrate_quad_tensor_vec(example_func, a, b, 100)