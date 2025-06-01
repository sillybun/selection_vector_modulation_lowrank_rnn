def power_method_batch(A, num_iterations=100, tolerance=1e-6):
    batch_size, n, _ = A.shape
    # Initialize a random vector for each batch
    b_k = torch.rand((batch_size, n), dtype=A.dtype, device=A.device)
    b_k = b_k / torch.norm(b_k, dim=1, keepdim=True)  # Normalize

    for _ in range(num_iterations):
        # Compute A * b_k for each batch
        b_k1 = torch.bmm(A, b_k.unsqueeze(-1)).squeeze(-1)

        # Normalize
        b_k1_norm = torch.norm(b_k1, dim=1, keepdim=True)
        b_k1 = b_k1 / b_k1_norm

        # Check for convergence
        if torch.norm(b_k1 - b_k) < tolerance:
            break

        b_k = b_k1

    # Compute the corresponding eigenvalues
    eigenvalues = torch.bmm(b_k.unsqueeze(1), torch.bmm(A, b_k.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    # Ensure the eigenvector is normalized
    eigenvalues = eigenvalues / torch.sum(b_k**2, dim=1)
    return eigenvalues, b_k