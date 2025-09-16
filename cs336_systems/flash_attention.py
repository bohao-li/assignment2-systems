import torch

class FlashAttentionForwardPass(torch.autograd.Function):
    """
    A pure PyTorch implementation of the FlashAttention-2 forward pass as an
    autograd.Function. This is intended for debugging purposes.
    """
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        batch_size = Q.shape[0]
        Br = 128
        Bc = 128

        # split Q into Br tiles, K and V into Bc tiles
        Q_tiles = Q.split(Br, dim=1)
        K_tiles = K.split(Bc, dim=1)
        V_tiles = V.split(Bc, dim=1)
        
        # We will collect the computed O_i tiles in this list
        O_list = []
        L_list = []

        Tr = Q.shape[1] // Br
        Tc = K.shape[1] // Bc

        for i in range(0, Tr):
            # pretend we're loading qtiles from HBM
            Q_tile = Q_tiles[i]

            # pretend we're initializing Otile, l_i and m_i on SRAM
            O_i = torch.zeros_like(Q_tile)
            l_i = torch.zeros(batch_size, Br, device=Q.device)
            m_i = torch.full((batch_size, Br), float('-inf'), device=Q.device)

            for j in range(0, Tc):
                # pretend we're loading ktiles and vtiles from HBM
                K_tile = K_tiles[j]
                V_tile = V_tiles[j]

                # compute attention scores
                S_ij = torch.einsum('b r d, b c d -> b r c', Q_tile, K_tile) / (Q.shape[-1] ** 0.5)
                
                m_i_prev = m_i.clone()
                m_i = torch.maximum(m_i, S_ij.amax(dim=-1))
                
                P_tilde_ij = torch.exp(S_ij - m_i.unsqueeze(-1))
                
                # Note: There's a slight error in your l_i update formula.
                # It should be based on the sum of P_tilde_ij values.
                # A more robust logsumexp update is:
                l_i = torch.exp(m_i_prev - m_i) * l_i + P_tilde_ij.sum(dim=-1)

                O_i = torch.exp(m_i_prev - m_i).unsqueeze(-1) * O_i + P_tilde_ij @ V_tile
            
            # Final normalization of O_i
            O_i = O_i / l_i.unsqueeze(-1)
            
            # Compute final L_i
            L_i = torch.log(l_i) + m_i

            # Append the computed tiles to the lists
            O_list.append(O_i)
            L_list.append(L_i)
        
        # Concatenate the lists of tensors back into single tensors
        O = torch.cat(O_list, dim=1)
        L = torch.cat(L_list, dim=1)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = 1 / (Q.shape[-1] ** 0.5)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        """
        PyTorch implementation of attention backward pass using saved L values.
        
        The attention mechanism is: O = softmax(Q @ K.T / scale) @ V
        We use the saved L (log-sum-exp) values to efficiently reconstruct P = softmax(S).
        
        Mathematical derivation:
        - S = Q @ K.T * scale
        - P = exp(S - L[:, None]) where L = logsumexp(S, dim=-1)
        - O = P @ V
        - dV = P.T @ dO
        - dP = dO @ V.T
        - dS = dP ⊙ P - P ⊙ (dP ⊙ P).sum(dim=-1, keepdim=True)  [softmax backward]
        - dQ = dS @ K * scale
        - dK = dS.T @ Q * scale
        """
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        
        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape
        
        # Initialize gradients
        grad_Q = torch.zeros_like(Q) if Q.requires_grad else None
        grad_K = torch.zeros_like(K) if K.requires_grad else None
        grad_V = torch.zeros_like(V) if V.requires_grad else None
        
        # Process each batch independently
        for b in range(batch_size):
            q_b = Q[b]  # (n_queries, d)
            k_b = K[b]  # (n_keys, d)
            v_b = V[b]  # (n_keys, d)
            L_b = L[b]  # (n_queries,)
            grad_output_b = grad_output[b]  # (n_queries, d)
            
            # Recompute S = Q @ K.T * scale
            S = torch.matmul(q_b, k_b.transpose(-1, -2)) * scale  # (n_queries, n_keys)
            
            # Apply causal mask if needed
            if is_causal:
                causal_mask = torch.tril(torch.ones(n_queries, n_keys, device=S.device, dtype=torch.bool))
                S = S.masked_fill(~causal_mask, float('-inf'))
            
            # Efficiently compute P using saved L values
            # P = exp(S - L[:, None]) is more numerically stable than softmax(S)
            P = torch.exp(S - L_b[:, None])  # (n_queries, n_keys)
            
            # Backward pass
            if grad_V is not None:
                # dV = P.T @ dO
                grad_V[b] = torch.matmul(P.transpose(-1, -2), grad_output_b)
            
            if grad_Q is not None or grad_K is not None:
                # dP = dO @ V.T
                dP = torch.matmul(grad_output_b, v_b.transpose(-1, -2))  # (n_queries, n_keys)
                
                # Apply causal mask to gradients if needed
                if is_causal:
                    causal_mask = torch.tril(torch.ones(n_queries, n_keys, device=dP.device, dtype=torch.bool))
                    dP = dP.masked_fill(~causal_mask, 0.0)
                
                # Softmax backward: dS = dP ⊙ P - P ⊙ (dP ⊙ P).sum(dim=-1, keepdim=True)
                # This is equivalent to: dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
                dP_P_sum = (dP * P).sum(dim=-1, keepdim=True)  # (n_queries, 1)
                dS = P * (dP - dP_P_sum)  # (n_queries, n_keys)
                
                if grad_Q is not None:
                    # dQ = dS @ K * scale
                    grad_Q[b] = torch.matmul(dS, k_b) * scale
                
                if grad_K is not None:
                    # dK = dS.T @ Q * scale
                    grad_K[b] = torch.matmul(dS.transpose(-1, -2), q_b) * scale
        
        # Return gradients in the same order as forward inputs
        # forward: (Q, K, V, scale, is_causal, Q_TILE_SIZE, K_TILE_SIZE)
        return grad_Q, grad_K, grad_V, None, None, None, None