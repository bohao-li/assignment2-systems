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
        return O

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Backward pass for the FlashAttention-2 autograd.Function.
        
        This method is not implemented for this task and should raise an error.
        """
        raise NotImplementedError("Backward pass is not implemented for this task.")