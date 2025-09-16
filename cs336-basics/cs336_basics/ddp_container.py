import torch
import torch.nn as nn
import torch.distributed as dist

class DDPContainerAsync(nn.Module):
    """
    Async version of DDPContainer
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self._pending_handles = []
        
        # Ensure identical initial parameters
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
        
        # Register hooks
        self._register_grad_hooks()
    
    def _register_grad_hooks(self):
        """Register async gradient synchronization hooks"""
        def make_hook(param):
            def hook(grad):
                if grad is not None:
                    # Average the gradient
                    grad.data.div_(self.world_size)
                    # Start async all-reduce
                    handle = dist.all_reduce(grad.data, async_op=True)
                    self._pending_handles.append(handle)
                return grad
            return hook
        
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(make_hook(param))
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Wait for all pending gradient reductions to complete"""
        for handle in self._pending_handles:
            handle.wait()
        self._pending_handles.clear()