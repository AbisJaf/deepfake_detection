import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs=5, total_epochs=30):
    """
    Implements the LR schedule defined in TDD Table 5.1:
    - Phase 1 (Epoch 1-5): Linear Warmup
    - Phase 2 (Epoch 6-30): Cosine Annealing
    """
    
    # 1. The Warmup Phase
    # We start the learning rate at 10% (0.1) of its target value.
    # Over the first 5 epochs, it linearly climbs up to 100% (1.0).
    # This prevents the newly initialized, chaotic fusion layers from destroying the weights early on.
    warmup = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )

    # 2. The Cosine Annealing Phase
    # Once the warm-up is done, this takes over. It traces the curve of a cosine wave,
    # gently curving the learning rate downwards until it hits near-zero (1e-6) at Epoch 30.
    # This allows the network to take tiny, precise steps as it gets closer to maximum accuracy.
    cosine = CosineAnnealingLR(
        optimizer, 
        T_max=(total_epochs - warmup_epochs), 
        eta_min=1e-6
    )

    # 3. The Sequence Manager
    # This acts as the gearbox, automatically shifting from the Warmup schedule to the Cosine schedule 
    # exactly at the 5-epoch milestone.
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup, cosine], 
        milestones=[warmup_epochs]
    )

    return scheduler