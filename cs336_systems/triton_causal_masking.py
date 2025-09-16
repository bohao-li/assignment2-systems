import torch
import triton.language as tl

# Let's simulate what q_indices looks like in different scenarios

def show_q_indices_examples():
    print("=" * 60)
    print("EXAMPLES OF q_indices = start_q + tl.arange(0, Q_TILE_SIZE)")
    print("=" * 60)
    
    # Example parameters
    Q_TILE_SIZE = 4  # Small tile size for clarity
    N_QUERIES = 10   # Total sequence length
    
    print(f"Scenario: N_QUERIES = {N_QUERIES}, Q_TILE_SIZE = {Q_TILE_SIZE}")
    print(f"This means we'll have {N_QUERIES // Q_TILE_SIZE + (1 if N_QUERIES % Q_TILE_SIZE else 0)} query tiles")
    print()
    
    # Simulate what happens for each query tile
    for query_tile_index in range((N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE):
        start_q = query_tile_index * Q_TILE_SIZE
        
        # This is what tl.arange(0, Q_TILE_SIZE) produces
        arange_result = list(range(Q_TILE_SIZE))
        
        # This is what start_q + tl.arange(0, Q_TILE_SIZE) produces
        q_indices = [start_q + i for i in arange_result]
        
        print(f"Query Tile {query_tile_index}:")
        print(f"  start_q = {query_tile_index} * {Q_TILE_SIZE} = {start_q}")
        print(f"  tl.arange(0, {Q_TILE_SIZE}) = {arange_result}")
        print(f"  q_indices = {start_q} + {arange_result} = {q_indices}")
        print(f"  → This tile processes queries at positions: {q_indices}")
        print()

def show_realistic_example():
    print("=" * 60)
    print("REALISTIC EXAMPLE WITH TYPICAL FLASH ATTENTION PARAMETERS")
    print("=" * 60)
    
    Q_TILE_SIZE = 128
    N_QUERIES = 512
    
    print(f"Scenario: N_QUERIES = {N_QUERIES}, Q_TILE_SIZE = {Q_TILE_SIZE}")
    print(f"Total query tiles needed: {(N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE}")
    print()
    
    # Show first few tiles
    for query_tile_index in range(min(4, (N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE)):
        start_q = query_tile_index * Q_TILE_SIZE
        
        # Show first and last few elements of q_indices
        q_indices_start = start_q
        q_indices_end = start_q + Q_TILE_SIZE - 1
        
        print(f"Query Tile {query_tile_index}:")
        print(f"  start_q = {start_q}")
        print(f"  q_indices = [{q_indices_start}, {q_indices_start + 1}, {q_indices_start + 2}, ..., {q_indices_end - 2}, {q_indices_end - 1}, {q_indices_end}]")
        print(f"  → Processes {Q_TILE_SIZE} queries from position {q_indices_start} to {q_indices_end}")
        print()

def show_causal_mask_example():
    print("=" * 60)
    print("HOW q_indices IS USED IN CAUSAL MASKING")
    print("=" * 60)
    
    # Small example for visualization
    Q_TILE_SIZE = 3
    K_TILE_SIZE = 4
    
    # Simulate query tile 1 (processes queries 3, 4, 5)
    query_tile_index = 1
    start_q = query_tile_index * Q_TILE_SIZE  # start_q = 3
    q_indices = [start_q + i for i in range(Q_TILE_SIZE)]  # [3, 4, 5]
    
    # Simulate key tile 0 (processes keys 0, 1, 2, 3)
    start_k = 0
    k_indices = [start_k + i for i in range(K_TILE_SIZE)]  # [0, 1, 2, 3]
    
    print(f"Query Tile {query_tile_index}: q_indices = {q_indices}")
    print(f"Key Tile 0: k_indices = {k_indices}")
    print()
    
    # Show the causal mask computation
    print("Causal mask computation: q_indices[:, None] >= k_indices[None, :]")
    print("This creates a 2D mask where mask[i,j] = (q_indices[i] >= k_indices[j])")
    print()
    
    # Simulate the mask
    print("     k_idx:  ", end="")
    for k in k_indices:
        print(f"{k:>5}", end="")
    print()
    
    for i, q in enumerate(q_indices):
        print(f"q_idx {q}: ", end="")
        for j, k in enumerate(k_indices):
            mask_val = q >= k
            print(f"{str(mask_val):>5}", end="")
        print()
    
    print()
    print("True  = attention allowed (q_pos >= k_pos)")
    print("False = attention blocked (q_pos < k_pos, violates causality)")

if __name__ == "__main__":
    show_q_indices_examples()
    print()
    show_realistic_example()
    print()
    show_causal_mask_example()