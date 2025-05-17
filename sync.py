import torch.distributed as dist
import os
import sys

def init_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group(backend, rank=rank, world_size=size)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    size = int(os.environ['SLURM_NTASKS'])
    init_process(rank, size)
