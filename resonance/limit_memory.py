"""Betrays the Zen of Python by running implicitly. It's more out of convenience than anything else."""

import resource

def limit_memory(maxsize: int = 2e10):
    """Sets a hard upper limit on memory usage.
    
    Borrowed from: https://bit.ly/3LUMk5B

    Args:
        maxsize (int): Optional. Maximum memory [bytes].
            Defaults to 2E10 (20 Gb).

    Raises:
        MemoryError if memory usage is exceeded.

    """

    maxsize = int(maxsize)

    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

limit_memory()