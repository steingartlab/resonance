import numpy as np

def smooth(x, window_len: int = 11, window='flat'):
    """Adopted DS."""
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len, 'd')
    else:
        w=eval(f"np.{window}(window_len)")

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    y = list(y)

    for _ in range(0, int(window_len)):
        y.pop(0)

    return np.array(y)

def significant_figures(x: float, n: int = 3):
    """Rounds input x to n significant figures.

    Adapted from the depths of StackOverflow.
    
    Args:
        x (float): The rounded-number-to-be
        n (int): Optional. Desired number of significant figures. Defaults to 3.
    
    Returns:
        x rounded to n significant figures.
    """

    return f'{float(f"{x:.{n}g}"):g}'