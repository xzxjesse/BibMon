import numpy as np

###############################################################################

def detecOutlier(data, lim, count = False, count_limit = 1):
    """
    Detects outliers in the given data using a specified limit.

    Parameters
    ----------
    data: array-like
        The input data to be analyzed.
    lim: float
        The limit value used to detect outliers.
    count: bool, optional
        If True, counts the number of outliers 
        exceeding the limit. Default is False.
    count_limit: int, optional
        The maximum number of outliers allowed. 
        Default is 1.

    Returns
    ----------
    alarm: ndarray or int
        If count is False, returns an array indicating 
        the outliers (0 for values below or equal to lim,
        1 for values above lim).
        If count is True, returns the number of outliers 
        exceeding the limit.
    """

    if np.isnan(data).any():
        data = np.nan_to_num(data)

    if count == False:
        alarm = np.copy(data)
        alarm = np.where(alarm<=lim, 0, alarm)
        alarm = np.where(alarm>lim, 1, alarm)
        return alarm
    else:
        alarm = 0
        local_count = np.count_nonzero(data > lim)
        if local_count > count_limit:
            alarm = +1
        return alarm
        

def detect_drift_bias(data, window=10, threshold=2.0):
    """
    Detects drift or bias in a time series using a sliding window approach.

    Parameters
    ----------
    data : array-like
        Input time series data.
    window : int
        Size of the window to check for drift/bias.
    threshold : float
        Minimum absolute difference between the mean of the first and second half of the window to trigger the alarm.

    Returns
    -------
    alarm : int
        1 if drift/bias is detected, 0 otherwise.
    """
    import numpy as np
    data = np.asarray(data)
    if len(data) < window:
        return 0
    for i in range(len(data) - window + 1):
        win = data[i:i+window]
        first_half = win[:window//2]
        second_half = win[window//2:]
        diff = np.abs(np.mean(second_half) - np.mean(first_half))
        if diff > threshold:
            return 1
    return 0
        