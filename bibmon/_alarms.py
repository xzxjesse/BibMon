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
        
def detect_nelson_rule1(data):
    """
    Detects Nelson Rule 1: one point above 3 standard deviations from the mean.

    Parameters
    ----------
    data : array-like
        Input time series data.

    Returns
    -------
    alarm : int
        1 if at least one point is above (mean + 3*std) or below (mean - 3*std), 0 otherwise.
    """
    import numpy as np
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data)
    if np.any(data > mean + 3*std) or np.any(data < mean - 3*std):
        return 1
    return 0
        
def detect_nelson_rule2(data):
    """
    Detects Nelson Rule 2: nine consecutive points on the same side of the mean.

    Parameters
    ----------
    data : array-like
        Input time series data.

    Returns
    -------
    alarm : int
        1 if nine or more consecutive points are above or below the mean, 0 otherwise.
    """
    import numpy as np
    data = np.asarray(data)
    mean = np.mean(data)
    above = data > mean
    below = data < mean
    # Check for 9 consecutive Trues in above or below
    def has_n_consecutive(arr, n):
        count = 0
        for val in arr:
            if val:
                count += 1
                if count >= n:
                    return True
            else:
                count = 0
        return False
    if has_n_consecutive(above, 9) or has_n_consecutive(below, 9):
        return 1
    return 0
        
def detect_nelson_rule3(data):
    """
    Detects Nelson Rule 3: six consecutive points all increasing or all decreasing.

    Parameters
    ----------
    data : array-like
        Input time series data.

    Returns
    -------
    alarm : int
        1 if six or more consecutive points are strictly increasing or strictly decreasing, 0 otherwise.
    """
    import numpy as np
    data = np.asarray(data)
    n = 6
    for i in range(len(data) - n + 1):
        window = data[i:i+n]
        if np.all(np.diff(window) > 0):
            return 1
        if np.all(np.diff(window) < 0):
            return 1
    return 0
        
def detect_nelson_rule4(data):
    """
    Detects Nelson Rule 4: fourteen points in a row alternating up and down.

    Parameters
    ----------
    data : array-like
        Input time series data.

    Returns
    -------
    alarm : int
        1 if fourteen or more consecutive points alternate above and below the mean, 0 otherwise.
    """
    import numpy as np
    data = np.asarray(data)
    mean = np.mean(data)
    n = 14
    # Create a boolean array: True if above mean, False if below
    above = data > mean
    # Check for 14 consecutive alternations
    for i in range(len(above) - n + 1):
        window = above[i:i+n]
        # Alternating means: window[0] != window[1], window[1] != window[2], ...
        if all(window[j] != window[j+1] for j in range(n-1)):
            return 1
    return 0
