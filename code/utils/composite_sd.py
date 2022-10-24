import math

__all__ = ['iterable', 'avg', 'sample_SD', 'composite_SD']


def iterable(obj):
    """True iff obj is iterable: a list, tuple, or string."""
    return hasattr(obj, '__contains__')


def avg(samples):
    if len(samples) >= 1:
        return sum(samples) / len(samples)
    return float('nan')


def sample_SD(samples):
    """input is an array of samples; result is the standard deviation"""
    mean = avg(samples)
    sum_of_squared_deviations = 0
    sd = 0
    if len(samples) >= 2:
        for datum in samples:
            sum_of_squared_deviations += ((datum - mean) * (datum - mean))
        sd = math.sqrt(sum_of_squared_deviations / (len(samples)-1))
    return sd


def composite_SD(means, SDs, ncounts):
    """Calculate combined standard deviation via ANOVA (ANalysis Of VAriance)
       See:  http://www.burtonsys.com/climate/composite_standard_deviations.html
       Inputs are:
         means, the array of group means
         SDs, the array of group standard deviations
         ncounts, number of samples in each group (can be scalar
                  if all groups have same number of samples)
       Result is the overall standard deviation.
    """
    G = len(means)  # number of groups
    if G != len(SDs):
        raise Exception('inconsistent list lengths')
    if not iterable(ncounts):
        ncounts = [ncounts] * G  # convert scalar ncounts to array
    elif G != len(ncounts):
        raise Exception('wrong ncounts list length')

    # calculate total number of samples, N, and grand mean, GM
    N = sum(ncounts)  # total number of samples
    if N <= 1:
        raise Exception("Warning: only " + str(N) + " samples, SD is incalculable")
    GM = 0.0
    for i in range(G):
        GM += means[i] * ncounts[i]
    GM /= N  # grand mean

    # calculate Error Sum of Squares
    ESS = 0.0
    for i in range(G):
        ESS += ((SDs[i])**2) * (ncounts[i] - 1)

    # calculate Total Group Sum of Squares
    TGSS = 0.0
    for i in range(G):
        TGSS += ((means[i]-GM)**2) * ncounts[i]

    # calculate standard deviation as square root of grand variance
    result = math.sqrt((ESS+TGSS)/(N-1))
    return result


# samples = range(10)
# print('avg=', avg(samples))
# sd = sample_SD(samples)
# print('sd=', sd)
# pt1 = [0,1,2]
# pt2 = [3,4]
# pt3 = [5,6,7,8,9]
# means = [avg(pt1), avg(pt2), avg(pt3)]
# SDs = [sample_SD(pt1), sample_SD(pt2), sample_SD(pt3)]
# ncounts = [len(pt1), len(pt2), len(pt3)]
# sd2 = composite_SD(means, SDs, ncounts)
# print('sd2=', sd2)
#
# samples = range(9)
# print('avg=', avg(samples))
# sd = sample_SD(samples)
# print('sd=', sd)
# pt1 = [0,1,2]
# pt2 = [3,4,5]
# pt3 = [6,7,8]
# means = [avg(pt1), avg(pt2), avg(pt3)]
# SDs = [sample_SD(pt1), sample_SD(pt2), sample_SD(pt3)]
# ncounts = 3
# sd2 = composite_SD(means, SDs, ncounts)
# print('sd2=', sd2)
