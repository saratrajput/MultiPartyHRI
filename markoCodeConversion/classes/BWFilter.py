from scipy.signal import butter, lfilter, filtfilt

class BWFilter():

    def __init__(self):
        pass
        # self.fs = fs
        # self.th = th

    def initFilter(self, fs, th, order):
        nyq = 0.5 * fs
        low = th / nyq
        b, a = butter(order, low)
        self.a = a
        self.b = b

    def getFilterVals(self):
        return [self.a, self.b]

    def filterData(self, data):
        y = lfilter(self.b, self.a, data)
        return y