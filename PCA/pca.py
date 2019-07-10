import numpy as np

def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis = 0)
    newData = dataMat - meanVal
    return newData, meanVal
def pca(dataMat,n):
    newMat, meanValue = zeroMean(dataMat)
    covMat = np.cov(newMat, rowvar = 0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice =np.argsort(eigVals)
    n_eigValIndice =  eigValIndice[-1: -(n+1): -1]
    n_eigVect = eigVects[:, n_eigValIndice]
    lowDDataMat=newMat * n_eigVect               #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T) + meanValue  #重构数据
    return lowDDataMat, reconMat 

if __name__ == "__main__":
    randomMat = np.random.rand(5,18)
    print(randomMat)
    lowDDataMat, reconMat = pca(randomMat, 3)
    print(lowDDataMat)
    print(reconMat)
