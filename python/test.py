import rpy2.robjects as robjects
import numpy as np
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
import e_divisive as pye

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
packnames = ['ecp', 'MASS']
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
print(names_to_install)
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

#y = e.divisive(X=x1,sig.lvl=0.05,R=199,k=NULL,min.size=30,alpha=1)
data = robjects.r("""
    library(ecp)
    set.seed(100)
    x1 = matrix(c(rnorm(100),rnorm(100,3),rnorm(100,0,2)))
    x2 = rbind(MASS::mvrnorm(100,c(0,0),diag(2)), MASS::mvrnorm(100,c(2,2),diag(2)))
    y = e.divisive(X = x1, k = 2, min.size = 30, alpha = 1)
    y2 = e.divisive(X = x2, k = 1, min.size = 30, alpha = 1)
    """)
pyX1 = np.array(robjects.r["x1"])
pyY1 = pye.e_divisive(X = pyX1, k = 2, min_size = 30, alpha = 1)
print(pyY1)
rY = robjects.r["y"]
print(rY)

# pyX2 = np.array(robjects.r["x2"])
# print(pyX2.shape)
# pyY2 = pye.e_divisive(X = pyX2, k = 1, min_size = 30, alpha = 1)
# print(pyY2)
# rY2 = robjects.r["y2"]
# print(rY2)