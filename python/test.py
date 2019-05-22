import rpy2.robjects as robjects
import numpy as np
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
import e_divisive as pye
import e_agglomerative

#Prints a comparison of all the attributes of two e_agglo calls.
#pydict must be the returned e_agglomerative Python dictionary.
#renv must be the returned robjects version of e.agglo
#output is whether each attribute is printed as well as the comparison.
#Returns None.
#The attributes are as follows: merged, fit, progression, estimates, cluster
#Note: Pydict indexing is 0 while Renv is 1, thus we add 1 to the index
#       sensitive attributes. This causes problems with merged, thus we ignore.
def agglo_compare(pydict, renv, output = False):
    print()
    print("Agglomerative Comparison.")
    print()

    print("Fitness: ")
    pyfit = np.array(pydict["fit"])
    rfit = np.array(renv.rx2("fit"))
    if(output):
        print("Python: " + str(pyfit))
        print("R: " + str(rfit))
    print("Comparison: " + \
        str(np.linalg.norm(pyfit - rfit)))
    print()

    print("Progression: ")
    pyprog = np.array(pydict["progression"])
    rprog = np.array(renv.rx2("progression"))
    if(output):
        print("Python: " + str(pyprog))
        print("R: " + str(rprog))
    pyprog[np.isnan(pyprog)] = -1
    rprog[np.isnan(rprog)] = 0 
    print("Comparison: " + \
        str(np.linalg.norm((pyprog + 1) - rprog)))
    print()

    print("Estimates: ")
    pyest = np.array(pydict["estimates"])
    rest = np.array(renv.rx2("estimates"))
    if(output):
        print("Python: " + str(pyest))
        print("R: " + str(rest))
    print("Comparison: " + \
        str(np.linalg.norm((pyest+1) - rest)))
    print()

    print("Cluster: ")
    pyclust = np.array(pydict["cluster"])
    rclust = np.array(renv.rx2("cluster"))
    if(output):
        print("Python: " + str(pyclust))
        print("R: " + str(rclust))
    print("Comparison: " + \
        str(np.linalg.norm((pyclust+1) - rclust)))
    return None

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
packnames = ['ecp', 'MASS', 'combinat', 'mvtnorm']
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
print(names_to_install)
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

data = robjects.r("""
    library(ecp)
    set.seed(100)
    x1 = matrix(c(rnorm(100),rnorm(100,3),rnorm(100,0,2)))
    x2 = rbind(MASS::mvrnorm(100,c(0,0),diag(2)), MASS::mvrnorm(100,c(2,2),diag(2)))
    y1 = e.divisive(X=x1,sig.lvl=0.05,R=199,k=NULL,min.size=30,alpha=1)
    y2 = e.divisive(X=x2,sig.lvl=0.05,R=499,k=NULL,min.size=30,alpha=1)
    y1_no_perm = e.divisive(X = x1, k = 2, min.size = 30, alpha = 1)
    y2_no_perm = e.divisive(X = x2, k = 1, min.size = 30, alpha = 1)
    
    mem = rep(c(1,2,3,4),times=c(10,10,10,10))
    x_agg = as.matrix(c(rnorm(10,0,1),rnorm(20,2,1),rnorm(10,-1,1)))
    y_agg = e.agglo(X=x_agg,member=mem,alpha=1,penalty=function(cp,Xts) 0)
    y_aggfit = y_agg$fit
    y_aggcluster = y_agg$cluster

    library(mvtnorm); library(combinat); library(MASS)
    set.seed(2013)
    lambda = 1500 #overall arrival rate per unit time
    muA = c(-7,-7) ; muB = c(0,0) ; muC = c(5.5,0)
    covA = 25*diag(2)
    covB = matrix(c(9,0,0,1),2)
    covC = matrix(c(9,.9,.9,9),2)
    time.interval = matrix(c(0,1,3,4.5,1,3,4.5,7),4,2)
    #mixing coefficents
    mixing.coef = rbind(c(1/3,1/3,1/3),c(.2,.5,.3), c(.35,.3,.35), 
        c(.2,.3,.5))
    stppData = NULL
    for(i in 1:4){
        count = rpois(1, lambda* diff(time.interval[i,]))
        Z = rmultz2(n = count, p = mixing.coef[i,])
        S = rbind(rmvnorm(Z[1],muA,covA), rmvnorm(Z[2],muB,covB),
            rmvnorm(Z[3],muC,covC))
        X = cbind(rep(i,count), runif(n = count, time.interval[i,1],
            time.interval[i,2]), S)
        stppData = rbind(stppData, X[order(X[,2]),])
    }
    member = as.numeric(cut(stppData[,2], breaks = seq(0,7,by=1/12)))
    y_agg2 = e.agglo(X=stppData[,3:4],member=member,alpha=1,
	penalty=function(cp,Xts) 0)
    x_agg2 = stppData[,3:4]
    x_mem2 = member
    
    """)

py_agglo_X1 = np.array(robjects.r["x_agg"])
py_agglo_mem1 = np.array(robjects.r["mem"])
py_agglo_Y1 = \
    e_agglomerative.e_agglo(X = py_agglo_X1, \
                            member=py_agglo_mem1, \
                            alpha = 1, \
                            penalty = (lambda cp: 0))

rY1 = robjects.r["y_agg"]
agglo_compare(py_agglo_Y1, rY1, True)

# Warning: In the current implementation this takes ~20 minutes to run.
# py_agglo_X2 = np.array(robjects.r["x_agg2"])
# py_agglo_mem2 = np.array(robjects.r["x_mem2"])
# py_agglo_Y2 = \
#     e_agglomerative.e_agglo(X = py_agglo_X2, \
#                             member = py_agglo_mem2, \
#                             alpha = 1, \
#                             penalty =  (lambda cp : 0))
# rY2 = robjects.r["y_agg2"]
# agglo_compare(py_agglo_Y2, rY2, False)

pyX1 = np.array(robjects.r["x1"])
pyY1_no_perm = pye.e_divisive(X = pyX1, k = 2, min_size = 30, alpha = 1)
print(pyY1_no_perm)
rY1_no_perm = robjects.r["y1_no_perm"]
print(rY1_no_perm)

pyX2 = np.array(robjects.r["x2"])
pyY2_no_perm = pye.e_divisive(X = pyX2, k = 1, min_size = 30, alpha = 1)
print(pyY2_no_perm)
rY2_no_perm = robjects.r["y2_no_perm"]
print(rY2_no_perm)

pyY1 = pye.e_divisive(X = pyX1, sig_lvl = 0.05, R = 199, k = None, min_size = 30, alpha = 1)
print(pyY1)
rY1 = robjects.r["y1"]
print(rY1)

pyY2 = pye.e_divisive(X = pyX2, sig_lvl = 0.05, R = 499, k = None, min_size = 30, alpha = 1)
print(pyY2)
rY2 = robjects.r["y2"]
print(rY2)
