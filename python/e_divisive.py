import numpy as np
from scipy.spatial.distance import pdist, squareform
import copy

def e_divisive(X, sig_lvl = 0.05, R = 199, k = None, min_size = 30, alpha = 1):
    """Test: documentation goes here."""

    #Checks for invalid arguments.
    if (R < 0) and (k is None):
        raise ValueError("R must be a nonnegative integer.")
    if (sig_lvl <= 0 or sig_lvl >= 1) and (k is None):
        raise ValueError("sig_lvl must be a positive real number between 0 and 1.")
    if (min_size < 2):
        raise ValueError("min_size must be an integer greater than 1.")
    if (alpha > 2 or alpha <= 0):
        raise ValueError("alpha must be in (0, 2].")
    
    #Length of time series
    n = np.size(X)

    if (k is None):
        k = n
    else:
        #No need to perform the permutation test
        R = 0
    
    ret = {"k_hat": 1}
    pvals = []
    permutations = []
    changes = [0, n] #List of change-points
    energy = np.zeros((n, 2)) #Matrix used to avoid recalculating statistics
    D = np.power(squareform(pdist(X)), alpha)
    con = -1

    while (k > 0):
        tmp_data = e_split(changes, D, min_size, False, energy) #find location for change point
        #print(tmp_data)
        e_stat = tmp_data["best"]
        tmp = copy.deepcopy(tmp_data["changes"])

        #newest change-point
        con = tmp[-1]

        #Not able to meet minimum size constraint
        if(con == -1):
            break 
        #run permutation test
        result = sig_test(D, R, changes, min_size, e_stat)
        pval = result[0] #approximate p-value
        permutations.append(result[1]) #number of permutations performed
        pvals.append(pval) #list of computed p-values

        #change point not significant
        if(pval > sig_lvl):
            break
        changes = copy.deepcopy(tmp) #update set of change points
        ret["k_hat"] = ret["k_hat"]+1 #update number of clusters
        k = k-1

    tmp = copy.deepcopy(changes)
    tmp.sort()
    ret["order_found"] = changes
    ret["estimates"] = tmp
    ret["considered_last"] = con
    ret["p_values"] = pvals
    ret["permutations"] = permutations
    ret["cluster"] = np.repeat(np.arange(0,len(np.diff(tmp))), np.diff(tmp))
    return ret

def e_split(changes, D, min_size, for_sim = False, energy = None):
    splits = copy.deepcopy(changes)
    splits.sort()
    best = [-1, float('-inf')]
    ii = -1
    jj = -1

    #If procedure is being used for significance test
    if(for_sim):

        #Iterate over intervals
        for i in range(1, len(splits)):
            tmp = splitPoint(splits[i-1], splits[i] - 1, D, min_size)
           
            #tmp[1] is the "energy released" when the cluster was split
            if(tmp[1]>best[1]):
                ii = splits[i-1]
                jj = splits[i] - 1
                best = tmp #update best split point found so far
        
        #update the list of changepoints
        changes.append(int(best[0]))
        return {"start": ii, "end": jj, "changes": changes, "best": best[1]}
    else:
        if(energy is None):
            raise ValueError("Must specify one of: for_sim, energy")
        
        #iterate over intervals
        for i in range(1, len(splits)):
            if(energy[splits[i-1],0]):
                tmp = energy[splits[i-1]]
            else:
                tmp = splitPoint(splits[i-1], splits[i]-1, D, min_size)
                energy[splits[i-1], 0] = tmp[0]
                energy[splits[i-1], 1] = tmp[1]

            #tmp[1] is the "energy released" when the cluster was split
            if(tmp[1] > best[1]):
                ii = splits[i-1]
                jj = splits[i] - 1
                best = tmp
            
        changes.append(int(best[0])) #update the list of change points
        energy[ii, 0] = 0 #update matrix to account for newly proposed change point
        energy[ii, 1] = 0 #update matrix to account for newly proposed change point
        return {"start": ii, "end": jj, "changes": changes, "best": best[1]}

def splitPoint(start, end, D, min_size):
    #interval too small to split
    if(end-start+1 < 2*min_size):
        return [-1, float("-inf")]
    
    #use needed part of distance matrix
    D = D[start:end, start:end]
    return splitPointpseudoC(start, end, D, min_size)

def splitPointpseudoC(s, e, D, min_size):
    """ This function used to be written in C++. However, it used SEXP to return
        a numeric vector data type, which is incompatible with Python. As such, 
        the function is temporarily rewritten in Python, but could be made faster
        by replacing this with a Python to C++ call.
    """
    best = [-1.0, float('-inf')]
    e = e - s + 1
    tau1 = min_size
    tau2 = min_size << 1
    cut1 = D[0:tau1-1, 0:tau1-1]
    cut2 = D[tau1:tau2-1, tau1:tau2-1]
    cut3 = D[0:tau1-1, tau1:tau2-1]
    A = np.sum(cut1)/2
    B1 = np.sum(cut2)/2
    AB1 = np.sum(cut3)
    tmp = 2*AB1/((tau2-tau1)*(tau1)) - 2*B1/((tau2-tau1-1)*(tau2-tau1)) - 2*A/((tau1-1)*(tau1))
    tmp *= (tau1*(tau2-tau1)/tau2)
    if(tmp > best[1]):
        best[0] = tau1+s
        best[1] = tmp
    
    tau2+=1
    B = np.full(e+1, B1)
    AB = np.full(e+1, AB1)

    while(tau2 < e):
        B[tau2] = B[tau2-1] + np.sum(D[tau2-1, tau1:tau2-2])
        AB[tau2] = AB[tau2-1] + np.sum(D[tau2-1, 0:tau1-1])
        tmp = 2*AB[tau2]/((tau2-tau1)*(tau1))-2*B[tau2]/((tau2-tau1-1)*(tau2-tau1))-2*A/((tau1)*(tau1-1))
        tmp *= (tau1*(tau2-tau1)/tau2)
        if(tmp > best[1]):
            best[0] = tau1+s
            best[1] = tmp
        tau2 += 1

    tau1+=1
    tau2 = tau1+min_size
    while(tau2 < e):
        addA = np.sum(D[tau1-1, 0:tau1-2])
        A += addA
        addB = np.sum(D[tau1-1, tau1:tau2-2])
        while(tau2 < e):
            addB += D[tau1-1, tau2-1]
            B[tau2] -=addB
            AB[tau2]+=(addB-addA)
            tmp = 2*AB[tau2]/((tau2-tau1)*(tau1))-2*B[tau2]/((tau2-tau1-1)*(tau2-tau1)) - 2*A/((tau1-1)*(tau1))
            tmp *= (tau1*(tau2-tau1)/tau2)
            if(tmp > best[1]):
                best[0] = tau1+s
                best[1] = tmp
            tau2+=1

        tau1 += 1
        tau2 = tau1+min_size
    return best       


def sig_test(D, R, changes, min_size, obs):
    #No permutations, so return a p-value of 0
    if(R == 0): 
        return [0, -1]
    over = 0
    for f in range(R):
        D1 = perm_cluster(D, changes) #permute within cluster
        tmp = e_split(changes, D1, min_size, True)
        if(tmp['best'] >= obs):
            over = over + 1
    f+=1
    pval = (1 + over)/(f + 1)
    return [pval, f]

def perm_cluster(D, points):
    points.sort()
    K = len(points) - 1 #number of clusters
    for i in range(K):
        u = np.arange(points[i], points[i+1]-1)
        np.random.shuffle(u)
        D[points[i]:(points[i+1]-1), :] = D[u, :]
        D[:, points[i]:(points[i+1]-1)] = D[:, u]
    return D