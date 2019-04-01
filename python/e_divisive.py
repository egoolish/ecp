import numpy as np
from scipy import spatial

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
    n = np.size(X, 1)

    if (k is None):
        k = n
    else:
        #No need to perform the permutation test
        R = 0
    
    ret = {"k_hat": 1}
    pvals = []
    permutations = []
    changes = [1, n+1] #List of change-points
    energy = np.zeros((n, 2)) #Matrix used to avoid recalculating statistics
    D = np.power(spatial.distance.pdist(X), alpha)
    con = -1

    while (k > 0):
        tmp = e_split(changes, D, min_size, False, energy) #find location for change point
        #i = tmp['start']
        #j = tmp['end']
        e_stat = tmp["best"]
        tmp = tmp["changes"]

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

        changes = tmp #update set of change points
        ret["k_hat"] = ret["k_hat"]+1 #update number of clusters
        k = k-1
    
    tmp = changes.sort()
    ret["order_found"] = changes
    ret["estimates"] = tmp
    ret["considered_last"] = con
    ret["p_values"] = pvals
    ret["permutations"] = permutations
    ret["cluster"] = np.repeat(np.arange(1,len(np.diff(tmp))), np.diff(tmp))
    return ret

def e_split(changes, D, min_size, for_sim = False, energy = None):
    splits = changes.sort()
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
        changes.append(best[0])
        return {"start": ii, "end": jj, "changes": changes, "best": best[1]}
    else:
        if(energy is None):
            raise ValueError("Must specify one of: for_sim, energy")
        
        #iterate over intervals
        for i in range(1, len(splits)):
            if(energy[splits[i-1]][0]):
                tmp = energy[splits[i-1]]
            else:
                tmp = splitPoint(splits[i-1], splits[i]-1, D, min_size)
                energy[splits[i-1]][0] = tmp[0]
                energy[splits[i-1]][1] = tmp[1]

            #tmp[1] is the "energy released" when the cluster was split
            if(tmp[1] > best[1]):
                ii = splits[i-1]
                jj = splits[i] - 1
                best = tmp
            
        changes.append(best[0]) #update the list of change points
        energy[ii][0] = 0 #update matrix to account for newly proposed change point
        energy[ii][1] = 0 #update matrix to account for newly proposed change point
        return {"start": ii, "end": jj, "changes": changes, "best": best[1]}

def splitPoint(start, end, D, min_size):

    #interval too small to split
    if(end-start+1 < 2*min_size):
        return [-1, float("-inf")]
    
    #use needed part of distance matrix
    D = D[start:end][start:end]

    raise Exception("currently unimplemented: uses C code")

def sig_test(D, R, changes, min_size, obs):
    #No permutations, so return a p-value of 0
    if(R == 0): 
        return 0
    over = 0
    for f in range(R):
        D1 = perm_cluster(D, changes) #permute within cluster
        tmp = e_split(changes, D1, min_size, True)
        if(tmp['best'] >= obs):
            over = over + 1
    
    pval = (1 + over)/(f + 1)
    return [pval, f]

def perm_cluster(D, points):
    points = points.sort()
    K = len(points) - 1 #number of cluster
    for i in range(K):
        u = np.random.shuffle(np.arange(points[i], points[i+1]))
        D[points[i]:(points[i+1]-1)][points[i]:(points[i+1]-1)] = D[u][u]
    return D