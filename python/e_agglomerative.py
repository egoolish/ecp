import numpy as np

# The only function that the user has to interact with. member is the initial membership
# array, X is the observed data set, alpha is the exponent used on the Euclidean distance.
def e_agglo(X, member = None, alpha = 1, penalty = lambda cps : 0):
    if member is None:
        member = np.arange(1, (X.shape)[0])
    if (alpha <= 0) or (alpha > 2):
        raise ValueError("The alpha argument must be in (0, 2].")
    if(not callable(penalty)):
        raise ValueError("The penalty argument must be a function.")
    
    ret = process_data(member, X, alpha)
    n = ret["n"]

    #Find which clusters optimize the GOF and then update distances
    for k in range(n, 2*n-1):

        #find which clusters to merge
        best = find_closest(k, ret) 

        #update GOF statistic
        ret["fit"].append(best[2])

        #Update information after merge
        ret = update_distance(best[0], best[1], k, ret)
    
    # Penalize the GOF statistic
    cps = #TODO: #apply(ret$progression,1,function(x){x[!is.na(x)]})
    ret["fit"] = #TODO: #ret$fit + sapply(cps,penalty)

    # Get the set of change points for the "best" clustering
    ret["estimates"] = np.argmax(ret["fit"])
    ret["estimates"] = np.sort(ret["progression"][ret["estimates"]])

    # Remove change point N + 1 if a cyclic merger was performed
    if(ret["estimates"][0] != 1):
        ret["estimates"] = ret["estimates"][:-1]
    
    # Create final membership vector
    tmp = ret["estimates"]
    if (tmp[0] == 0):
        ret["cluster"] = np.repeat(np.arange(0,len(np.diff(tmp))), np.diff(tmp))
    else:
        tmp = np.insert(tmp, 0, 0)
        ret["cluster"] = np.repeat(np.arange(0,len(np.diff(tmp))), np.diff(tmp))
        k = X.shape[0] - np.size(ret["cluster"])
        np.append(ret["cluster"], np.zeros(k))
    
    # Remove unnecessary output info
    rem = ["n", "left", "open", "d", "right", "lm", "sizes"]
    for each in rem:
        ret.pop(each, None)
    return ret

#Initialize all the necessary components of the dictionary.
def process_data(member, X, alpha):
    ret = {} #the dictionary with the necessary information
    u = np.unique(member)
    ret["n"] = u.size #number of clusters
    n = ret["n"]
    fit = 0

    #relabel clusters
    for i in range(n): 
        member[member == u[i]] = i

    #Check that segments conssit only of adjacent observations
    if(not np.all(member[:-1] <= member[1:])):
        raise ValueError("Segments must be contiguous.")
    
    ret["sizes"] = np.zeros(2*n)

    #Left and Right neighbors of a cluster
    ret["right"] = np.zeros(2*n - 1)
    ret["left"] = np.zeros(2*n - 1)

    #True means that a cluster has not been merged
    ret["open"] = np.ones(2*n - 1, dtype = bool)

    #Calculate initial cluster sizes
    for i in range(n):
        ret["sizes"][i] = np.sum(member==i)

    #Set up left and right neighbors
    for i in range(1, n-1):
        ret["left"][i] = i - 1
        ret["right"][i] = i + 1

    #Special case for clusters 1 and N to allow for cyclic merging
    ret["left"][n-1] = n - 2
    ret["right"][n] = 0
    ret["left"][0] = n - 1
    ret["right"][0] = 1

    #Matrix to say which clusters were merged at each step
    ret["merged"] = np.zeros((n-1, 2))

    #Array of within distances
    within = np.zeros(n)
    for i in range(n): #TODO: np.cov?
        within[i] = get_within(alpha, np.full((ret["sizes"][i], X.shape[1]), X[member == i]))
    
    #Make distance matrix
    ret["d"] = np.full((2*n, 2*n), 0)
    for i in range(n):
        for j in range(i, n):
            if (j != i): #TODO: scipy.stats.energy_distance?
                gb = get_between(alpha,
                    np.full((ret["sizes"][i], X.shape[1]), X[member==i]),
                    np.full((ret["sizes"][j], X.shape[1]), X[member==j])) - within[i] - within[j]
                ret["d"][i, j] = gb
                ret["d"][j, i] = gb
    
    #Set initial GOF value
    for i in range(n):
        fit = fit + ret["d"][i, ret["left"][i]] + ret["d"][i, ret["right"][i]]
    ret["fit"] = fit

    #Create matrix for change point progressoin
    ret["progression"] = np.full((n, n+1), np.nan)
    ret["progression"][0, 0] = 1
    for i in range(1, n + 1):
        ret["progression"][0, i] = ret["progression"][0, i - 1] + ret["sizes"][i-1]
    
    #Vector to specify the starting point of a cluster
    ret["lm"] = np.arange(n)
    return ret

# Determine which clusters will be merged. Returns a tuple of length 3: the first
# element is the left cluster, the second is the right, and the third is the newest
# GOF value.
def find_closest(k, ret):
    n = ret["n"]
    best = float('-inf')
    triple = (0, 0, 0)
    
    #iterate to see how the GOF value changes
    for i in range(k):
        if (ret["open"[i]]):

            #Get updated GOF value
            x = gof_update(i, ret)

            #Better value found so update
            if (x > best):
                best = x
                triple = (i, ret["right"][i], x)
    if (not triple[0] and not triple[1] and k !=(2*n - 2)):
        raise ValueError("There was a problem finding which clusters to merge.")
    return triple

# Function to calculate the new GOF value. The i argument is assumed to be the 
# left cluster.
def gof_update(i, ret):
    fit = ret["fit"][-1]
    j = ret["right"][i]

    #Get new left and right clusters
    rr = ret["right"][j]
    ll = ret["left"][i]

    #Remove unneded values in the GOF
    fit = fit - 2*(ret["d"][i, j]) + ret["d"][i, ll] + ret["d"][j, rr]

    #Get cluster sizes
    n1 = ret["sizes"][i]
    n2 = ret["sizes"][j]

    #Add distance to new left cluster
    n3 = ret["sizes"][ll]
    k = ((n1 + n3)*ret["d"][i, ll] + (n2 + n3)*ret["d"][j, ll] - n3 * ret["d"][i, j])/(n1 + n2 + n3)
    fit = fit + 2 *k

    #Add distance to new right cluster
    n3 = ret["sizes"][rr]
    k = ((n1+n3)*ret["d"][i, rr] + (n2+n3)*ret["d"][j, rr] - n3*ret["d"][i, j])/(n1+n2+n3)
    fit = fit + 2*k

    return fit

# Function to update the distance from the new cluster to the other clusters.
# i is assumed to be the left cluster. Also #updates any other necessary 
# information in [ret].
def update_distance(i, j, k, ret):
    n = ret["n"]

    # Say which clusters were merged
    if (i <= n):
        ret["merged"][k - n, 0] = -i
    else:
        ret["merged"][k - n + 1, 0] = i - n
    if (j <= n):
        ret["merged"][k - n + 1, 1] = -j
    else:
        ret["merged"][k - n + 1, 1] = j - n
    
    # Update left and right neighbors
    ll = ret["left"][i]
    rr = ret["right"][j]
    ret["left"][k + 1] = ll
    ret["right"][k + 1] = rr
    ret["right"][ll] = k + 1
    ret["left"][rr] = k + 1

    # Update information on which clusters have been merged
    ret["open"][i] = False
    ret["open"][j] = False

    #Assign size to newly created cluster
    n1 = ret["sizes"][i]
    n2 = ret["sizes"][j]
    ret["sizes"][k + 1] = n1+ n2

    #Update set of change points
    ret["progression"][k-n+2] = ret["progression"][k-n+1]
    ret["progression"][k-n+2, ret["lm"][j]] = np.nan
    ret["lm"][k+1] = ret["lm"][i]

    #Update distances
    for kk in range(k):
        if (ret["open"][kk]):
            n3 = ret["sizes"][kk]
            nn = n1+n2+n3
            hold = ((nn -n2)*ret["d"][i, kk] + (nn-n1)*ret["d"][j, k] - n3*ret["d"][i, j])/nn
            ret["d"][k+1, kk] = hold
            ret["d"][kk, k+1] = hold

    return ret