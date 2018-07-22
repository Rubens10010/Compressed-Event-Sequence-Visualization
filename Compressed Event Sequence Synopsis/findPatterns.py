from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance	# damerau_levenshtein_distance
from datasketch import MinHash, MinHashLSH	# LSH, Jaccard Similarity
import numpy as np	# event sequences
from math import *
from heap import Node, PriorityQueue	# priority queue max heap
import uuid	# unique id
import copy	# deepcopy

# Path of database files
dbPath = "../AgavueDB/"
outputPath = "../Outputs/"

# Contador de clusters para establecer el id de un nuevo cluster
counter_cluster_id = 0

# -- Cluster of event sequences object --
#@ <NumpyArray> p: Representative patron of the cluster
#@ <List<Int>>  g: List of event sequences Ids
class Cluster(object):
	def __init__(self, p, g):
		self.p = p
		self.g = g

# -- Tuple of clusters for priority queue on MinDL algorithm --
#@ <float> l: Description Length Reduction
#@ <int>  c_: Id of Cluster containing Ci and Cj elements
#@ <int>  ci: Id of Cluster i
#@ <int>  cj: Id of Cluster j
class PriorityQueue_tuple(object):
	def __init__(self, l, c_, ci, cj):
		self.l = l
		self.c_ = c_
		self.ci = ci
		self.cj = cj

###############	containers	#####################
# Maps Ids to clusters
cluster_list = dict()
# Maps PriorityQueue elements to PriorityQueue_Tuple Objects 
priorityQueue_dict = dict()
# Main Priority Queue for MinDL algorithm <heap>
q = PriorityQueue([])

# -- Adds a new cluster (c) to cluster_list --
#@<Cluster> c: new cluster object
#@<int>     returns: new cluster's id
def add_cluster(c):
	global counter_cluster_id
	cluster_list[counter_cluster_id] = c
	counter_cluster_id += 1
	return counter_cluster_id-1

# -- Removes the cluster identified by Id cKey from cluster_list --
#@<int> ckey: Id or key of cluster c
def remove_cluster(ckey):
	cluster_list.pop(ckey)	## check

# -- Utilitary Function: returns a unique identifier id on this session
#@<uuid> returns: unique identifier id
def getNextLabel():
	return uuid.uuid4()

# -- Queue function: enqueues a tuple formed by (L, c_, ci, cj) to Queue (q) --
#@<float> L: Total description length reduction
#@<int> c_: Id of Cluster containing ci and cj
#@<int> ci: Id of Cluster ci
#@<int> cj: Id of Cluster cj
def enqueue(L, c_, ci, cj):
	global priorityQueue_dict, q
	id = getNextLabel()
	x = Node(label=id, priority = L)
	q.insert(x)
	priorityQueue_dict[id] = PriorityQueue_tuple(L, c_, ci, cj)

# -- Deques the top element from queue q --
#@Top element from queue: Max element reducing Description Length of two clusters
#				if queue is empty or only has invalid entries it returns None
def dequeue():
	global priorityQueue_dict, q
	while(True):
		min = q.shift()
		if min is None:
			break
		# check if it is removed
		if min.label not in priorityQueue_dict:
			continue

		break
	return min

# -- Removes clusters ci and cj from all possible containers --
#@<int> ci: Id of Cluster ci
#@<int> cj: Id of Cluster cj
def remove_clusters(ci, cj):
	global priorityQueue_dict
	remove_cluster(ci)	# remove ci from cluster_list
	remove_cluster(cj)  # remove cj from cluster_list

	# Removing them from priorityQueue_dict makes them invisible for queue q
	for key, value in priorityQueue_dict.items():
		if value.ci == ci or value.ci == cj or value.cj == ci or value.cj == cj:
			priorityQueue_dict.pop(key)

# Dynamic programming implementation of Longest Common Subsequence (LCS)
#@<list() or NumpyArray> X: List of events (sequence)
#@<list() or NumpyArray> Y: List of events (sequence)
#@Returns: (np.array(lcs): x_ids, y_ids)
#		<NumpyArray> -> np.array(lcs): longest common subsequence
#		<list>		 -> x_ids: X events which dont belong to lcs
#		<list>		 -> y_ids: Y events which dont belong to lcs
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0 for x in xrange(n+1)] for x in xrange(m+1)]
 
    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] 
    for i in xrange(m+1):
        for j in xrange(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # Following code is used to print LCS
    index = L[m][n]
 
    # Create a character array to store the lcs string
    #lcs = [""] * (index+1)
    #lcs[index] = ""
 
    lcs = [""]*(index)

    x_ids = list()
    y_ids = list()

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    while i > 0 and j > 0:
        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i-1] == Y[j-1]:
            #x_ids.append(i-1)
            #y_ids.append(j-1)
            lcs[index-1] = X[i-1]
            i-=1
            j-=1
            index-=1
 
        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i-1][j] > L[i][j-1]:
            x_ids.append(i-1)
            i-=1
        else:
            y_ids.append(j-1)
            j-=1
    
    while i > 0:
        x_ids.append(i-1)
        i -= 1
    while j > 0:
        y_ids.append(j-1)
        j -= 1
 
    #print "LCS of {} and {} is {}".format(X,Y,lcs)
    return (np.array(lcs), x_ids, y_ids)

# Space optimized Python implementation of LCS problem
# Returns length of LCS for X and Y
# Pattern: X[0..m-1]
# Secuence: Y[0..n-1]
def lcs_distance(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0 for i in range(n+1)] for j in range(2)]

    # Binary index, used to index current row and
    # previous row.
    bi = bool
    for i in range(m):
        # Compute current binary index
        bi = i&1
        for j in range(n+1):
            if (i == 0 or j == 0):
                L[bi][j] = 0 
            elif (X[i] == Y[j - 1]):
                L[bi][j] = L[1 - bi][j - 1] + 1
            else:
                L[bi][j] = max(L[1 - bi][j], 
                               L[bi][j - 1])
    # Last filled entry contains length of LCS for X[0..n-1] and Y[0..m-1]
    return L[bi][n]


def compress_sequence(y):
	"""Compress signal y by omitting repeated values for y.

	Takes a signal y as an array-like. Returns the
	compressed signal y1 as the tuple (y1,counts) where y1 contains
	the first and last values of y, and values of y that are different
	from the preceeding or succeeding value, and 'counts' contains the
	corresponding count of missing elements of the each element of y
	in order of apperance

	"""
	#x, y = np.asarray(x), np.asarray(y)
	y = np.asarray(y)
	x = np.arange(len(y))
	keep = np.empty_like(x, dtype=bool)

	if len(x) > 1:
		keep[0] = keep[-1] = True
		keep[1:-1] = (y[1:-1] != y[:-2])
	elif len(x) == 1:
		return y, np.array([1])
	if y[-1] == y[-2]:
		keep[-1] = False
	x = x[keep]
	a = x[1:]
	b = x[:-1]

	repCount = a - b
	lastCount = len(y) - x[-1]
	repCount = np.concatenate((repCount,[lastCount]))
	# & (y[1:-1] != y[2:])
	return (y[keep], repCount)

# -- Candidate events of MinDL Merge algorithm are those who appear most frequently in (X - I) U (Y - I) --
#@<NumpyArray>     X-Y: Event Sequence, i.e ['1','3','4','2','4']
#@<List> x_ids - y_ids: List of Ids from events not in lcs(X,Y)
#@Returns: Concatenated NumpyArray containing (X - I) U (Y - I)
def candidateEvents(X, Y, x_ids, y_ids):
	return np.concatenate((X[x_ids],Y[y_ids]))

# -- Frecuency Sort Algorithm of MinDL Merge algorithm, orders events on X descendently
#@<NumpyArray> X: Event Sequence
#@Returns: Unique: unique events on X, Counts: frecuency count of event from Unique on X
def frecuencySort(X):
	unique, counts = np.unique(X, return_counts = True)
	frecuency_sort_index = np.argsort(counts)[::-1]
	assert len(unique) == len(frecuency_sort_index)
	unique = unique[list(frecuency_sort_index)]
	counts = counts[list(frecuency_sort_index)]
	return (unique, counts)

##################### DATABASE LOADING #####################
# -- Agavue Database User Interactions --
Secuences = {}	# Dictionary containing Ids to Secuences of events Ids. {'1':['1','3','4'],'2':['2','4','6'],...}	# this lists are further converted to numpy.array
Events = {}		# Dictionary mapping Ids to Event's Names. {'1':AppInit, '2': Resize,...}
ChartIds = {}	# Dictionary mapping Ids to ChartIds's sessions. {'1':session1,'2':session2,...}
Ips = {}		# Dictionary mapping Ids to User Ips. {'1':192431, '2':231344,...}
FingerprintSequenceCompresion = {}	# Dictionary mapping Ids to Sequence Compreession fingerprints, each id corresponds to Secuence

def loadData(nSecuences):
	ofile = open(dbPath  + "agavue_final.csv","r")
	beg = True
	for line in ofile:
		if beg:
			beg = False
			continue

		line = line.strip()
		elements = line.split(',')
		Secuences.setdefault(elements[1], []).append(elements[2])
		if len(Secuences) == nSecuences:
			break

	ofile.close()

	# Convert to numpy array
	"""
	for key in Secuences.keys():
		if len(Secuences[key]) < 5:
			Secuences.pop(key)
			continue
		Secuences[key] = np.array(Secuences[key])
	"""	

	# Compress and convert to numpy arry
	for key in Secuences.keys():
		(Secuences[key] ,  fingerprint)= compress_sequence(np.array(Secuences[key]))
		if len(Secuences[key]) < 5:
			Secuences.pop(key)
			continue
		FingerprintSequenceCompresion[key] = fingerprint
		#Secuences[key] = np.array(Secuences[key])

	ofile = open(dbPath + "ChartIds.csv","r")
	beg = True
	for line in ofile:
		if beg:
			beg = False
			continue

		line = line.strip()
		elements = line.split(',')
		ChartIds.setdefault(elements[0], elements[1])
	ofile.close()

	ofile = open(dbPath + "Eventos.csv","r")
	beg = True
	for line in ofile:
		if beg:
			beg = False
			continue

		line = line.strip()
		elements = line.split(',')
		Events.setdefault(elements[0], elements[1])
	ofile.close()


##################### MinDL Alg #####################
# -- This section implments the fast version of Minimun Description Length (MinDL) Algorithm --

alpha = 0.5	# control iterations
lamda = 1	# number of items on pattern

# -- Add event e to sequence p --
#@<NumpyArray> p: Event sequence
#@<String>	   e: Id of event to add to p. i.e: {'2','34','431'}
#@returns<NumpyArray> p: Event sequence containing p + e
def add(p, e):
	#p.append(e)
	return np.concatenate((p,[e]))

# -- Main subrouting of MinDL Algorithm --
# -- Merges two clusters giving as a result a new cluster containing both and the maximun
# 	 Description Length reduction archived --
#@<int/uuid>	ci: Unique Id of cluster ci in cluster_list
#@<int/uuid>	cj: Unique Id of cluster cj in cluster_list
#@Returns<float>		 L: Reduction length from merging ci and cj in c_
#@Returns<cluster>		 c_: New cluster object formed from ci and cj
def merge(ci, cj):
	cluster1 = cluster_list[ci]
	cluster2 = cluster_list[cj]

	# get lcs between patrons and receive difference ids
	pi = cluster1.p	# patron of cluster1
	pj = cluster2.p

	# pi_ids has ids from events in (pi - p)
	(p, pi_ids, pj_ids) = lcs(pi, pj)	# get common patron
	p_ = copy.deepcopy(p)

	# Candidate Events Ec = (Pi - P) U (Pj - P)
	cE = candidateEvents(pi, pj, pi_ids, pj_ids)

	# Sort Ec by frequency in desc order
	e, f = frecuencySort(cE)	# e: event id, f: frecuency

	L = -1

	## Gi U Gj
	#G_u = np.concatenate(cluster1.g, cluster2.g)
	gi = cluster1.g	# event sequences ids forming cluster1 
	gj = cluster2.g

	G_u = gi + gj

	n_edits_gi = sum([damerau_levenshtein_distance(list(Secuences[e_]),list(pi)) for e_ in gi])
	n_edits_gj = sum([damerau_levenshtein_distance(list(Secuences[e_]),list(pj)) for e_ in gj])

	# Pattern buildup phase
	for e in cE:
		p = add(p, e)
		n_edits_p = sum([damerau_levenshtein_distance(list(Secuences[e_]), list(p)) for e_ in G_u]) 
		L_ = len(pi) + len(pj) - len(p) + alpha*(n_edits_gi) + alpha*(n_edits_gj) - alpha*(n_edits_p) + lamda

		if L_ < 0 or L_ < L:
			break
		else:
			L = L_
			p_ = p
	c_ = Cluster(p_, G_u)
	return (L, c_)

# -- MinDL Algorithm --
# -- Main Algorithm from this program, performs the clustering of event sequences archiving the maximun reduction of description length among the sequences on each cluster, giving patterns.
# Input S = [S1, S2 .... Sn]
# Output C = {(P1,G1),(P2,G2),...,(PK,GK)}
#@Uses: Secuences, cluster_list, q
#@returns: Cluster_list filled with clusters formed during execution, each one with its representative pattern
def MinDL():
	global Secuences, cluster_list

	# Initial clusters are formed from existing Sequences
	for (k,v) in Secuences.items():
		P = Secuences[k]	# Cluster pattern is equal to actual sequence
		G = [k]				# Cluster's set of sequences is formed only by actual sequence
		c = Cluster(P,G)	# Create cluster
		add_cluster(c)		# Add new cluster to cluster_list (C)
	#assert len(cluster_list) == 58581

	# for all pairs Ci, Cj E C and i != j do
	C_size = counter_cluster_id		# number of clusters
	# Merge each pair of clusters (ci,cj) from C reducing Description Length
	for i in range(C_size):			# because each id came from (0,counter_cluster_id)
		for j in range(i+1, C_size):
			L, c_ = merge(i,j)
			# exit(0)	# debug merge
			#c_id = add_cluster(c_)
			if L > 0:
				enqueue(L, c_, i, j)	# Add id of tuple formed by (L,c_, ci, cj) to PriorityQueue q

	# Iterative merging phase
	while q.size is not 0:	# PriorityQueue it's not empty
		t = dequeue()		# Test and retrieve if there are elements left
		if t is None:		# PriorityQueue is empty
			break
		r = priorityQueue_dict[t.label]	# Pull tuple reference of item (L, c_, i, j) from q
		cnew = r.c_						# The chosen cluster is going to be added to cluster_list
		remove_clusters(r.ci, r.cj)		# Remove cluster forming cnew from containers
		c_id = add_cluster(cnew)		# Add cnew to cluster_list and get assigned id c_id
		# for c E C - cnew do
		for key in cluster_list:		# Now we can compute MinDL Reduction of cnew against others
			if key is c_id:				# dont merge with itself
				continue
			L, c_ = merge(key, c_id)	# Merge is possible because c_id is already on cluster_list
			if L > 0:
				enqueue(L, c_, key, c_id)	# Repeat process

	# cluster_list contains clusters and patterns found and you can play with them
	print "Patrones"
	for key, value in cluster_list.items():
		print "{}: {}".format(key, value.p)
		print "Secuences:"
		for sid in value.g:
			print "s{}: {}".format(sid,Secuences[sid])
		print "\n"

##### LSH FUNCTIONS ####
MinHashDict = {}
thStart = 0.8
thEnd = 0.2
thRate = 0.6

def lshInit(th):
	global MinHashDict
	print "Creating index"
	# Create LSH index
	MinHashDict = {}
	lsh = MinHashLSH(threshold=th, num_perm=50)
	
	# Create an LSH index for all clusters
	for Id, c in cluster_list.items():
		Sequence = c.p		# cluster pattern 
		MinHashDict[Id] = MinHash(num_perm=50)
		for d in Sequence:
			MinHashDict[Id].update(d.encode('utf8'))
		lsh.insert(Id, MinHashDict[Id])
	return lsh

def lshInsert(lsh, key):			# c is a cluster object
	MinHashDict[key] = MinHash(num_perm=50)
	Sequence = cluster_list[key].p
	for d in Sequence:
		MinHashDict[key].update(d.encode('utf8'))
	lsh.insert(key, MinHashDict[key])

def lshQuery(lsh, c_key):
	result = lsh.query(MinHashDict[c_key])	# pass Min hash value of cluster
	return result

def lshDelete(lsh,c_key):
	MinHashDict.pop(c_key)
	lsh.remove(c_key)			# this c_key has to be in MinHashDict

# Minimun Description Length + Locality Sensitive Hashing
def MinDLLSH():
	global Secuences, cluster_list

	# Initial clusters are formed from existing Sequences
	for (k,v) in Secuences.items():
		P = Secuences[k]	# Cluster pattern is equal to actual sequence
		G = [k]				# Cluster's set of sequences is formed only by actual sequence
		c = Cluster(P,G)	# Create cluster
		add_cluster(c)		# Add new cluster to cluster_list (C)
	#assert len(cluster_list) == 58581

	# for all pairs Ci, Cj E C and i != j do
	C_size = counter_cluster_id		# number of clusters

	th = thStart
	while th > thEnd:
		print "Lsh"
		# LSH table initialization with threshold th
		lsh = lshInit(th)		# initialize lsh index with all clusters from cluster_list

		print "MinDL"
		# MinDL
		for key in cluster_list.keys():
			similars = lshQuery(lsh, key)
			similars.remove(key)
			for key2 in similars:
				L, c_ = merge(key, key2)
				# exit(0)	# debug merge
				#c_id = add_cluster(c_)
				if L > 0:
					enqueue(L, c_, key, key2)	# Add id of tuple formed by (L,c_, ci, cj) to PriorityQueue q

		print "Merging"
		# Iterative merging phase
		while q.size is not 0:	# PriorityQueue it's not empty
			t = dequeue()		# Test and retrieve if there are elements left
			if t is None:		# PriorityQueue is empty
				break
			r = priorityQueue_dict[t.label]	# Pull tuple reference of item (L, c_, i, j) from q
			cnew = r.c_						# The chosen cluster is going to be added to cluster_list
			remove_clusters(r.ci, r.cj)		# Remove cluster forming cnew from containers

			cnew_id = add_cluster(cnew)		# Add cnew to cluster_list and get assigned id c_id

			lshDelete(lsh,r.ci)
			lshDelete(lsh,r.cj)
			lshInsert(lsh,cnew_id)	#

			clist_similar = lshQuery(lsh, cnew_id)
			clist_similar.remove(cnew_id)		# always pop same id
			# for c E C - cnew do
			for key in clist_similar:		# Now we can compute MinDL Reduction of cnew against others
				L, c_ = merge(key, cnew_id)	# Merge is possible because c_id is already on cluster_list
				if L > 0:
					enqueue(L, c_, key, cnew_id)	# Repeat process
		th = th*thRate

	# cluster_list contains clusters and patterns found and you can play with them
	print "Patrones"
	"""for key, value in cluster_list.items():
		print "{}: {}".format(key, value.p)
		print "Secuences: {}".format(len(value.g))
		#for sid in value.g:
		#	print "s{}: {}".format(sid, Secuences[sid])
		print "\n"
	"""
	save_clusters()

def save_clusters():
	ofile = open(outputPath + "clusters.dat","w")
	for key, value in cluster_list.items():
		ofile.write("Patron\n")
		pattern = value.p
		for e in pattern:
			ofile.write(e + " ")
		ofile.write("\n")
		for sid in value.g:
			sequence = Secuences[sid]
			for e in sequence:
				ofile.write(e + " ")
			ofile.write("\n")
	ofile.close()

	ofile = open(outputPath + "clusters_ids.dat","w")
	for key, value in cluster_list.items():
		ofile.write("Patron\n")
		pattern = value.p
		for e in pattern:
			ofile.write(e + " ")
		ofile.write("\n")
		for sid in value.g:
			ofile.write(sid + " ")
		ofile.write("\n")
	ofile.close()

###############################################33####

if __name__ == "__main__":
	# 500 works real good
	loadData(100)
	print "Loaded Correctetly: {}".format(len(Secuences))
	MinDLLSH()

