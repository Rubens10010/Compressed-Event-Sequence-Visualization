import Queue as Q
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import numpy as np
from math import *
from heap import Node, PriorityQueue
import uuid
import copy

dbpath = "../AgavueDB/"
counter_cluster_id = 0

class Cluster(object):
	def __init__(self, p, g):
		self.p = p	# patron
		self.g = g	# list of event secuences Ids

class PriorityQueue_tuple(object):
	def __init__(self, l, c_, ci, cj):
		self.l = l
		self.c_ = c_
		self.ci = ci
		self.cj = cj

###############	containers	#####################3
cluster_list = dict()	### Stores the clusters formed
priorityQueue_dict = dict()	### Maps the priorityQueue elements to Tuple objects
q = PriorityQueue([])	### Priority Queue
#############//////////////##################

def add_cluster(c):
	global counter_cluster_id
	cluster_list[counter_cluster_id] = c
	counter_cluster_id += 1
	return counter_cluster_id-1

def remove_cluster(ckey):
	cluster_list.pop(ckey)	## check

def getNextLabel():
	return uuid.uuid4()

def enqueue(L, c_, ci, cj):
	global priorityQueue_dict, q
	id = getNextLabel()
	x = Node(label=id, priority = L)
	q.insert(x)
	priorityQueue_dict[id] = PriorityQueue_tuple(L, c_, ci, cj)

def remove_clusters(ci, cj):
	# Remove ci, cj from C
	global priorityQueue_dict
	remove_cluster(ci)
	remove_cluster(cj)
	#print "Removed {} and {} clusters".format(ci, cj)

	# remove all pairs containing ci or cj from Q
	# Removing them from dict makes them invisible for Q
	for key, value in priorityQueue_dict.items():
		if value.ci == ci or value.ci == cj or value.cj == ci or value.cj == cj:
			priorityQueue_dict.pop(key)

def dequeue():
	global priorityQueue_dict, q
	while(True):
		min = q.shift()
		if min is None:
			break
		#print "Pop {}".format(min.label)
		# check if it is removed
		if min.label not in priorityQueue_dict:
			#print "{} was removed".format(min.label)
			continue
		#print "Got {}".format(min.label)
		break
	return min

# Dynamic programming implementation of LCS problem
# Returns length of LCS for X[0..m-1], Y[0..n-1] 
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

# Space optimized Python
# implementation of LCS problem
 
# Returns length of LCS for 
# Pattern: X[0..m-1]
# Secuence: Y[0..n-1]
def lcs_distance(X, Y):
    # Find lengths of two strings
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
 
    # Last filled entry contains length of LCS
    # for X[0..n-1] and Y[0..m-1]
    return L[bi][n]

def candidateEvents(X, Y, x_ids, y_ids):
	return np.concatenate((X[x_ids],Y[y_ids]))

def frecuencySort(X):
	unique, counts = np.unique(X, return_counts = True)
	frecuency_sort_index = np.argsort(counts)[::-1]
	assert len(unique) == len(frecuency_sort_index)
	unique = unique[list(frecuency_sort_index)]
	counts = counts[list(frecuency_sort_index)]
	return (unique, counts)

##################### DATABASE LOADING #####################
Secuences = {}
Events = {}
ChartIds = {}
Ips = {}

def loadData():
	nSecuences = 100
	ofile = open(dbpath + "agavue_final.csv","r")
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

	# For index list retriving
	for key in Secuences.keys():
		Secuences[key] = np.array(Secuences[key])

	ofile = open(dbpath + "ChartIds.csv","r")
	beg = True
	for line in ofile:
		if beg:
			beg = False
			continue

		line = line.strip()
		elements = line.split(',')
		ChartIds.setdefault(elements[0], elements[1])
	ofile.close()

	ofile = open(dbpath + "Eventos.csv","r")
	beg = True
	for line in ofile:
		if beg:
			beg = False
			continue

		line = line.strip()
		elements = line.split(',')
		Events.setdefault(elements[0], elements[1])
	ofile.close()


##################### MinDL Alg #####################3
alpha = 0.5
lamda = 0

def add(p, e):
	#p.append(e)
	return np.concatenate((p,[e]))

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

# Input S = [S1, S2 .... Sn]
# Output C = {(P1,G1),(P2,G2),...,(PK,GK)}
def MinDL():
	global Secuences, cluster_list
	for (k,v) in Secuences.items():
		P = Secuences[k]
		G = [k]
		c = Cluster(P,G)
		add_cluster(c)
	# q = {}
	#assert len(cluster_list) == 58581
	i = 0
	print counter_cluster_id

	# for all pairs Ci, Cj E C and i != j do
	C_size = counter_cluster_id
	for i in range(C_size):
		for j in range(i+1, C_size):
			L, c_ = merge(i,j)
			# exit(0)	# debug merge
			#c_id = add_cluster(c_)
			if L > 0:
				enqueue(L, c_, i, j)

	print q.size
	# Iterative mergin phase
	while q.size is not 0:	# is not empty
		t = dequeue()
		if t is None:
			break
		r = priorityQueue_dict[t.label]
		cnew = r.c_
		remove_clusters(r.ci,r.cj)
		#print cnew.p
		c_id = add_cluster(cnew)
		# for c E C - cnew do
		for key in cluster_list:
			if key is c_id:
				continue
			L, c_ = merge(key, c_id)
			if L > 0:
				enqueue(L, c_, key, c_id)
		#for c in 

	print "Patrones"
	for key, value in cluster_list.items():
		print "{}: {}".format(key, value.p)
		print "Secuences:"
		for sid in value.g:
			print "s{}: {}".format(sid,Secuences[sid])
		print "\n"
		
###############################################33####

#print Interactions.keys()[:10]
#print Interactions.values()[:10]

"""
loadData()
i = 0
for k in Interactions.keys():
	if i > 10:
		break
	i+=1
	print k 
	print Interactions[k]
"""

# Driver program
#X = "AGGTAB"
#Y = "GXTXAYB"
#lcs(X, Y, m, n)

"""
X = np.array(['10','50','10','20','30','20','40','50'])
Y = np.array(['10','20','30','40','30','50'])
(p, x_ids, y_ids) = lcs(X, Y)

print X
print Y
print "lcs:"
print p
print x_ids
print y_ids

print lcs_distance(X,Y)	# edits
print damerau_levenshtein_distance(list(X),list(Y))	# edits
cE = candidateEvents(p, X, Y, x_ids, y_ids)
print cE
u,idx = frecuencySort(cE)
print zip(u,idx)
"""
loadData()
MinDL()
