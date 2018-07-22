import Queue as Q
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import numpy as np
from math import *

class QueueTuple(object):
	def __init__(self, priority, c_, ci, cj):
		self.priority = priority
		self.c_ = c_
		self.ci = ci
		self.cj = cj

	def __cmp__(self, other):
		return -1*cmp(self.priority, other.priority)

"""
ofile = open("agavue_final.csv","r")

Interactions = {}
Events = {}
ChartIds = {}
Ips = {}

beg = True
for line in ofile:
	if beg:
		beg = False
		continue

	line = line.strip()
	elements = line.split(',')
	Interactions.setdefault(elements[1], []).append(elements[2])

ofile.close()

# For index list retriving
for key in Interactions.keys():
	Interactions[key] = np.array(Interactions[key])

ofile = open("ChartIds.csv","r")
beg = True
for line in ofile:
	if beg:
		beg = False
		continue

	line = line.strip()
	elements = line.split(',')
	ChartIds.setdefault(elements[0], elements[1])
ofile.close()

ofile = open("Eventos.csv","r")
beg = True
for line in ofile:
	if beg:
		beg = False
		continue

	line = line.strip()
	elements = line.split(',')
	Events.setdefault(elements[0], elements[1])
ofile.close()
"""

#print Interactions.keys()[:10]
#print Interactions.values()[:10]

"""
i = 0
for k in Interactions.keys():
	if i > 10:
		break
	i+=1
	print k 
	print Interactions[k]
"""
# Dynamic programming implementation of LCS problem
# https://www.geeksforgeeks.org/longest-common-subsequence/
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
    return (lcs, x_ids, y_ids)

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

def candidateEvents(p, X, Y, x_ids, y_ids):
	return np.concatenate((X[x_ids],Y[y_ids]))

def frecuencySort(X):
	unique, counts = np.unique(X, return_counts = True)
	frecuency_sort_index = np.argsort(counts)[::-1]
	assert len(unique) == len(frecuency_sort_index)
	unique = unique[list(frecuency_sort_index)]
	counts = counts[list(frecuency_sort_index)]
	return (unique, counts)

"""
Compute the Damerau-Levenshtein distance between two given
strings (s1 and s2)
"""
"""
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in xrange(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in xrange(-1,lenstr2+1):
        d[(-1,j)] = j+1
 
    for i in xrange(lenstr1):
        for j in xrange(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            # algorithm difference with Levenshtein distance
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
 
    return d[lenstr1-1,lenstr2-1]
"""
# Driver program
#X = "AGGTAB"
#Y = "GXTXAYB"
#lcs(X, Y, m, n)

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

################################



################################



class cluster(object):
	def __init__(self, P, G):
		self.p = P
		self.g = G

l1 = 10
c1 = cluster(p,[0,1])

p2 = ['1','3','4']

l2 = 20
c2 = cluster(p2, [1,2,3])

p3 = ['3','5']
c_ = cluster(p3,[1,2,3,4,5,6])

q = Q.PriorityQueue()
q.put(QueueTuple(l1, c_, c1, c2))
q.put(QueueTuple(l2, c_, c2, c1))

while not q.empty():
	qTuple = q.get()
	print "Got: ", qTuple.priority

pq = []                         # list of entries arranged in a heap
entry_finder = {}               # mapping of tasks to entries
REMOVED = '<removed-task>'      # placeholder for a removed task
counter = itertools.count()     # unique sequence count

def add_task(task, priority=0):
    'Add a new task or update the priority of an existing task'
    if task in entry_finder:
        remove_task(task)
    count = next(counter)
    entry = [priority, count, task]
    entry_finder[task] = entry
    heappush(pq, entry)

def remove_task(task):
    'Mark an existing task as REMOVED.  Raise KeyError if not found.'
    entry = entry_finder.pop(task)
    entry[-1] = REMOVED

def pop_task():
    'Remove and return the lowest priority task. Raise KeyError if empty.'
    while pq:
        priority, count, task = heappop(pq)
        if task is not REMOVED:
            del entry_finder[task]
            return task
    raise KeyError('pop from an empty priority queue')
