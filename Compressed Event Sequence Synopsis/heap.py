from collections import deque


def is_iterable(obj):
    '''Test if `obj` is iterable.'''
    try: iter(obj)
    except TypeError: return False
    return True

def has_label(obj):
    '''Test if `obj` has a "label" attribute.'''
    try: return obj.label
    except AttributeError: return False

def are_labeled(objs):
    '''Test if `objs` all have "label" attributes.'''
    return all(is_iterable(obj) and has_label(obj) for obj in objs)


class PriorityQueue:
    '''
    Queue of elements/nodes ordered by priority.

    Implementation uses a binary heap where each node is less than
    or equal to its children.  Note that nodes can be anything as 
    long as they're comparable.

    If you initialize your queue with node elements that contain
    `node.label` attributes, you can then delete nodes by label.

    '''
    def __init__(self, nodes):
        self.size = 0
        self.heap = deque([None])
        self.labeled = False
        for n in nodes: self.insert(n)
        if are_labeled(nodes):
            self.labeled = True
            self.position = {node.label: i+1 for i, node in enumerate(self.heap)
                                                         if i > 0}

    def __str__(self):
        return str(list(self.heap)[1:])

    def __eq__(self, other):
        return list(self.heap)[1:] == other

    def node(self, i):
        '''
        Return {index, value} of node at index i.

        This is used for testing parent/child relations.
        
        '''
        return dict(index=i, value=self.heap[i])

    def parent(self, child):
        '''
        Return {index, value} for parent of child node.

        '''
        i = child['index']
        p = i // 2
        return self.node(p)

    def children(self, parent):
        '''
        Return list of child nodes for parent.

        '''
        p = parent['index']
        l, r = (p * 2), (p * 2 + 1)     # indices of left and right child nodes
        if r > self.size:
            return [self.node(l)]
        else:
            return [self.node(l), self.node(r)]

    def swap(self, i, j):
        '''
        Swap the values of nodes at index i and j.

        '''
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        if self.labeled:
            I, J = self.heap[i], self.heap[j]
            self.position[I.label] = i
            self.position[J.label] = j

    def shift_up(self, i):
        '''
        Percolate upward the value at index i to restore heap property.

        '''
        p = i // 2                  # index of parent node
        while p: 
            if self.heap[i] < self.heap[p]:
                self.swap(i, p)     # swap with parent
            i = p                   # new index after swapping with parent
            p = p // 2              # new parent index

    def shift_down(self, i):
        '''
        Percolate downward the value at index i to restore heap property.

        '''
        c = i * 2
        while c <= self.size:
            c = self.min_child(i)
            if self.heap[i] > self.heap[c]:
                self.swap(i, c)
            i = c                   # new index after swapping with child
            c = c * 2               # new child index

    def min_child(self, i):
        '''
        Return index of minimum child node.

        '''
        l, r = (i * 2), (i * 2 + 1)     # indices of left and right child nodes
        if r > self.size:
            return l
        else:
            return l if self.heap[l] < self.heap[r] else r

    @property
    def min(self):
        '''
        Return minimum node in heap.

        '''
        return self.heap[1]

    @property
    def top(self):
        '''
        Return top/minimum element in heap.

        '''
        return self.min

    def shift(self):
        '''
        Shift off top/minimum node in heap.

        '''
        return self.pop(1)

    def pop(self, i):
        '''
        Pop off node at index `i` in heap.

        '''
        if self.size == 0: return None
        v = self.heap[i]                # return specified node
        self.swap(self.size, i)         # move last element to i
        self.heap.pop()                 # delete last element
        self.size -= 1                  # decrement size
        self.shift_down(i)              # percolate top value down if necessary
        return v

    def delete(self, label): 
        '''
        Pop off node with specified label attribute.

        '''
        try:
            i = self.position[label]
            self.position[label] = None
        except KeyValueError:
            print 'node with label "{}" does not exist'.format(label)
            return None
        return self.pop(i)

    def insert(self, node):
        '''
        Append `node` to the heap and percolate up
        if necessary to maintain heap property.

        '''
        if has_label(node) and self.labeled:
            self.position[node.label] = self.size
        self.heap.append(node)
        self.size += 1
        self.shift_up(self.size)

    def sort(self):
        '''
        Return sorted array of elements in current heap.

        '''
        sorted = [self.shift() for i in range(self.size)]
        self.heap = deque([None] + sorted)
        self.size = len(self.heap) - 1
        return sorted


class Node(dict):
    '''
    Nodes are just dicts comparable by their `priority` key.

    Your nodes should contain `label` attributes when you're 
    inserting into a PriorityQueue and you'd like to delete 
    a node by its label from the underlying heap.

        >>> a = Node(dict(label='a', priority=5, msg='hi!'))

    '''
    def __cmp__(self, other):
        '''
        should return a negative integer if self < other, zero if
        self == other, and positive if self > other.

        '''
        """
        if self['priority'] < other['priority']:
            return -1
        elif self['priority'] == other['priority']:
            return 0
        else:
            return 1
        """
        if self['priority'] < other['priority']:
            return 1
        elif self['priority'] == other['priority']:
            return 0
        else:
            return -1

    def __eq__(self, other):
        return self['priority'] == other['priority']

    def __getattr__(self, attr):
        return self.get(attr, None)


if __name__ == '__main__':

    q = PriorityQueue([3, 1, 2, 4])
    assert q.min == 1
    assert q.sort() == [1, 2, 3, 4]
    
    x = q.shift()
    assert x == 1
    assert q.sort() == [2, 3, 4]

    a = Node(label='a', msg="boom!", priority=1)
    b = Node(label='b', msg="hi", priority=2)
    c = Node(label='c', msg="ok", priority=3)
    d = Node(label='d', msg="oh", priority=4)
    e = Node(msg="no", priority=5)

    assert a < b < c < d

    q = PriorityQueue([b, c, d])
    assert q.top == b
    assert q.top.msg == 'hi'
    assert q == [b, c, d]

    q.insert(a)
    assert q.sort() == [a, b, c, d]

