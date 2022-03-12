import collections
import heapq

def unique_types(words):
    return sorted(set(words))

def types_with_frequency(words, frequency):
    counter = collections.Counter(words)
    items = (item for item in counter.items() if item[1] >= frequency)
    items = sorted(items, key=lambda x: (-x[1], x[0]))
    return (w for w, c in items)

def top_k_types(words, k):
    counter = collections.Counter(words)
    items = heapq.nsmallest(k, counter.items(), key=lambda x: (-x[1], x[0]))
    return (w for w, c in items)
