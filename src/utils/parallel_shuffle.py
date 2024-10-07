import random

def parallel_shuffle(*lists):
    # Assert all lists are of the same length
    length = len(lists[0])
    for lst in lists:
        assert len(lst) == length, "All lists must be of the same length"
    
    # Create a sequence of indices and shuffle it
    indices = list(range(length))
    random.shuffle(indices)
    
    # Create new lists with the shuffled order
    shuffled_lists = []
    for lst in lists:
        shuffled_lists.append([lst[i] for i in indices])
    
    return tuple(shuffled_lists)

