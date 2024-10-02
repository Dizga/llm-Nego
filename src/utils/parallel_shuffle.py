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

# Example usage:
list1 = [1, 2, 3, 4]
list2 = ['a', 'b', 'c', 'd']
shuffled_list1, shuffled_list2 = parallel_shuffle(list1, list2)
print(shuffled_list1)
print(shuffled_list2)