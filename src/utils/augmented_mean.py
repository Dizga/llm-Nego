def augmented_mean(data: list):
    if all(isinstance(i, (int, float)) for i in data):
        return sum(data) / len(data)
    
    if all(isinstance(i, list) for i in data):
        max_length = max(len(sublist) for sublist in data)
        means = []
        for i in range(max_length):
            elements = [sublist[i] for sublist in data if i < len(sublist)]
            means.append(sum(elements) / len(elements))
        return means
    
    raise ValueError("Input must be a list of numbers or a list of lists of numbers.")
