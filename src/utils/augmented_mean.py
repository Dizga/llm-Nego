import numpy as np

def augmented_mean(data: list):
    if all(isinstance(i, (int, float, type(np.nan))) for i in data):
        filtered_data = [i for i in data if i is not np.nan]
        if not filtered_data:
            return np.nan
        return sum(filtered_data) / len(filtered_data)
    
    if all(isinstance(i, list) for i in data):
        max_length = max(len(sublist) for sublist in data)
        means = []
        for i in range(max_length):
            elements = [sublist[i] for sublist in data if i < len(sublist) and isinstance(sublist[i], (int, float)) and sublist[i] is not np.nan]
            elements = [e for e in elements if not np.isnan(e)]
            
            if not elements:
                means.append(np.nan)
            else:
                means.append(sum(elements) / len(elements))
        return means
    
    else:
        return [np.nan] * len(data)
    


