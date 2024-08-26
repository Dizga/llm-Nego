

def inherit_args(dict_0, dict_1, inherit_keyword="inerit"):
    if not dict_0.has_keys: return
    for key in dict_0.keys():
        if key == inherit_keyword: dict_0['key'] = dict_1['key']
        else: inherit_args(dict_0[key], dict_1[key], inherit_keyword)
    
