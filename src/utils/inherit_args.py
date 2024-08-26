import omegaconf

def inherit_args(conf_source, conf_sink, inherit_keyword="inerit"):
    if not isinstance(conf_sink, omegaconf.dictconfig.DictConfig): return
    for key in conf_sink.keys():
        if conf_sink[key] == inherit_keyword: 
            conf_sink[key] = conf_source[key]
        else: inherit_args(conf_source[key], conf_sink[key], inherit_keyword)

