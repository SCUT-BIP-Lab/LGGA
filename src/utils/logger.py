'''
@Project ：DHGA 
@File    ：logger.py
@IDE     ：PyCharm 
@Author  ：WXL
@Date    ：2026/4/25
'''


import os

logger = None
need_loghead = True
def config_logging(log_file = "runtime.log", log_level = "INFO", from_info = True):
    global logger
    global need_loghead
    if not logger:
        import logging
        level = logging.INFO
        if log_level.lower() == "info":
            level = logging.INFO
        elif log_level.lower() == "debug":
            level = logging.DEBUG
        logger = logging.getLogger()
        logger.setLevel(level = level)
        log_dir = os.path.dirname(log_file)
        if(not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        need_loghead = from_info
    return logger