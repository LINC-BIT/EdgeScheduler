import time


def get_cur_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
