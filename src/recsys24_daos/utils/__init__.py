import time


class Timer:
    def __init__(self, get_time=time.perf_counter, print=False):
        self.get_time = get_time
        self.print = print
    
    def __enter__(self):
        self.start = self.get_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = self.get_time() - self.start
        if self.print:
            print(self.time)
