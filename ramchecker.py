import psutil
import os

class RamChecker:
    def __init__(self):
        self.max_ram = 0
        self.process = psutil.Process(os.getpid())

    def update_max_ram(self):
        cur_ram = self.process.memory_info().rss / (1024 ** 3) # in GB
        if(cur_ram > self.max_ram):
            self.max_ram = cur_ram

    def get_max_ram(self):
        return self.max_ram

ram_checker = RamChecker()