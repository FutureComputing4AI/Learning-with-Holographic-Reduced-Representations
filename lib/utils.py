"""
Utility functions.
"""

from prettytable import PrettyTable
import pandas as pd
from time import time
import torch

GB_DIV = 1024 * 1024 * 1024


def print_memory_profile():
    """
    Get basic memory information.
    """
    device = torch.cuda.current_device()
    print("Allocated: {:.4f}".format(int(torch.cuda.memory_allocated()) / GB_DIV))
    print("Reserved: {:.4f}\n".format(int(torch.cuda.memory_allocated()) / GB_DIV))

# https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
def print_command_arguments(args):
    table = PrettyTable(['Parameter', 'Value'])
    table.title = 'Experimental Setup'
    for arg in vars(args):
        table.add_row([arg, getattr(args, arg)])
    print(table)

class Measure(object):
    """
    Manage runtimes for a specific code block.
    """
    def __init__(self, name):
        self._measure = name
        self._is_measuring = False
        self._elapsed_time = 0

    def is_measuring(self):
        return self._is_measuring

    def start(self):
        self._stime = time()
        self._is_measuring = True

    def end(self):
        self._etime = time()
        self._elapsed_time += self._etime - self._stime
        self._is_measuring = False

    def get_elapsed_time(self):
        return self._elapsed_time

    def get_name(self):
        return self._measure


class ExperimentTime(object):
    """
    Manage time for different parts in an experiment.
    """
    def __init__(self):
        self._table = pd.DataFrame(columns=['Measurement', 'Elapsed Time'])
        self._pos = 0
        self.measure = {}

    def _append(self, name):
        self._table.loc[self._pos] = [name, self.measure[name].get_elapsed_time()]
        self._pos += 1

    def register(self, name):
        if name in self.measure:
            print("Measurement with same name previously added.")
        else:
            self.measure[name] = Measure(name)

    def measure_time(self, name):
        if self.measure[name].is_measuring():
            self.measure[name].end()
            # Add time to the dataframe.
            self._append(name)
        else:
            self.measure[name].start()

    def get_measurements(self):
        return self._table
