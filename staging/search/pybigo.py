import itertools
import time
from copy import deepcopy

import pandas as pd


class Comparison:
    def __init__(self, functions, genfunc=None) -> None:
        self.functions = functions
        self.genfunc = genfunc or (lambda x: x)

    def run(self, inputs, runs=1):
        results = []
        for x, k in itertools.product(inputs, range(runs)):
            testinput = self.genfunc(x)
            for fun in self.functions:
                test_instance = deepcopy(testinput)
                start = time.time()
                fun(test_instance)
                testtime = time.time() - start
                results.append(
                    {"fun": fun.__name__, "time": testtime, "insize": x, "run": k}
                )
        return pd.DataFrame(results)


def draw_comp(funs, inputs, runs=1, genfunc=None, figsize=(14, 7), log=""):
    res = Comparison(funs, genfunc).run(inputs, runs)
    res.pivot_table(
        index="insize", columns="fun", values="time", aggfunc="median"
    ).plot(figsize=figsize, logx="x" in log, logy="y" in log)
