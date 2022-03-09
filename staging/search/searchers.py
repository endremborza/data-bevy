import bisect
from dataclasses import dataclass
from random import Random


@dataclass
class LinFinder:
    dpow: int = 8
    seed: int = 742

    def get_repeat(self):
        rng = Random(self.seed)
        stack = self._empty_stack()
        i = 0
        while True:
            e = rng.randint(0, int(10 ** self.dpow))
            if self._find(e, stack):
                break
            self._add(e, stack)
            i += 1
        return i, e

    def _empty_stack(self):
        return []

    def _find(self, e, stack):
        return e in stack

    def _add(self, e, stack):
        stack.append(e)


class LogFinder(LinFinder):
    def _find(self, e, stack):
        self._ind = bisect.bisect_left(stack, e)
        try:
            if e == stack[self._ind]:
                return True
        except IndexError:
            pass

    def _add(self, e, stack):
        stack.insert(self._ind, e)


class HashFinder(LinFinder):
    def _empty_stack(self):
        return set()

    def _add(self, e, stack):
        stack.add(e)

class BigFinder(LinFinder):
    def _empty_stack(self):
        return [0] * int(10 ** self.dpow)
    
    def _find(self, e, stack):
        return stack[e] > 0
    
    def _add(self, e, stack):
        stack[e] += 1

if __name__ == "__main__":
    for c in [LinFinder, LogFinder, HashFinder]:
        print(c.__name__, c().get_repeat())
