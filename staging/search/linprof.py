from memory_profiler import profile

from searchers import LinFinder

if __name__ == "__main__":
    finder = LinFinder()
    profile(finder.get_repeat)()
