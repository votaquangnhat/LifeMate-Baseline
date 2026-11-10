"""Cache repeated reads until a non-read call; never suppress writes or inspect hidden state."""
import copy
import json


class ReadRepeatGuard:
    def __init__(self):
        self.cache = {}
        self.blocks = 0

    def lookup(self, call, kind):
        if kind != 'read':
            # Even a failed or unknown non-read action invalidates the cache.
            self.cache.clear()
            return False, None
        key = json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        if key not in self.cache:
            return False, None
        self.blocks += 1
        return True, copy.deepcopy(self.cache[key])

    def record(self, call, result, kind):
        if kind != 'read':
            self.cache.clear()
            return
        key = json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        self.cache[key] = copy.deepcopy(result)
