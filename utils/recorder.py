import os
import math
import random
import numpy as np
from sortedcollections import ValueSortedDict

PATH = os.path.dirname(__file__)
DIR = os.path.join(PATH, "../predictions/disc/")
if not os.path.exists(DIR):
    os.makedirs(DIR)

class Recorder:

    def __init__(self, fname, restart=False):
        if '/' not in fname:
            fname = os.path.join(DIR, fname)
        self.fname = fname
        print "Loading Recorder from", self.fname
        self.record = ValueSortedDict(lambda x: -x[0])

        self.write_freq = 10
        
        self.read(restart)
        self.write()

    def read(self, restart):
        if os.path.exists(self.fname):
            for line in open(self.fname, 'r'):
                line = line.replace('[', '')
                line = line.replace(']', '')
                split = line.split()
                k = split[0]
                v = [float(e) for e in split[1:]]
                if restart:
                    if v[0] < 0:
                        self.record[k] = v
                else:
                    self.record[k] = v

    def save(self, k, v):
        if type(v) is np.ndarray:
            v = v.flatten()
            v = v.tolist()
        if type(v) is not list:
            v = [v]
        self.record[k] = v
        if len(self.record) % self.write_freq == 0:
            self.write()

    def contains(self, k):
        return k in self.record
    def get(self, k):
        return self.record[k]

    def write(self):
        with open(self.fname, 'w') as f:
            for k in self.record:
                v = [str(e) for e in self.record[k]]
                line = "{} {}\n".format(k, " ".join(v))
                f.write(line)
        #self.write_percentiles()

    def write_percentiles(self):
        # write percentiles
        percentiles = {}
        for k in self.record:
            p = int(math.floor(self.record[k][0] * 10))
            if p in percentiles:
                percentiles[p].append((k, self.record[k]))
            else:
                percentiles[p] = [(k, self.record[k])]

        with open(self.fname.replace(".txt", "_percentiles.txt"), 'w') as f:
            for p in xrange(10,-1,-1):
                if p in percentiles:
                    l = percentiles[p]
                    random.shuffle(l)
                    l = l[:10]
                    for k,v in l:
                        v = [str(e) for e in v]
                        f.write("{} {}\n".format(k," ".join(v)))


