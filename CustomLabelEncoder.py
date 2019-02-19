from collections import defaultdict

class CustomLabelEncoder:

    def __init__(self):

        self.l2i = defaultdict(int)
        self.i2l = defaultdict(str)
        self.i = 0

    def encode(self, m):

        em = []
        for row in m:
            er = []
            for val in row[:-1]: # not encoding last one (label), remains True or False
                if val in self.l2i:
                    er.append(self.l2i[val])
                else:
                    self.i += 1
                    self.l2i[val] = self.i
                    self.i2l[self.i] = val
                    er.append(self.i)
            er.append(row[-1])
            em.append(er)
        return em

    def vocabIndex(self, m):

        self.w2i = defaultdict(int)
        self.i2w = defaultdict(str)
        self.wi = 1 # 0 is reserved for unknown words
        for row in m:
            w = row[0]
            if not w in self.w2i:
                self.w2i[w] = self.wi
                self.i2w[self.wi] = w
                self.wi += 1
        
