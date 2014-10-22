class BinomTable(object):
    def __init__(self, max):
        self.table = [[0 for k in range(max + 1)] for n in range(max + 1)]
        self.max = max
        self.build()
        
    def build(self):
        for n in range(self.max + 1):
            for k in range(self.max + 1):
                if k == 0:      b = 1
                elif  k > n:    b = 0
                elif k == n:    b = 1
                elif k == 1:    b = n
                elif k > n-k:   b = self.table[n][n-k]
                else:
                    b = self.table[n-1][k] + self.table[n-1][k-1]
                self.table[n][k] = b
