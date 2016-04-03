import os, sys
import random
import string
import math

n_class = 10
n_data = 10000
f = open('gt.txt','r')
f2 = open('input.txt', 'w')
for line in f:
    tokens = line.strip().split()
    seq = list(tokens[1])
    print(seq)
    for j in xrange(n_data):
        data_line = ''.join(random.sample(string.uppercase, random.randint(0,3)))

        for each in seq:
            data_line += each + ''.join(random.sample(string.uppercase, random.randint(0,3)))
        data_line += ''.join(random.sample(string.uppercase, random.randint(3,4)))
        f2.write(tokens[0]+' '+data_line+'\n')

