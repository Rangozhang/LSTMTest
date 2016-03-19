import os, sys
import random
import string
import math

n_data = 5
f = open('gt.txt','r')
f2 = open('test.txt', 'w')
f3 = open('input.txt','r')

train_list = []
for line in f3:
    train_list.append(line.strip().split()[1])

for line in f:
    tokens = line.strip().split()
    print tokens
    seq = list(tokens[1])
    for j in xrange(n_data):
        data_line = ''
        flag = False
        while not flag:
            data_line = ''.join(random.sample(string.uppercase, random.randint(0,3)))
            for each in seq:
                data_line += each + ''.join(random.sample(string.uppercase, random.randint(0,3)))
            if data_line not in train_list:
               flag = True
        f2.write(tokens[0]+' '+data_line+'\n')
