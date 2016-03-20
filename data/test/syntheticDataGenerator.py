import os, sys
import random
import string
import math

n_class = 10
n_data = 10000
f = open('gt.txt','w')
f2 = open('input.txt', 'w')
for i in xrange(1,n_class+1):
    length = random.randint(4, 6)
    seq = random.sample(string.lowercase, length)
    print str(i) + ' ' + ''.join(seq)
    f.write(str(i)+' '+''.join(seq)+'\n')
    for j in xrange(n_data):
        data_line = ''.join(random.sample(string.uppercase, random.randint(0,3)))

        for each in seq:
            data_line += each + ''.join(random.sample(string.uppercase, random.randint(0,3)))
        data_line += ''.join(random.sample(string.uppercase, random.randint(3,4)))
        f2.write(str(i)+' '+data_line+'\n')

#os.system('python syntheticTestDataGenerator.py')
