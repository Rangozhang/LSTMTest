import random
import string

n_class = 2
n_data = 10000
f = open('gt.txt','r')
f2 = open('input.txt', 'w')
for line in f:
    tokens = line.strip().split()
    seq = list(tokens[1])
    print(seq)
    for j in xrange(n_data):
        data_line = ''.join(random.sample(string.uppercase, random.randint(0,1)))

        for each in seq:
            repeated_num = random.randint(1, 5)
            for z in xrange(repeated_num):
                data_line += each
            #data_line += ''.join(random.sample(string.uppercase, random.randint(3,4)))
            data_line += ''.join(random.sample(string.uppercase, repeated_num))
        data_line += ''.join(random.sample(string.uppercase, random.randint(0,1)))
        f2.write(tokens[0]+' '+data_line+'\n')
