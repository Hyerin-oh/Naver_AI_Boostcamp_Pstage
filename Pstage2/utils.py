import random

def random_unk_modify(x):
    y = x[1].split(' ')
    while True:
      idx  = random.randrange(len(y))   # 1
      if (y[idx] != x[2] and y[idx] != x[5]) : break
    y[idx] = '<unk>'
    y = " ".join(y)
    return y

def entity_modify(x):
    print(x)
    y = x[1]
    print(y)
    y[int(x[3]) : int(x[4])+1] = ' # ' + x[2] + ' # '
    y[int(x[6]) : int(x[7])+1] = ' @ ' + x[5] + ' @ '
    return y[:int(x[3])] + ' # ' + x[2] + ' # ' + y[int(x[4])+1 : int(x[6])] + ' @ ' + x[5] + ' @ ' + y[int(x[7])+1 : ]