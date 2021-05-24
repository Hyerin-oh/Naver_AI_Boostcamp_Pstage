import random

def random_unk_modify(x):
  """
  entity가 아닌 토큰들 중 하나를 unk로 변경해주는 함수
  ex) 이순신은 조선 <unk>의 무신이다.
  """
    y = x[1].split(' ')
    while True:
      idx  = random.randrange(len(y))   # 1
      if (y[idx] != x[2] and y[idx] != x[5]) : break
    y[idx] = '<unk>'
    y = " ".join(y)
    return y

def entity_modify(x):
  """
  special token을 붙여주는 함수 
  ENTITY1 앞뒤로 # , ENTITY2 앞뒤로 @ 라는 special token을 붙여줌
  ex ) # 이순신 # 은 조선 중기의 @ 무신 @ 이다.
  """
    y = x[1]
    y[int(x[3]) : int(x[4])+1] = ' # ' + x[2] + ' # '
    y[int(x[6]) : int(x[7])+1] = ' @ ' + x[5] + ' @ '
    return y[:int(x[3])] + ' # ' + x[2] + ' # ' + y[int(x[4])+1 : int(x[6])] + ' @ ' + x[5] + ' @ ' + y[int(x[7])+1 : ]