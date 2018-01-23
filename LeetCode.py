def step(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    else:
        return step(n - 1) + step(n - 2)
import math

if __name__ == '__main__':
    d=0.3
    while(1):
        a=-(d*math.log(d,2)+(1-d)*math.log(1-d,2))
        if a<0.5:
            print(d)
            break
        else:
            d=d-0.0001