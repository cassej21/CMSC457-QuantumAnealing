import math
import numpy as np
import random

class ePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Curve:
    def __init__(self, x):
        self.b = x[0]
        self.m = x[1]
        self.n = x[2]
    
    def f(self, x):
        return self.b + self.m*x + self.n*(x**2)

def least_squares(curve, points):
    sum = 0
    for point in points:
        sum += (point.y - curve.f(point.x))**2
    return sum


points = [ePoint(-1.0,-1.5),
          ePoint(-0.5,-.75),
          ePoint(-0.25,-0.375),
          ePoint(0.25,0.375),
          ePoint(0.0,0.0), 
          ePoint(.5,0.75), 
          ePoint(1.0,1.5),
          ePoint(1.5,2.25),
          ePoint(2.0,3.0), 
          ePoint(3.0,4.5), 
          ePoint(4.0,6.0)]
#Cycle count
n = 50
#Trials per cycle
m = 50
#Moves made
na = 0.0

#probs of accepting at given trial
p1 = .5
p50 = .001
#temp at given trial
t1 = -1.0/math.log(p1)
t50 = -1.0/math.log(p50)
tFrac = (t50/t1)**(1.0/(n-1.0))

consList = np.zeros((n+1,3))
eList = np.zeros(n+1)

cons_curr = [0, 1.50,-0.0]
cons_test = cons_curr
consList[0] = cons_curr
curv_curr = Curve(cons_curr)
E = least_squares(curv_curr, points)
eList[0] = E

energ_avg = 0.0
t = t1
for i in range(n):
    print('Cycle number ' + str(i) + ", Temp = " + str(t))
    for j in range(m):
        if (j % 3 == 0):
            cons_test[0] = cons_curr[0] + random.uniform(-.5,.5)
            cons_test[0] = min(max(cons_test[0],-2.0), 2.0)
        elif (j % 3 == 1):
            cons_test[1] = cons_curr[1] + random.uniform(-.5,.5)
            cons_test[1] = min(max(cons_test[1],-2.0), 2.0)
        else:
            cons_test[2] = cons_curr[2] + random.uniform(-.5,.5)
            cons_test[2] = min(max(cons_test[2],-2.0), 2.0)
        

        curv_curr = Curve(cons_test)
        deltaE = least_squares(curv_curr, points) - E
        if (deltaE > 0):
            if (i == 0 and j == 0):
                energ_avg = abs(deltaE)
            p = math.exp(-abs(deltaE)/(energ_avg*t))
            if (random.random() < p):
                accept = True
            else:
                accept = False
        else:
            accept = True
        if (accept):
            cons_curr = cons_test
            E = least_squares(curv_curr, points)
            na = na + 1
            energ_avg = (energ_avg * (na-1.0) + abs(deltaE)) / na
    consList[i+1] = cons_curr
    eList[i+1] = E
    t = tFrac * t

print(consList)
print(eList)