import numpy as np
import matplotlib.pyplot as plt


x = []
y = []

epsilon = 0.995
epsilon_discount= 0.998 


for i in range(6000):
    epsilon = epsilon*epsilon_discount
    x.append(i)
    y.append(epsilon)
   

plt.plot(x,y)
plt.xlabel('episode')
plt.ylabel('epsilon')
plt.title('epsilon decay')
plt.savefig('epsilon_decay.png')
plt.show()



