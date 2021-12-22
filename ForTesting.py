import numpy as np 
import random
import matplotlib.pyplot as plt 

def GenerateRandom(min, max, num):
	ourNum = np.zeros(num)

	for i in range(num):
		ourNum[i] = random.randrange(min, max)

	return ourNum

num = 300

x = GenerateRandom(-100, 100, num)
y = GenerateRandom(-100, 100, num)

print (7 % 3)

plt.scatter(x, y)
plt.show()

