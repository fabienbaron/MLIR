using PyCall
@pyimport matplotlib.pyplot as plt
plt.plot(rand(10))
plt.ion()
plt.show()
#plt.draw()
plt.ioff()


