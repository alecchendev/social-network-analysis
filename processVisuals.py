import matplotlib.pylab as plt
import matplotlib.colors as colors
import seaborn as sns

data = [5, 12, 8]
colors = ['#022B3A', '#1F7A8C', '#02C3BD']

fig, ax = plt.subplots(figsize=(10,6))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks([])
plt.xticks([])

for i in range(3):
    plt.bar(x=i, height=data[i], color=colors[i], width=1)

plt.xlim(-0.55,len(data)-.45)
plt.ylim(0, 30)

plt.show()