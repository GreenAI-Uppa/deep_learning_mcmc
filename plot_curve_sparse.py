import matplotlib.pyplot as plt
import json
import sys

json_file = sys.argv[1]
results = json.load(open(json_file))

plt.locator_params(nbins=4)
x = sorted([int(t) for t in results.keys()])
ac_test = [ results[str(t)]['test']['testing accuracy'] for t in x ]
ac_test_sparse =  [ results[str(t)]['test']['testing accuracy sparse'] for t in x ]
fig, ax = plt.subplots()
ax.plot(x, ac_test, label="test accuracy")
ax.plot(x, ac_test_sparse, label="sparse version")
ax.legend(loc="lower right", frameon=False)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.savefig(json_file.replace(".json","")+"_acc.png")

plt.show()
