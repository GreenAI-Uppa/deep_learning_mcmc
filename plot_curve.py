import matplotlib.pyplot as plt
import json
import sys

json_file = sys.argv[1]
results = json.load(open(json_file))

plt.locator_params(nbins=4)
x = sorted([int(t) for t in results.keys()])
ac_train = [ results[str(t)]['train']['training accuracy'] for t in x ]
ac_test =  [ results[str(t)]['test']['testing accuracy'] for t in x ]
loss_train = [ results[str(t)]['train']['training loss'] for t in x ]
loss_test =  [ results[str(t)]['test']['test loss'] for t in x ]
fig, ax = plt.subplots()
ax.plot(x, ac_train, label="training accuracy")
ax.plot(x, ac_test, label="test accuracy")
ax.legend(loc="lower right", frameon=False)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.savefig(json_file.replace(".json","")+"_acc.png")

fig, ax = plt.subplots()
ax.plot(x, loss_train, label="training loss")
ax.plot(x, loss_test, label="test loss")
ax.legend(loc="lower right", frameon=False)
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.savefig(json_file.replace(".json","")+"_loss.png")
#plt.show()
