import pickle
import matplotlib.pyplot as plt

noncausal = pickle.load(open('44history_noncausal.p', 'rb'))
causal1 = pickle.load(open('44history_causal1.p', 'rb'))
causal2 = pickle.load(open('44history_causal2.p', 'rb'))
comb = pickle.load(open('44history_comb.p', 'rb'))

plt.title('Validation and Accuracy')
plt.subplot(4, 1, 1)
plt.xlabel('Epochs')
plt.ylabel('NonCausal LossAcc')
epochs = range(1, len(noncausal['acc']) + 1)
plt.plot(epochs, noncausal['val_loss'], 'ro', label='Non_causal Val Loss')
plt.plot(epochs, noncausal['val_acc'], 'r+', label='Non_causal Val Acc')
plt.plot(epochs, noncausal['loss'], 'r-', label='Non Causal1 Loss')
plt.legend()
plt.grid()

plt.subplot(4, 1, 2)
plt.xlabel('Epochs')
plt.ylabel('Causal1 LossAcc')
epochs = range(1, len(causal1['acc']) + 1)
plt.plot(epochs, causal1['val_loss'], 'mo', label='Causal1 Val Loss')
plt.plot(epochs, causal1['val_acc'], 'm+', label='Causal1 Acc')
plt.plot(epochs, causal1['loss'], 'm-', label='Causal1 Loss')
plt.legend()
plt.grid()


plt.subplot(4, 1, 3)
plt.xlabel('Epochs')
plt.ylabel('Causal2 LossAcc')
epochs = range(1, len(causal2['acc']) + 1)
plt.plot(epochs, causal2['val_loss'], 'bo', label='Causal2 Val Loss')
plt.plot(epochs, causal2['val_acc'], 'b+', label='Causal2 Val Acc')
plt.plot(epochs, causal2['loss'], 'b-', label='Causal2 Loss')
plt.legend()
plt.grid()


plt.subplot(4, 1, 4)
plt.xlabel('Epochs')
plt.ylabel('Comb LossAcc')
epochs = range(1, len(comb['acc']) + 1)
plt.plot(epochs, comb['val_loss'], 'bo', label='comb Val Loss')
plt.plot(epochs, comb['val_acc'], 'b+', label='comb Val Acc')
plt.plot(epochs, comb['loss'], 'b-', label='comb Loss')
plt.legend()
plt.grid()

plt.savefig("causal_L216_L44.png")
#plt.save("Training_validation_loss_orig.jpg")

plt.show()
