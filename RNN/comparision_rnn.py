import pickle
import matplotlib.pyplot as plt

binadd = pickle.load(open('history_binaryadd.p', 'rb'))
plt.title('Validation and Accuracy')



plt.subplot(1, 1, 1)
plt.xlabel('Epochs')
plt.ylabel('LossAcc')
epochs = range(1, len(binadd['acc']) + 1)
plt.plot(epochs, binadd['val_loss'], 'bo', label='comb Val Loss')
plt.plot(epochs, binadd['val_acc'], 'b+', label='comb Val Acc')
plt.plot(epochs, binadd['loss'], 'b-', label='comb Loss')
plt.plot(epochs, binadd['acc'], 'r-', label='acc')
plt.legend()
plt.grid()

#plt.savefig("causal_L216_L44.png")
#plt.save("Training_validation_loss_orig.jpg")

plt.show()