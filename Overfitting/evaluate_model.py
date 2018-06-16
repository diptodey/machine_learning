import pickle



data = pickle.load(open("44pred.p", "rb"))

epochs = range(0, len(data['oup_exp']))

i = 0
for epoch in epochs:
    if (data['oup_modelComb_data_test'][epoch] !=  data['oup_exp'][epoch] ):
        print("%f   %f  %f  %f %f" %(data['oup_model_data_test'][epoch],
                              data['oup_modelc1_data_test'][epoch],
                              data['oup_modelc2_data_test'][epoch],
                              data['oup_modelComb_data_test'][epoch],
                              data['oup_exp'][epoch] ))
        i+=1

print("Matches = %f", (1 - i/25000.0))
