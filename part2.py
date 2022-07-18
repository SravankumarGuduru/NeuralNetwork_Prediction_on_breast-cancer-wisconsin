import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# from sklearn.model_selection import StratifiedKFold
# from sklearn.neural_network import MLPClassifier

logging.basicConfig(filename="part2.log", level=logging.INFO, format='%(message)s')


class NN:

    def __init__(self, data_file_path):
            self.data_file_path = data_file_path

    def normalize(self,data):
        for col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
            data[col] = np.exp(-(data[col] - data[col].mean()) ** 2 / (2 * (data[col].std()) ** 2))
        return data


    def split_data(self,df):
        X = df[["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]]
        X = self.normalize(X)
        Y = df["Class"]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
        return X_train, X_test, Y_train, Y_test
            

    def get_data(self,dataFile):
            raw_input = pd.read_csv(dataFile,names=["Sample code number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"])

            del raw_input["Sample code number"]
            raw_input = raw_input.drop(raw_input[raw_input["Bare Nuclei"] == "?"].index)
            raw_input = raw_input.astype(dtype={"Bare Nuclei":"int64"})
            
            raw_input = raw_input.reset_index()
            del raw_input["index"]
            
            raw_input["Class"][raw_input["Class"] == 4] = 1
            raw_input["Class"][raw_input["Class"] == 2] = 0
            
            #logging.info(f"shape {raw_input.shape}")
            #logging.info(raw_input.describe())
            
            return raw_input
        

    def fit(self):
        raw_input = self.get_data("https://drive.google.com/uc?id=1SSZEE1lEQnN2JpMcnQooCdjDdMYhhFIQ")
        X_train, X_test, Y_train, Y_test = self.split_data(raw_input) 
        self.train_evaluate(X_train, Y_train, X_test, Y_test)

                    
    def print_logs(self,history,activation_function,lr,epoch,num_hidden_layers):
        train_loss = history.history["loss"]
        train_accuracy= history.history["accuracy"]
        test_loss= history.history["val_loss"]
        test_accuracy = history.history["val_accuracy"]
        
        logging.info(f"activation_function {activation_function}, lr {lr}, epoch {epoch}, num_hidden_layers {num_hidden_layers}")
        logging.info(f"train_loss {train_loss[-1]}, train_accuracy {train_accuracy[-1]}, test_loss {test_loss[-1]}, test_accuracy {test_accuracy[-1]}")
        
        
    def confusion_matrix_function(self,X_test,Y_test,model):
        pred = model.predict(X_test.to_numpy(copy=True))
        pred = np.round(pred).astype(int).reshape(1,-1)[0]

        #from sklearn.metrics import confusion_matrix
        m = confusion_matrix(pred,Y_test)
        tn, fn, fp, tp=confusion_matrix(pred,Y_test).ravel()
        m=pd.crosstab(pred,Y_test)

        logging.info("===========Confusion matrix============")
        logging.info(m.to_string())


    def graphs(self,losses):
        fig, axs = plt.subplots(2,figsize=(25,25))
        fig.suptitle(f'epoch:{losses[0]["epoch"]}')

        axs[0].set_xlabel('epoch')
        axs[0].set_ylabel('Loss')  
        axs[0].set_title('Train vs Test loss')

        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel('Loss')  
        axs[1].set_title('Train vs Test Accuracy')

        for loss in losses:        
            axs[0].plot(loss['train_loss'],label =f'{loss["activation_function"]},lr:{loss["learning_rate"]},E:{loss["epoch"]},layers:{loss["num_hidden_layers"]} train_loss')
            axs[0].plot(loss['test_loss'],label = f'{loss["activation_function"]},lr:{loss["learning_rate"]},E:{loss["epoch"]},layers:{loss["num_hidden_layers"]} test_loss')

            axs[1].plot(loss['train_accuracy'],label = f'{loss["activation_function"]},lr:{loss["learning_rate"]},E:{loss["epoch"]},layers:{loss["num_hidden_layers"]} train_accuracy')
            axs[1].plot(loss['test_accuracy'],label =f'{loss["activation_function"]},lr:{loss["learning_rate"]},E:{loss["epoch"]},layers:{loss["num_hidden_layers"]} test_accuracy')

        axs[0].legend()
        axs[1].legend()
        plt.show()


    def model_train(self,X_train, Y_train, X_test, Y_test, learning_rate = 0.1, activation_function = "sigmoid", epochs = 200, hidden_layers = 2, loss_function = "binary_crossentropy"):
        model = Sequential()
        model.add(Dense(2, input_dim = 9, kernel_initializer ='normal', activation = activation_function))
        #model.add(Dense(3, kernel_initializer ='normal', activation = activation_function))
        for i in range(hidden_layers - 1):
            model.add(Dense(3, kernel_initializer ='normal', activation = activation_function))
            
        model.add(Dense(1, kernel_initializer ='normal', activation = 'sigmoid'))
        model.compile(loss = loss_function, optimizer = SGD(learning_rate=learning_rate), metrics = ['accuracy'])

        history = model.fit(X_train.to_numpy(copy=True), Y_train.to_numpy(copy=True), validation_data = (X_test.to_numpy(copy=True), Y_test.to_numpy(copy=True)), epochs = epochs, batch_size = 5, verbose = 0)
        scores = model.evaluate(X_test.to_numpy(copy=True), Y_test.to_numpy(copy=True),  batch_size=137,verbose=0)        

        logging.info(f"scores - error: {scores[0]}, acc: {scores[1]}")

        return history,model

        
    def train_evaluate(self,X_train, Y_train, X_test, Y_test):
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]
        num_hidden_layers = [2, 3]

        losses = []
        subplot_losses = []
        
        for epoch in max_iterations:
            for activation_function in activations:
                for lr in learning_rate:
                    
                    history,model = self.model_train(X_train, Y_train, X_test, Y_test, learning_rate = 0.1, activation_function = activation_function, epochs = epoch, hidden_layers = num_hidden_layers[0])

                    loss = {
                        "epoch":epoch,
                        "activation_function": activation_function,
                        "learning_rate": lr,
                        "num_hidden_layers":num_hidden_layers[0],
                        "train_loss" : history.history["loss"],
                        "train_accuracy" : history.history["accuracy"],
                        "test_loss" : history.history["val_loss"],
                        "test_accuracy" : history.history["val_accuracy"]
                    }
                    losses.append(loss.copy())
                    subplot_losses.append(loss.copy())
                    
                    self.print_logs(history,activation_function,lr,epoch,num_hidden_layers[0])
                    self.confusion_matrix_function(X_test,Y_test,model)
                    
                    history,model = self.model_train(X_train, Y_train, X_test, Y_test, learning_rate = 0.1, activation_function = activation_function, epochs = epoch, hidden_layers = num_hidden_layers[1])
                    loss = {
                        "epoch":epoch,
                        "activation_function": activation_function,
                        "learning_rate": lr,
                        "num_hidden_layers":num_hidden_layers[0],
                        "train_loss" : history.history["loss"],
                        "train_accuracy" : history.history["accuracy"],
                        "test_loss" : history.history["val_loss"],
                        "test_accuracy" : history.history["val_accuracy"]
                    }
                    losses.append(loss.copy())
                    subplot_losses.append(loss.copy())
                    
                    self.print_logs(history,activation_function,lr,epoch,num_hidden_layers[1])
                    self.confusion_matrix_function(X_test,Y_test,model)
                    
                    #break
                # subplot_losses = losses.copy() 
                # self.graphs(subplot_losses) 
                # subplot_losses = []
                #break
        self.graphs(losses)
        losses = []  

    
def main():
    data_file_path = "https://drive.google.com/uc?id=1SSZEE1lEQnN2JpMcnQooCdjDdMYhhFIQ"
    newural_network = NN(data_file_path)
    newural_network.fit()

if __name__ == '__main__':
    main()