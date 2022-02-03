import os
import train 

def menu(trained : bool):
    if trained: 
        print("1 - Predict image")
        print("2 - Train again")
        opt = int(input())
        if opt == 1:
            opt = int(input(
                "1 - Predict a random image" + 
                "\n2 - Enter the path of some image"
            ))
            if opt == 1:
                train.predict() 
            else:
                print("Not avaible yet")
        elif opt ==2:
            print("Enter the parameters")
            bz = int(input("Batch Size: "))
            epochs = int(input("Epochs: "))
            verbose = int(input("Verbose:"))
            train.train_model(bz,  epochs , verbose)
    else:
        print("There is no model to load, trainning a model")
        train.main()

def main():
    print("Verifing if you need to train the model")
    print(".............")
    trained = False 
    if os.path.exists('brain_tumor.hdf5'):
        print("The model was already trained")
        trained = True
    while(True):
        menu(trained) 
if __name__ == "__main__":
    main()