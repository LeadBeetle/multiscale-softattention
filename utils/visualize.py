from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os 
import os.path as osp
import json 
import numpy as np




        
def getAccuracies(dataset, model,aggregation,NumOfLayers, accuracy ):
    newDir = ".\\results\\"+dataset
    cwd =os.getcwd()
    os.chdir(newDir)
    with open('CompressedResults.json') as f:
        d = json.loads(f.read())
        for key in d:
            d2=d[key]
            for key1 in d2:
                if model== "all" or model == key1:
                    d3=d2[key1]
                    for key2 in d3:
                        if aggregation== "all" or aggregation == key2:
                            d4=d3[key2]
                            for key3 in d4:
                                if NumOfLayers== "all" or NumOfLayers == key3:
                                    d5=d4[key3]
                                    for key4 in d5:
                                        if key4 == accuracy:
                                            d5=d4[key4]                            
                                    #print(key +"  "+ key1 +"  "+ key2+"  "+key3)    
                                            if accuracy== "val_accs":
                                                accuracyLabel= "validation accuracy"
                                            elif accuracy =="test_accs":
                                                accuracyLabel= "validation accuracy"
                                            elif accuracy== "train_accs ":
                                                accuracyLabel= "trainings accuracy"
                                            else:
                                                accuracyLabel= accuracy
                                    
                                    title= "Dataset: "+ dataset +"   Model: "+key1+ "  Aggregation: "+key2+ " NumberOfLayers: "+key3+"  "+accuracyLabel
                                    valAtr = list(map(float,d5))
                                    karray=[]
                                    for x in range(len(valAtr)):
                                        karray.append(x+1)    
                                    plot(valAtr,karray, title, accuracyLabel, "k-Value")                           
    os.chdir(cwd)


def getAccuraciesScatter(dataset, model, matrix, accuracy ):
    newDir = ".\\results\\"+dataset
    cwd =os.getcwd()
    os.chdir(newDir)
    mdl=[]
    kVal=[]
    acc=[]
    if accuracy== "val_accs":
        accuracyLabel= "validation accuracy"
    elif accuracy =="test_accs":
        accuracyLabel= "validation accuracy"
    elif accuracy== "train_accs ":
        accuracyLabel= "trainings accuracy"
    else:
        accuracyLabel= key3    
    with open('CompressedResults.json') as f:
        d = json.loads(f.read())
        for key in d:
            d2=d[key]
            for key1 in d2:
                if model== "all" or key1 in model:
                    d3=d2[key1]
                    for key2 in d3:
                        if matrix== "all" or key2 in matrix:
                            d4=d3[key2]
                            for key3 in d4:
                                if key3 == accuracy:
                                    d5=d4[key3]                            
                                    #print(key +"  "+ key1 +"  "+ key2+"  "+key3)    

                                    
                                    
                                    valAtr = list(map(float,d5))
                                    for x in range(len(valAtr)):
                                        kVal.append(x+1)
                                        acc.append(valAtr[x])
                                        mdl.append(key1)    
    scatterplot(mdl,kVal,acc)  
    heatplot(mdl, kVal, acc)                                                        
    os.chdir(cwd)


def plot (valArr, descArr, title, yAxisDenom, xAxisDenom):

    #print(karray)
    plt.figure(figsize=(len(valArr), 3))
    plt.bar(descArr, valArr)
    plt.title(title)
    plt.xlabel(xAxisDenom)
    plt.ylabel(yAxisDenom)
    plt.ylim(0.6, 0.99)
    #plt.yscale('log')


def scatterplot(modelArray, kValues, accuracyValues):
    size=[]
    colors=[]
    classes = ["GATV1","GATV2","TRANS","Unknown"]
    for x in range(len(modelArray)):
        size.append(50)
        if modelArray[x] == "GATV1":
               colors.append(1)
        elif modelArray[x] == "GATV2":
            colors.append(2)
        elif modelArray[x] == "TRANS":
            colors.append(3)
        else:
            colors.append(10)
    colours = ListedColormap(['r','g','b','y'])
    scatter = plt.scatter(kValues,accuracyValues,c=colors,cmap=colours)
    plt.ylim(0,1)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc="lower left")
    
    #fig, ax= plt.subplots()
    #scatter= ax.scatter(kValues,accuracyValues,s=size,c=colors)
    #legend= ax.legend(*scatter.legend_elements(),loc="lower left", title= "Models")
    #ax.add_artist(legend)
    #plt.show
    
    
def heatplot(modelArray, kValues, accuracyValues):
    models=[]
    #matrix ={}
    matrix=[[]]
    kList=[]
    for x in range(len(modelArray)):
        if modelArray[x]not in models:
            models.append(modelArray[x])
            matrix.append([])
        if kValues[x] not in kList:
            kList.append(kValues[x])
        #matrix[models.index(modelArray[x]),kValues[x]-1]=accuracyValues[x]
        matrix[models.index(modelArray[x])].append(accuracyValues[x])
        #print(models.index(modelArray[x]))
        #print(kValues[x])
        #print(accuracyValues[x])
    matrix= matrix[:-1]
    print(matrix)
    #hier startet die Heatmap Generation
    npArray = np.array(matrix)     
    fig, ax = plt.subplots()
    im = ax.imshow(npArray)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(kList)))
    ax.set_yticks(np.arange(len(models)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(kList)
    ax.set_yticklabels(models)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(kList)):
            text = ax.text(j, i, round(npArray[i, j],2),
                        ha="center", va="center", color="k")
    



    ax.set_title("Validation accuracy in different Models")
    fig.tight_layout()
    plt.show()
    
    