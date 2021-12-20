from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os 
import os.path as osp
import json 
import numpy as np

from utils.constants import Dataset




        
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
                                            d6=d5[key4]                            
                                    #print(key +"  "+ key1 +"  "+ key2+"  "+key3)    
                                            if accuracy== "val_accs":
                                                accuracyLabel= "validation accuracy"
                                            elif accuracy =="test_accs":
                                                accuracyLabel= "test accuracy"
                                            elif accuracy== "train_accs ":
                                                accuracyLabel= "trainings accuracy"
                                            else:
                                                accuracyLabel= accuracy
                                    
                                    title= "Dataset: "+ dataset +"   Model: "+key1+ "  Aggregation: "+key2+ " NumberOfLayers: "+key3+"  "+accuracyLabel
                                    valAtr = list(map(float,d6))
                                    karray=[]
                                    for x in range(len(valAtr)):
                                        karray.append(x+1)    
                                    plot(valAtr,karray, title, accuracyLabel, "k-Value")                        
    os.chdir(cwd)


def getAccuraciesScatter(dataset, model,aggregation,NumOfLayers, accuracy ):
    newDir = ".\\results\\"+dataset
    cwd =os.getcwd()
    os.chdir(newDir)
    position=330
    layerDict= {"NumOfLayers_2":"2 Layers","NumOfLayers_3":"3 Layers","NumOfLayers_4":"4 Layers"}
    plt.figure(figsize=(20,20))
    with open('CompressedResults.json') as f:
        JasonFile = json.loads(f.read())
        for DataSetName in JasonFile:
            JasonFileDataset=JasonFile[DataSetName]
            for ModelName in JasonFileDataset:
                if model== "all" or ModelName in model:
                    JasonFileModel=JasonFileDataset[ModelName]
                    for AggregationName in JasonFileModel:
                        if aggregation== "all" or AggregationName in aggregation:
                            JasonFileAggregation=JasonFileModel[AggregationName]
                            NoL=[]
                            KVal=[]
                            acc=[]
                            position=position+1
                            for Layers in JasonFileAggregation:
                                if NumOfLayers == "all" or Layers in NumOfLayers:
                                    JasonFileLayers=JasonFileAggregation[Layers]
                                    for accuracyName in JasonFileLayers:
                                        if accuracyName == accuracy:
                                            JasonFIleAccuracy=JasonFileLayers[accuracyName]                           
                                             #print(key +"  "+ key1 +"  "+ key2+"  "+key3)                            
                                            valAtr = list(map(float,JasonFIleAccuracy))
                                            if AggregationName == 'NONE':
                                                stellen=4
                                            else:
                                                stellen=3
                                            if len(valAtr)<stellen:
                                                stellen=len(valAtr)                                    
                                            for x in range(stellen):
                                            #for x in range(len(valAtr)):
                                                if AggregationName == "NONE":
                                                    KVal.append(x+1)
                                                else:
                                                    KVal.append(x+2)
                                                acc.append(valAtr[x])
                                                NoL.append(layerDict[Layers])
                            title= DataSetName + "   " + ModelName + "   " + AggregationName
                            
                            plt.subplot(position,xlabel="Neighbourdegree",ylabel="Test accuracy",title=title)
                            plt.locator_params(axis='x',integer=True )
                            scatterplot(NoL, KVal, acc)
                            #heatplot(NoL,KVal,acc)
                                                       
    os.chdir(cwd)
    os.chdir(".\\savedPlots\\")
    plt.savefig(dataset+'--'+model+'--scatter-layer-degree.png')
    os.chdir(cwd)
    
    
def getKBranchScatter(dataset,model):
    newDir = ".\\results\\"+dataset
    cwd =os.getcwd()
    os.chdir(newDir)
    position=330
    aggDict= {"NONE":"1-Branch","MEAN":"k-Branches; Mean aggregation","MAX":"k-Branches; Maximum aggregation"}
    plt.figure(figsize=(20,20))
    with open('CompressedResults.json') as f:
        JasonFile = json.loads(f.read())
        for DataSetName in JasonFile:
            JasonFileDataset=JasonFile[DataSetName]
            for ModelName in JasonFileDataset:
                if model== "all" or ModelName in model:
                    JasonFileModel=JasonFileDataset[ModelName]
                    agg=[]
                    KVal=[]
                    acc=[]
                    position=position+1
                    for AggregationName in JasonFileModel:
                            JasonFileAggregation=JasonFileModel[AggregationName]
                            for Layers in JasonFileAggregation:
                                if Layers == "NumOfLayers_2":
                                    JasonFileLayers=JasonFileAggregation["NumOfLayers_2"]
                                    JasonFIleAccuracy=JasonFileLayers["test_accs"]
                                    valAtr = list(map(float,JasonFIleAccuracy))
                                    if AggregationName == 'NONE':
                                        stellen=4
                                    else:
                                        stellen=3
                                    if len(valAtr)<stellen:
                                        stellen=len(valAtr)                                    
                                    for x in range(stellen):
                                        if AggregationName == "NONE":
                                            KVal.append(x+1)
                                        else:
                                            KVal.append(x+2)
                                        acc.append(valAtr[x])
                                        agg.append(aggDict[AggregationName])
                    title= DataSetName+"   "+ModelName+"   "+ "2 Layers"
                    if acc:
                        plt.subplot(position,xlabel="Neighbourdegree",ylabel="Test accuracy",title=title)
                        plt.locator_params(axis='x',integer=True )
                        scatterplot(agg, KVal, acc)               
    os.chdir(cwd)
    os.chdir(".\\savedPlots\\")
    plt.savefig(dataset+'--'+model+'--branch-scatter-layer-degree.png')
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


def scatterplot(NumberofLayers, kValues, accuracyValues):
    #size=[]
    colors=[]
    classes = []
    for x in range(len(NumberofLayers)):
        #size.append(50)
        if NumberofLayers[x] not in classes:
            classes.append(NumberofLayers[x])
        colors.append(classes.index(NumberofLayers[x]))
    Mapcolors = ListedColormap(['r','g','b','y'])
    scatter = plt.scatter(kValues,accuracyValues,c=colors,cmap=Mapcolors,)
    lowerbound= min(accuracyValues)*0.9
    upperbound= max(accuracyValues)+(1-max(accuracyValues))/3
    plt.ylim(lowerbound,upperbound)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc="lower left")
    
    #fig, ax= plt.subplots()
    #scatter= ax.scatter(kValues,accuracyValues,s=size,c=colors)
    #legend= ax.legend(*scatter.legend_elements(),loc="lower left", title= "Models")
    #ax.add_artist(legend)
    #plt.show
    
    
def heatplot(NumberofLayers, kValues, accuracyValues):
    layers=[]
    #matrix ={}
    matrix=[[]]
    kList=[]
    
    for x in range(len(NumberofLayers)):
        if NumberofLayers[x]not in layers:
            layers.append(NumberofLayers[x])
            matrix.append([])
        if kValues[x] not in kList:
            kList.append(kValues[x])
        #matrix[models.index(modelArray[x]),kValues[x]-1]=accuracyValues[x]
        matrix[layers.index(NumberofLayers[x])].append(accuracyValues[x])
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
    ax.set_yticks(np.arange(len(layers)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(kList)
    ax.set_yticklabels(layers)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(layers)):
        for j in range(len(kList)):
            text = ax.text(j, i, round(npArray[i, j],2),
                        ha="center", va="center", color="k")
    



    ax.set_title("Validation accuracy in different Models")
    fig.tight_layout()
    plt.show()
    
    