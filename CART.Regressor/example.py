from CART import *
from sklearn.datasets import load_boston

if __name__=="__main__":   
    boston=load_boston()
    data=pd.DataFrame(boston.data)
    data[8]=data[8].astype('int32').astype('category')
    data[3]=data[3].astype('int32').astype('category')
    target=pd.Series(boston.target)
    config=Config(20,1000)
    rtree=CART(target,data,config)
    rtree.Fit("bfs")
    newdata=data
    print(rtree.Predict(newdata))