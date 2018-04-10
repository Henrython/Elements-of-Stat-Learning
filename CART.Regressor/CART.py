import pandas as pd, numpy as np
from HeapAndQueue import *

    
        
class Config(object):
    def __init__(self,minobv=50,maxleaves=4,var=None,stop="CaseInNode"):
        self.stop=stop
        self.minobv=minobv
        self.var=var
        self.maxleaves=maxleaves
        
class CART(object):
    """Classification and Regression Tree, implemented by max_heap data structure and binary tree.
       Only implemented square error loss and Gini index impurity, other loss could also be added by user, but would have to modify
       part of the code"""
    def __init__(self,target,dataset,Config):
        self.dataset=dataset.copy()
        self.target=target
        self.dataset.columns=pd.RangeIndex(0,self.dataset.shape[1])
        self.dataset.index=pd.RangeIndex(0,self.dataset.shape[0])
        self.config=Config
        if self.config.var is None:
            self.config.var=self.dataset.columns
            
    def TestSplit(self,node):
        """test if the node can be splitted. If yes, find the optimal splitting 
        variable and point and make an update to the node about how much it will 
        decrease the lossfunction"""
        if self.config.stop=="CaseInNode":
            sprime=self.FindBestSplit(node)
            if sprime is None:
                node.set_pred(self.target.loc[node.index].mean())
                return False
            else:  
                node.set_info(**sprime)
                return True

                
    def FindBestSplit(self,node):
        """find the best splitting variable and cut point of a node""" 
        if node.index.size<=self.config.minobv:
            return 
        else:
            varprime,cutprime,effectprime,dtypeprime=None,None,None,None
            for arg in self.config.var:
                if self.dataset[arg].dtype == "float":
                    effect,cut,newLindex,newRindex=self.ContinuousSplite(node,arg)# effect is the decent in loss caused by arg, cut is the best cut point, idx is the 
                    dtype="Continuous"
                else:
                    test_outcome=self.DiscreteSplite(node,arg)
                    if test_outcome == None:
                        continue
                    else:
                        effect,cut,newLindex,newRindex=test_outcome
                        dtype="Categorical"
                if varprime==None or effect>effectprime:
                    varprime,cutprime,effectprime,Lindex,Rindex,dtypeprime=arg,cut,effect,newLindex,newRindex,dtype
            if varprime!=None:            
                sprime={"effect":effectprime,"var":varprime,"cut":cutprime,"Lindex":Lindex,"Rindex":Rindex,"dtype":dtypeprime}
                return sprime
            else:
                return 
                
    def ContinuousSplite(self,node,arg):
        """arg is a continuous variable, return the best cut point and effect of splitting on arg"""
        subindex=node.index
        nt=subindex.size
        subset=self.dataset.loc[subindex,arg]
        subset.sort_values(inplace=True)
        subtarget=self.target.loc[subset.index]
        bestcut=subset.iloc[[0,1]].mean()
        i=partpoint=1
        maxeffect=(subtarget.iloc[0]**2)/nt+((subtarget.iloc[1:].sum())**2)/(nt*(nt-1))
        sl,sr=subtarget.iloc[0],subtarget.iloc[1:].sum()
        while i<=nt-2:
            sl+=subtarget.iloc[i]
            sr-=subtarget.iloc[i]
            if subset.iloc[i+1]>subset.iloc[i]:
                effect=((sl**2)/(i+1)+(sr**2)/(nt-i-1))/nt
                if effect>maxeffect:
                    bestcut=subset.iloc[[i,i+1]].mean()
                    maxeffect=effect
                    partpoint=i
            i+=1
        Lindex=subset.index[:partpoint+1]
        Rindex=subset.index[partpoint+1:]
        return maxeffect,bestcut,Lindex,Rindex
    
    def DiscreteSplite(self,node,arg):
        """return the best split on arg if arg is discrete (categorical) variable"""
        xarg=self.dataset.loc[node.index,arg]
        categories=pd.Categorical(list(xarg))
        if categories.categories.size>1:
            source={i:[0,0,0] for i in categories.categories}
            idxalloc={i:[] for i in categories.categories}
            for i in xarg.index:
                cate=xarg.loc[i]
                source[cate][0]+=1
                source[cate][1]+=self.target.loc[cate]
                source[cate][2]=source[cate][1]/source[cate][0]
                idxalloc[cate].append(i)
                assert len(idxalloc[cate])==source[cate][0]
            source=pd.DataFrame(source)
            source.sort_values(by=2,axis=1,inplace=True)
            sl=source.iloc[1,0]
            sr=source.iloc[1,1:].sum()
            maxeffect=(sl)**2/source.iloc[0,0]+(sr)**2/source.iloc[0,1:].sum()
            bestcut=1
            for i in range(1,source.columns.size-1):
                sl+=source.iloc[1,i]
                sr-=source.iloc[1,i]
                effect=sl**2/source.iloc[1,:i+1].sum()+sr**2/source.iloc[1,i+1:].sum()
                if effect>maxeffect:
                    maxeffect=effect
                    bestcut=i
            Lcolumns=source.columns[:bestcut]
            Rcolumns=source.columns[bestcut:]
            Lindex,Rindex=[],[]
            for i in Lcolumns:
                Lindex+=idxalloc[i]
            for i in Rcolumns:
                Rindex+=idxalloc[i]
            return maxeffect,Lcolumns,pd.Int64Index(Lindex),pd.Int64Index(Rindex)
        else:
            return 
            
    def Splite(self,node):
        leftnode=Node(node.Lindex,node)
        rightnode=Node(node.Rindex,node)  
        self.tree.set_left(node,leftnode)
        self.tree.set_right(node,rightnode)
        return leftnode,rightnode
        
    def Part_Max(self):
        """choose the node which cause the largest descent to to splite"""
        root=Node(self.dataset.index)
        self.tree=Tree(root)
        leaves=1
        self.heap=MaxHeap()
        if self.TestSplit(root):
            self.heap.insert(root)
        while self.heap.size!=0 and leaves<self.config.maxleaves:
            priority=self.heap.extract_max
            lnode,rnode=self.Splite(priority)
            leaves+=1
            if self.TestSplit(lnode):
                self.heap.insert(lnode)
            if self.TestSplit(rnode):
                self.heap.insert(rnode)
        for leaf in self.tree.get_leaves():
            if leaf.pred == None:
                leaf.set_pred(self.target.loc[leaf.index].mean())
        return
    
    def Part_BFS(self):
        """Doing breadth first search to build up the tree"""
        root=Node(self.dataset.index)
        self.queue=Dequeue()
        self.tree=Tree(root)
        leaves=1
        if self.TestSplit(root):
            self.queue.Inqueue(root)
        while self.queue.size!=0 and leaves<self.config.maxleaves:
            priority=self.queue.Dequeue()
            lnode,rnode=self.Splite(priority)
            leaves+=1
            if self.TestSplit(lnode):
                self.queue.Inqueue(lnode)
            if self.TestSplit(rnode):
                self.queue.Inqueue(rnode)
        for leaf in self.tree.get_leaves():
            if leaf.pred == None:
                leaf.set_pred(self.target.loc[leaf.index].mean())
        return 
    
    def Fit(self,search="max"):
        """fit the tree using max search or breadth first search"""
        if search == "max":
            self.Part_Max()
        elif search == "bfs":
            self.Part_BFS()
        else:
            print("invalid search method!")
        return
        
    def Predict(self,newdata):
        assert newdata.shape[1]==self.dataset.shape[1]
        newdata=newdata.copy()
        newdata.columns=pd.RangeIndex(0,newdata.shape[1])
        newdata.index=pd.RangeIndex(0,newdata.shape[0])
        pred=[self.tree.predict(newdata.loc[i]) for i in newdata.index]
        return pred
    
        
        

