import pandas as pd, numpy as np

class Node(object):
    def __init__(self,info,parent=None,right=None,left=None):
        """info should contain: index, cutting variable and cutting point, greatest decent value"""
        self.info=info
        self.parent=parent
        self.right=right
        self.left=left
    
    @property
    def get_info(self):
        return self.info
        
    @property    
    def get_parent(self):
        return self.parent
    
    @property
    def get_left(self):
        return self.left
    
    @property
    def get_right(self):
        return self.right
    
    def set_left(self,node):
        assert isinstance(node,Node)
        self.left=node
    
    def set_right(self,node):
        assert isinstance(node,Node)
        self.right=node
        
    def __repr__(self):
        return str(self.info)
        
class Tree(object):
    def __init__(self,root):
        assert isinstance(root,Node)
        self.root=root
        self.size=1

    def is_leaf(self,node):
        if node.get_left is None and node.get_right is None:
            return True
        else:
            return False
            
    def get_leaves(self):
        leaves=[]
        def helper(root):
            if self.is_leaf(root):
                leaves.append(root)
                return 
            else:
                if root.get_left is not None:
                    helper(root.get_left)
                if root.get_right is not None:
                    helper(root.get_right)              
        helper(self.root)
        return leaves
    
    def set_left(self,root,node):
        root.set_left(node)
        self.size+=1
    
    def set_right(self,root,node):
        root.set_right(node)
        self.size+=1
        
    def predict(self,newdata):
        """find which leaf does the input data belongs to and make prediction"""
        root=self.root
        while not self.is_leaf(root):
            print(root.get_info["var"])
            x=newdata[root.get_info["var"]]
            if root.get_info["dtype"]!="Continuous":
                if x in root.get_info["cut"]:
                    root=root.get_left
                else:
                    print(root.get_info["cut"])
                    root=root.get_right
            else:
                if x<=root.get_info["cut"]:
                    root=root.get_left
                    print("float left")
                else:
                    root=root.get_right
                    print("float right")
        return root.get_info["pred"]
           
    def print_var(self):
        """for every non-leaf node print out the splitting variable"""
        def helper(node):
            if self.is_leaf(node):
                return
            else:
                print(node.get_info["var"])
                helper(node.get_left)
                helper(node.get_right)
        
        helper(self.root)
        return 
    

class MaxHeap(object):
    """heap structure for holding all leaf node and the most possible
    node to be splited w.r.t the splite criteria"""
    
    def __init__(self,struct=None):
        if struct:
            self.struct=list(struct)
            self.size=len(self.struct)
            self.heapify()
        else:
            self.struct=[]
            self.size=len(self.struct)

    def parent_id(self,i):
        assert i<self.size
        if i%2==0:
            return int(i/2)-1
        else:
            return int(i/2)

    def children_id(self,i):
        assert i<self.size
        id=[]
        if (i+1)*2<self.size:
            id.append(i*2+1)
            id.append((i+1)*2)
        elif i*2+1==self.size-1:
            id.append(i*2+1)
        return id
            
    def swap(self,i,j):
        self.struct[i],self.struct[j]=self.struct[j],self.struct[i]
        
    def bubble_up(self,j):
        parentid=self.parent_id(j)
        while j>0 and self.struct[j].get_info["effect"]>self.struct[parentid].get_info["effect"]:
            self.swap(j,parentid)
            j=parentid
            parentid=self.parent_id(j)
        return 

    def _maxid(self,idx):
        """find the node with the largest key value given multiple input of index"""
        if len(idx)==0:
            return 
        else:
            maxid=idx[0]
            for i in idx:
                if self.struct[i].get_info["effect"]>self.struct[maxid].get_info["effect"]:
                    maxid=i
            return maxid

    def bubble_down(self,j):
        """let the node j to sink down to next layer of tree to preserve the heap structure"""
        childrenid=self.children_id(j)
        maxid=self._maxid(childrenid)
        while childrenid:
            if self.struct[j].get_info["effect"]>=self.struct[maxid].get_info["effect"]:
                break
            else:
                self.swap(j,maxid)
                j=maxid
                childrenid=self.children_id(j)
                maxid=self._maxid(childrenid)
        return 

    def _max_heapify(self,j):
        """heapify a subtree with root j"""
        childrenid=self.children_id(j)
        maxid=self._maxid(childrenid)
        if not childrenid:
            return 
        else:
            if self.struct[j].get_info["effect"]<self.struct[maxid].get_info["effect"]:
                self.swap(j,maxid)
                self._max_heapify(maxid)
            return 

    def heapify(self):
        if self.size<2:
            return
        else:
            idx=int(self.size/2)
            while idx>=0:
                self._max_heapify(idx)
                idx-=1
            return 

    @property
    def extract_max(self):
        assert self.size>0
        self.swap(0,self.size-1)
        maxnode=self.struct.pop()
        self.size-=1
        if self.size>0:
            self.bubble_down(0)
        return maxnode

    def insert(self,node):
        self.struct.append(node)
        self.size+=1
        self.bubble_up(self.size-1)
        
    
class Config(object):
    def __init__(self,minobv=50,maxleaves=4,var=None,family="Gaussian",LossFunction="LeastSquare",stop="CaseInNode"):
        self.family=family
        self.LossFunction=LossFunction
        self.stop=stop
        self.minobv=minobv
        self.var=var
        self.maxleaves=maxleaves
        
class CART(object):
    """Classification and Regression Tree, implemented by max_heap data structure and binary tree"""
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
                node.get_info["pred"]=self.target.loc[node.get_info["index"]].mean()
                return False
            else:
                node.get_info["effect"]=sprime["effect"]
                node.get_info["var"]=sprime["var"]
                node.get_info["cut"]=sprime["cut"]
                node.get_info["Lindex"]=sprime["Lindex"]
                node.get_info["Rindex"]=sprime["Rindex"]
                node.get_info["dtype"]=sprime["dtype"]
                return True

                
    def FindBestSplit(self,node):
        """find the best splitting variable and cut point of a node""" 
        if node.get_info["index"].size<=self.config.minobv:
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
        subindex=node.get_info["index"]
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
        """return the best split on arg if arg is discrete variable"""
        xarg=self.dataset.loc[node.get_info["index"],arg]
        categories=pd.Categorical(list(xarg))
        if categories.categories.size>1:
            source={i:[0,0] for i in categories.categories}
            idxalloc={i:[] for i in categories.categories}
            for i in xarg.index:
                cate=xarg.loc[i]
                source[cate][0]+=1
                source[cate][1]+=self.target.loc[cate]
                idxalloc[cate].append(i)
            for i in source:
                source[i].append(source[i][1]/source[i][0])
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
        leftnode=Node({"index":node.get_info["Lindex"]},node)
        rightnode=Node({"index":node.get_info["Rindex"]},node)  
        self.tree.set_left(node,leftnode)
        self.tree.set_right(node,rightnode)
        return leftnode,rightnode
        
    def Part(self):
        root=Node({"index":self.dataset.index})
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
            if "pred" not in leaf.get_info:
                leaf.get_info["pred"]=self.target.loc[leaf.get_info["index"]].mean()
        return
    
    def Predict(self,newdata):
        assert newdata.shape[1]==self.dataset.shape[1]
        newdata=newdata.copy()
        newdata.columns=pd.RangeIndex(0,newdata.shape[1])
        newdata.index=pd.RangeIndex(0,newdata.shape[0])
        pred=[self.tree.predict(newdata.loc[i]) for i in newdata.index]
        return pred
    
        
        
if __name__=="__main__":
    from sklearn.datasets import load_boston

    boston=load_boston()
    data=pd.DataFrame(boston.data)
    data[8]=data[8].astype('int32').astype('category')
    data[3]=data[3].astype('int32').astype('category')
    target=pd.Series(boston.target)
    config=Config(10,10)
    rtree=CART(target,data,config)
    rtree.Part() 
    newdata=data
    print(rtree.Predict(newdata))
    root=rtree.tree.get_leaves()[0].get_parent