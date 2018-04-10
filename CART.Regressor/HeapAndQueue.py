# MaxHeap and Dequeue

class Node(object):
    def __init__(self,index,parent=None,left=None,right=None,**karg):
        """info should contain: index, cutting variable and cutting point, greatest decent value"""
        self.index=index
        self.parent=parent
        self.right=right
        self.left=left
        if karg:
            self.effect=karg["effect"]
            self.var=karg["var"]
            self.cut=karg["cut"]
            self.Lindex=karg["Lindex"]
            self.Rindex=karg["Rindex"]
            self.dtype=karg["dtype"]
        self.pred=None
        self.prev=None
        self.next=None
        
    @property
    def get_info(self):
        info={"index":self.index,"effect":self.effect,"var":self.var,"cut":self.cut,"Lindex":self.Lindex,"Rindex":self.Rindex}
        print(info)
        
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
    
    def set_info(self,**karg):
        """set the node info with input dictionary"""
        self.effect=karg["effect"]
        self.var=karg["var"]
        self.cut=karg["cut"]
        self.Lindex=karg["Lindex"]
        self.Rindex=karg["Rindex"]
        self.dtype=karg["dtype"]
        return 
    
    def set_pred(self,pred):
        self.pred=pred
        return 
    
    def set_next(self,node):
        self.next=node
        
    def set_prev(self,node):
        self.prev=node
        
    def __repr__(self):
        if self.pred == None:
            return str({"var":self.var,"cut":self.cut})
        else:
            return str({"pred":self.pred})
    
    
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
            x=newdata[root.var]
            if root.dtype!="Continuous":
                if x in root.cut:
                    root=root.get_left
                else:
                    root=root.get_right
            else:
                if x<=root.cut:
                    root=root.get_left
                else:
                    root=root.get_right
        return root.pred
           
    def print_var(self):
        """for every non-leaf node print out the splitting variable"""
        def helper(node):
            if self.is_leaf(node):
                return
            else:
                print(node.var)
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
        while j>0 and self.struct[j].effect>self.struct[parentid].effect:
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
                if self.struct[i].effect>self.struct[maxid].effect:
                    maxid=i
            return maxid

    def bubble_down(self,j):
        """let the node j to sink down to next layer of tree to preserve the heap structure"""
        childrenid=self.children_id(j)
        maxid=self._maxid(childrenid)
        while childrenid:
            if self.struct[j].effect>=self.struct[maxid].effect:
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
            if self.struct[j].effect<self.struct[maxid].effect:
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

class Dequeue(object):
    def __init__(self,head=None):
        self.head=head
        self.tail=head
        if head == None:
            self.size=0
        else:
            self.size=1
            
    def Inqueue(self,node):
        """left insert the node"""
        if self.size>0:
            node.set_next(self.head)
            self.head.set_prev(node)
            self.head=node
        else:
            self.head=self.tail=node
        self.size+=1
        return 
        
    def Dequeue(self):
        """right pop out the node"""
        if self.size>=1:
            node=self.tail
            if self.size>1:
                self.tail=self.tail.prev
                self.tail.next.set_prev(None)
                self.tail.set_next(None)
            elif self.size==1:
                self.head=self.tail=None
            self.size-=1            
            return node
        else:
            print("empty queue!")
            return 