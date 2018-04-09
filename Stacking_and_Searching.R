library(MASS)
library(ISLR)
library(glmnet)
library(CVXR)

set.seed(123)
data0=Auto
data0$cylinders=as.factor(data0$cylinders)
data1=data0[,1:6]
idx=sample(seq(1,dim(data1)[1]),300)
train.x=model.matrix(mpg~.,data1[idx,])
train.y=data1[idx,1]
test.x=model.matrix(mpg~.,data1[-idx,])
test.y=data1[-idx,1]

#--------bayes average------------------------------------------------
  #average the model using posterior prob (BIC approximated)
  get_bic=function(model){
    dev=(1-model$dev.ratio)*model$nulldev
    bic=model$nobs*log(dev/model$nobs)+model$df*log(model$nobs)
    return (bic)
  }
  
  Bayes.ave=function(x,y,alpha,lambdas){
    n=length(lambdas)
    init=glmnet(x,y,family = "gaussian",alpha=alpha,lambda = lambdas[1])
    model.cache=rep(list(init),n)
    model.cache[[1]]$bic=get_bic(init)
    for (i in 2:n){
      model=glmnet(x,y,family = "gaussian",alpha=alpha,lambda = lambdas[i])
      model.cache[[i]]=model
      model.cache[[i]]$bic=get_bic(model)
    }
    return (model.cache)
  }
  
  # constants:
  lambdas=seq(0,10,1)
  mymodels=Bayes.ave(train.x,train.y,0.05,lambdas)  
  
  prediction.bayes=function(x,stack){
    #posterior prob=exp{-0.5*bic_i}/sum(exp{-0.5*bic_i}
    bic_sum=0
    for (i in stack){
      bic_sum=bic_sum+exp(-0.5*i$bic)
    }
    pred=rep(0,dim(x)[1])
    for (i in 1:length(stack)){
      pred=pred+predict.glmnet(stack[[i]],newx=x)*(exp(-0.5*stack[[i]]$bic)/bic_sum)
    }
    return (pred)
  }
  
  grid.glm=cv.glmnet(train.x,train.y,lambda=lambdas,alpha=0.05,type.measure = "deviance",nfolds = 4)
  
  pred1=prediction.bayes(test.x,mymodels)
  mse1=mean((test.y-pred1)**2)
  
  glm0=glmnet(train.x,train.y,lambda = grid.glm$lambda.min,alpha = 0.05)
  pred0=predict.glmnet(glm0,newx = test.x )
  mse0=mean((test.y-pred0)**2)
  
  print(paste("single mse is:",mse0))
  print(paste("stacked mse is:",mse1))
  
  
  plot(test.y)
  points(pred0,col="red")
  points(pred1,col="green")
  
#---------------frequentist ave-----------------------------
  # stack the models in frequentist method
  # to be more accurate, we could restrict the weight to be >0 and <1 and sum to 1.
  weight.freq=function(x,y,stack){
    n=length(stack)
    new.design=matrix(data=0, nrow = dim(x)[1], ncol = length(stack))
    
    for (i in 1:length(stack)){
      new.design[,i]=predict.glmnet(stack[[i]],newx = x)
    }
    
    ensembler=lm(y~0+new.design)
    weight=ensembler$coefficients
    weight[is.na(weight)]=0
    return (weight)
  }
  
  weight.freq.modify=function(x,y,stack){
    n=length(stack)
    nob=dim(x)[1]
    nfeat=dim(x)[2]
    new.design=matrix(data=0, nrow = dim(x)[1], ncol = length(stack))
    for (i in 1:length(stack)){
      new.design[,i]=predict.glmnet(stack[[i]],newx = x)
    }
    Beta=Variable(n)
    Constr=list(Beta>=0,Beta<=1,sum_entries(Beta)==1)
    Obj=sum_squares(y-new.design %*% Beta)
    prob=Problem(Minimize(Obj),Constr)
    solution=solve(prob)
    print(solution$getValue(Beta))
    return (solution$getValue(Beta))
  }
  
  prediction.freq=function(x,weights,stack){
    
    new.desig=new.design=matrix(data=0, nrow = dim(x)[1], ncol = length(stack))
    
    for (i in 1:length(stack)){
      new.desig[,i]=predict.glmnet(stack[[i]],newx = x)
    }
    
    weights=matrix(weights,ncol = 1)
    
    pred=new.desig%*%weights
    
    return (pred)
  }
  
  weights=weight.freq.modify(test.x,test.y,mymodels)
  pred2=prediction.freq(test.x,weights,mymodels)
  
  mse3=mean((test.y-pred2)**2)
  
  print(paste("mse of frequentist method: ",mse3))
  
#-----------------------bumping-------------------------------
  
  model.boostrap=function(y,x,size){
    model.cache=rep(list(glmnet(x,y,alpha=0.05,lambda = 0.1)),size)
    for (i in 2:size){
      idx=sample(1:length(y),length(y),replace = TRUE)
      y.bp=y[idx]
      x.bp=x[idx,]
      model=glmnet(x.bp,y.bp,alpha = 0.05,lambda = 0.1)
      model.cache[[i]]=model
    }
    return (model.cache)
  }
  
  bumping=function(y,x,stack){
    opt=1
    pred.opt=predict.glmnet(stack[[opt]],newx = x)
    mse.opt=mean((y-pred.opt)**2)
    for (i in 2:length(stack)){
      pred=predict.glmnet(stack[[i]],newx= x)
      mse=mean((y-pred)**2)
      if (mse.opt>mse){
        opt=i
        mse.opt=mse
        print(mse.opt)
      }
    }
    print(opt)
    return (stack[[opt]])
  }
  
  
  Beta=Variable(9)
  #Constr=list(Beta>=0,Beta<=1,sum_entries(Beta)==1)
  Obj=sum((train.y-matrix(train.x,nrow = 300) %*% Beta)**2)
  prob=Problem(Minimize(Obj))
  solution=solve(prob)
  print(solution$Beta)
  