# EM algo for simple mixture gaussian model

data1=c(-0.39, 0.12, 0.94, 1.67, 1.76, 2.44, 3.72, 4.28, 4.92, 5.53,
0.06, 0.48, 1.01, 1.68, 1.80, 3.25, 4.12, 4.60, 5.28, 6.22)

hist(data1)

npdf=function(x,u,sigma)
{
  par1=exp(-((x-u)**2)/(2*sigma))
  par2=1/(sqrt(2*pi*sigma))
  return (par1*par2)
}

EStep=function(dat,initial_val)
{
  u1=initial_val[1]
  u2=initial_val[2]
  sigma1=initial_val[3]
  sigma2=initial_val[4]
  phi=initial_val[5]
  
  n=length(dat)
  respons=rep(0,n)
  for (i in c(1:n))
  {
    denom=phi*npdf(dat[i],u1,sigma1)
    numra=denom+(1-phi)*npdf(dat[i],u2,sigma2)
    respons[i]=denom/numra
  }
  return (respons)
}

MStep=function(dat,respons)
{
  n=length(dat)
  u1=sum(dat*respons)/sum(respons)
  u2=sum(dat*(1-respons))/sum(1-respons)
  sigma1=sum(respons*(dat-u1)**2)/sum(respons)
  sigma2=sum((1-respons)*(dat-u2)**2)/sum(1-respons)
  phi=sum(respons)/n
  return (c(u1,u2,sigma1,sigma2,phi))
}

EM=function(dat,initial_val,converge=1e-5)
{
  continue=TRUE
  argmax=initial_val
  while(continue)
  {
    old_phi=argmax[5]
    respons=EStep(dat,argmax)
    argmax=MStep(dat,respons)
    phi=argmax[5]
    if (abs(phi-old_phi)<converge){continue=FALSE}
  }
  return (argmax)
}

initializer=function(dat)
{
  u1=sample(dat,1)
  u2=sample(dat,1)
  sigma1=sigma2=var(dat)
  phi=0.5
  return (c(u1,u2,sigma1,sigma2,phi))
}

initial_val=initializer(data1)
res=EM(data1,initial_val)

visualizer=function(dat,argmax)
{
  u1=argmax[1]
  u2=argmax[2]
  sigma1=argmax[3]
  sigma2=argmax[4]
  phi=argmax[5]
  
  p=seq(min(dat)-1,max(dat)+1,0.05)
  
  hist(dat,freq = FALSE,ylim = c(0,0.5))
  lines(p,npdf(p,u1,sigma1),type = 'l',col='red')
  lines(p,npdf(p,u2,sigma2),col='blue')
  lines(p,phi*npdf(p,u1,sigma1)+(1-phi)*npdf(p,u2,sigma2),col="green")
}

visualizer(data1,res)

