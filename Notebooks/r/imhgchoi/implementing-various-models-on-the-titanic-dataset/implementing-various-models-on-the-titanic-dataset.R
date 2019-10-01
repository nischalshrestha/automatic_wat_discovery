data <- read.csv("../input/train.csv")

question <- read.csv("../input/test.csv")
summary(data)
str(data)
summary(question)
str(question)
  data <- data[c('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked')]

  data <- data[data$Embarked != '',]

  data$Embarked <- as.character(data$Embarked)

  data$Embarked <- as.factor(data$Embarked)
par(mfrow=c(2,2))

counts <- table(data$Survived)

barplot(counts, main='Survived', names.arg=c('no','yes'), col ='lightpink')

counts <- table(data$Pclass)

barplot(counts, main='PassengerClass', names.arg=c('1st','2nd','3rd'), col='lemonchiffon')

counts <- table(data$Sex)

barplot(counts, main='Gender', names.arg=c('female','male'), col='lemonchiffon')

counts <- table(data$Embarked)

barplot(counts, main='Port of Embarkation', names.arg=c('C','Q','S'), col='lemonchiffon')
par(mfrow=c(2,2))

hist(data$Age)

hist(data$SibSp)

hist(data$Parch)

hist(data$Fare)
age.data <- data[!is.na(data$Age),]
par(mfrow=c(1,1))

plot(age.data$Embarked,age.data$Age, main='Age and Embarked Port')
v<- c(mean(age.data$Age[age.data$Pclass==1]),

      mean(age.data$Age[age.data$Pclass==2]),

      mean(age.data$Age[age.data$Pclass==3]))

barplot(v,main='Age and Class',names.arg=c('1st','2nd','3rd'), col='lemonchiffon')
v<- c(mean(age.data$Age[age.data$SibSp==0]),

      mean(age.data$Age[age.data$SibSp==1]),

      mean(age.data$Age[age.data$SibSp==2]),

      mean(age.data$Age[age.data$SibSp==3]),

      mean(age.data$Age[age.data$SibSp==4]),

      mean(age.data$Age[age.data$SibSp==5]))

par(mfrow=c(1,2))

barplot(v,main='Age and Sibling/Spouse',names.arg=c(0,1,2,3,4,5), col='lemonchiffon')

hist(data$SibSp)
v<- c(mean(age.data$Age[age.data$Parch==0]),

      mean(age.data$Age[age.data$Parch==1]),

      mean(age.data$Age[age.data$Parch==2]),

      mean(age.data$Age[age.data$Parch==3]),

      mean(age.data$Age[age.data$Parch==4]),

      mean(age.data$Age[age.data$Parch==5]),

      mean(age.data$Age[age.data$Parch==6]))

par(mfrow=c(1,2))

barplot(v,main='Age and Parent/Children',names.arg=c(0,1,2,3,4,5,6), col='lemonchiffon')

hist(data$Parch)
age.data$FamSize <- age.data$SibSp + age.data$Parch

v<- c(mean(age.data$Age[age.data$FamSize==0]),

      mean(age.data$Age[age.data$FamSize==1]),

      mean(age.data$Age[age.data$FamSize==2]),

      mean(age.data$Age[age.data$FamSize==3]),

      mean(age.data$Age[age.data$FamSize==4]),

      mean(age.data$Age[age.data$FamSize==5]),

      mean(age.data$Age[age.data$FamSize==6]),

      mean(age.data$Age[age.data$FamSize==7]))

barplot(v,main='Age and Family Size',names.arg=c(0,1,2,3,4,5,6,7), col='lemonchiffon')
data$FamSize <- data$SibSp + data$Parch
plot(age.data$Age,age.data$Fare)

sprintf('correlation coefficient : %s',cor(age.data$Age, age.data$Fare))
set.seed(0)

smplSize <-floor(nrow(age.data) * .75)

index <- sample(seq_len(nrow(age.data)), size = smplSize)

age.data.train <- age.data[index, ]

age.data.test <- age.data[-index, ]

#linear regression for age

age.reg <- lm(Age~Pclass+SibSp+FamSize, data=age.data.train)

age.reg.p <- round(predict(age.reg, newdata=age.data.test))

age.reg.p[age.reg.p<0] <- 0



library(car)

print("Variance_Inflation_Factor(VIF)")

print(vif(age.reg))
set.seed(0)

smplSize <-floor(nrow(age.data) * .75)

index <- sample(seq_len(nrow(age.data)), size = smplSize)

age.data.train <- age.data[index, ]

age.data.test <- age.data[-index, ]

#linear regression for age

age.reg <- lm(Age~Pclass+SibSp+FamSize, data=age.data.train)

age.reg.p <- round(predict(age.reg, newdata=age.data.test))

age.reg.p[age.reg.p<0] <- 0

print(paste("sum_of_errors:",sum(abs(age.reg.p - age.data.test$Age))))

print(paste("average_errors:",sum(abs(age.reg.p - age.data.test$Age))/178))

summary(age.reg)
library(nnet)

errorSum <- 10000

for (i in 1:100){

  age.ann <- nnet(Age~Pclass+SibSp+FamSize, linout = TRUE, data= age.data.train, size = 4, rang= .1,decay=.0005,maxit=1000)

  age.ann.p<- predict(age.ann, newdata=age.data.test)

  tmp <- sum(abs((age.ann.p - age.data.test$Age)))

  if (errorSum > tmp){

    errorSum <- tmp

    final.age.ann <- age.ann

    final.age.ann.p <- age.ann.p

  }

}
plot.nnet <- function(mod.in,nid=T,all.out=T,all.in=T,bias=T,wts.only=F,rel.rsc=5,circle.cex=5,

                    node.labs=T,var.labs=T,x.lab=NULL,y.lab=NULL,line.stag=NULL,struct=NULL,cex.val=1,

                    alpha.val=1,circle.col='lightblue',pos.col='black',neg.col='grey', max.sp = F, ...){

  

  require(scales) 

  

  #sanity checks

  if('mlp' %in% class(mod.in)) warning('Bias layer not applicable for rsnns object')

  if('numeric' %in% class(mod.in)){

    if(is.null(struct)) stop('Three-element vector required for struct')

    if(length(mod.in) != ((struct[1]*struct[2]+struct[2]*struct[3])+(struct[3]+struct[2])))

      stop('Incorrect length of weight matrix for given network structure')

  }

  if('train' %in% class(mod.in)){

    if('nnet' %in% class(mod.in$finalModel)){

      mod.in<-mod.in$finalModel

      warning('Using best nnet model from train output')

    }

    else stop('Only nnet method can be used with train object')

  }

  

  #gets weights for neural network, output is list

  #if rescaled argument is true, weights are returned but rescaled based on abs value

  nnet.vals<-function(mod.in,nid,rel.rsc,struct.out=struct){

    

    require(scales)

    require(reshape)

    

    if('numeric' %in% class(mod.in)){

      struct.out<-struct

      wts<-mod.in

    }

    

    #neuralnet package

    if('nn' %in% class(mod.in)){

      struct.out<-unlist(lapply(mod.in$weights[[1]],ncol))

    	struct.out<-struct.out[-length(struct.out)]

    	struct.out<-c(

    		length(mod.in$model.list$variables),

    		struct.out,

    		length(mod.in$model.list$response)

    		)    		

      wts<-unlist(mod.in$weights[[1]])   

    }

    

    #nnet package

    if('nnet' %in% class(mod.in)){

      struct.out<-mod.in$n

      wts<-mod.in$wts

    }

    

    #RSNNS package

    if('mlp' %in% class(mod.in)){

      struct.out<-c(mod.in$nInputs,mod.in$archParams$size,mod.in$nOutputs)

      hid.num<-length(struct.out)-2

      wts<-mod.in$snnsObject$getCompleteWeightMatrix()

      

      #get all input-hidden and hidden-hidden wts

      inps<-wts[grep('Input',row.names(wts)),grep('Hidden_2',colnames(wts)),drop=F]

      inps<-melt(rbind(rep(NA,ncol(inps)),inps))$value

      uni.hids<-paste0('Hidden_',1+seq(1,hid.num))

      for(i in 1:length(uni.hids)){

        if(is.na(uni.hids[i+1])) break

        tmp<-wts[grep(uni.hids[i],rownames(wts)),grep(uni.hids[i+1],colnames(wts)),drop=F]

        inps<-c(inps,melt(rbind(rep(NA,ncol(tmp)),tmp))$value)

        }

      

      #get connections from last hidden to output layers

      outs<-wts[grep(paste0('Hidden_',hid.num+1),row.names(wts)),grep('Output',colnames(wts)),drop=F]

      outs<-rbind(rep(NA,ncol(outs)),outs)

      

      #weight vector for all

      wts<-c(inps,melt(outs)$value)

      assign('bias',F,envir=environment(nnet.vals))

      }

    

    if(nid) wts<-rescale(abs(wts),c(1,rel.rsc))

    

    #convert wts to list with appropriate names 

    hid.struct<-struct.out[-c(length(struct.out))]

    row.nms<-NULL

    for(i in 1:length(hid.struct)){

      if(is.na(hid.struct[i+1])) break

      row.nms<-c(row.nms,rep(paste('hidden',i,seq(1:hid.struct[i+1])),each=1+hid.struct[i]))

    }

    row.nms<-c(

      row.nms,

      rep(paste('out',seq(1:struct.out[length(struct.out)])),each=1+struct.out[length(struct.out)-1])

      )

    out.ls<-data.frame(wts,row.nms)

    out.ls$row.nms<-factor(row.nms,levels=unique(row.nms),labels=unique(row.nms))

    out.ls<-split(out.ls$wts,f=out.ls$row.nms)

    

    assign('struct',struct.out,envir=environment(nnet.vals))

    

    out.ls

    

    }

  

  wts<-nnet.vals(mod.in,nid=F)

  

  if(wts.only) return(wts)

  

  #circle colors for input, if desired, must be two-vector list, first vector is for input layer

  if(is.list(circle.col)){

                    circle.col.inp<-circle.col[[1]]

                    circle.col<-circle.col[[2]]

                    }

  else circle.col.inp<-circle.col

  

  #initiate plotting

  x.range<-c(0,100)

  y.range<-c(0,100)

  #these are all proportions from 0-1

  if(is.null(line.stag)) line.stag<-0.011*circle.cex/2

  layer.x<-seq(0.17,0.9,length=length(struct))

  bias.x<-layer.x[-length(layer.x)]+diff(layer.x)/2

  bias.y<-0.95

  circle.cex<-circle.cex

  

  #get variable names from mod.in object

  #change to user input if supplied

  if('numeric' %in% class(mod.in)){

    x.names<-paste0(rep('X',struct[1]),seq(1:struct[1]))

    y.names<-paste0(rep('Y',struct[3]),seq(1:struct[3]))

  }

  if('mlp' %in% class(mod.in)){

    all.names<-mod.in$snnsObject$getUnitDefinitions()

    x.names<-all.names[grep('Input',all.names$unitName),'unitName']

    y.names<-all.names[grep('Output',all.names$unitName),'unitName']

  }

  if('nn' %in% class(mod.in)){

    x.names<-mod.in$model.list$variables

    y.names<-mod.in$model.list$respons

  }

  if('xNames' %in% names(mod.in)){

    x.names<-mod.in$xNames

    y.names<-attr(terms(mod.in),'factor')

    y.names<-row.names(y.names)[!row.names(y.names) %in% x.names]

  }

  if(!'xNames' %in% names(mod.in) & 'nnet' %in% class(mod.in)){

    if(is.null(mod.in$call$formula)){

      x.names<-colnames(eval(mod.in$call$x))

      y.names<-colnames(eval(mod.in$call$y))

    }

    else{

      forms<-eval(mod.in$call$formula)

      x.names<-mod.in$coefnames

      facts<-attr(terms(mod.in),'factors')

      y.check<-mod.in$fitted

      if(ncol(y.check)>1) y.names<-colnames(y.check)

      else y.names<-as.character(forms)[2]

    } 

  }

  #change variables names to user sub 

  if(!is.null(x.lab)){

    if(length(x.names) != length(x.lab)) stop('x.lab length not equal to number of input variables')

    else x.names<-x.lab

  }

  if(!is.null(y.lab)){

    if(length(y.names) != length(y.lab)) stop('y.lab length not equal to number of output variables')

    else y.names<-y.lab

  }

  

  #initiate plot

  plot(x.range,y.range,type='n',axes=F,ylab='',xlab='',...)

  

  #function for getting y locations for input, hidden, output layers

  #input is integer value from 'struct'

  get.ys<-function(lyr, max_space = max.sp){

  	if(max_space){ 

  		spacing <- diff(c(0*diff(y.range),0.9*diff(y.range)))/lyr

   	} else {

    	spacing<-diff(c(0*diff(y.range),0.9*diff(y.range)))/max(struct)

   	}

    

  		seq(0.5*(diff(y.range)+spacing*(lyr-1)),0.5*(diff(y.range)-spacing*(lyr-1)),

        length=lyr)

  }

  

  #function for plotting nodes

  #'layer' specifies which layer, integer from 'struct'

  #'x.loc' indicates x location for layer, integer from 'layer.x'

  #'layer.name' is string indicating text to put in node

  layer.points<-function(layer,x.loc,layer.name,cex=cex.val){

    x<-rep(x.loc*diff(x.range),layer)

    y<-get.ys(layer)

    points(x,y,pch=21,cex=circle.cex,col=in.col,bg=bord.col)

    if(node.labs) text(x,y,paste(layer.name,1:layer,sep=''),cex=cex.val)

    if(layer.name=='I' & var.labs) text(x-line.stag*diff(x.range),y,x.names,pos=2,cex=cex.val)      

    if(layer.name=='O' & var.labs) text(x+line.stag*diff(x.range),y,y.names,pos=4,cex=cex.val)

  }

  

  #function for plotting bias points

  #'bias.x' is vector of values for x locations

  #'bias.y' is vector for y location

  #'layer.name' is  string indicating text to put in node

  bias.points<-function(bias.x,bias.y,layer.name,cex,...){

    for(val in 1:length(bias.x)){

      points(

        diff(x.range)*bias.x[val],

        bias.y*diff(y.range),

        pch=21,col=in.col,bg=bord.col,cex=circle.cex

      )

      if(node.labs)

        text(

          diff(x.range)*bias.x[val],

          bias.y*diff(y.range),

          paste(layer.name,val,sep=''),

          cex=cex.val

        )

    }

  }

  

  #function creates lines colored by direction and width as proportion of magnitude

  #use 'all.in' argument if you want to plot connection lines for only a single input node

  layer.lines<-function(mod.in,h.layer,layer1=1,layer2=2,out.layer=F,nid,rel.rsc,all.in,pos.col,

                        neg.col,...){

    

    x0<-rep(layer.x[layer1]*diff(x.range)+line.stag*diff(x.range),struct[layer1])

    x1<-rep(layer.x[layer2]*diff(x.range)-line.stag*diff(x.range),struct[layer1])

    

    if(out.layer==T){

      

      y0<-get.ys(struct[layer1])

      y1<-rep(get.ys(struct[layer2])[h.layer],struct[layer1])

      src.str<-paste('out',h.layer)

      

      wts<-nnet.vals(mod.in,nid=F,rel.rsc)

      wts<-wts[grep(src.str,names(wts))][[1]][-1]

      wts.rs<-nnet.vals(mod.in,nid=T,rel.rsc)

      wts.rs<-wts.rs[grep(src.str,names(wts.rs))][[1]][-1]

      

      cols<-rep(pos.col,struct[layer1])

      cols[wts<0]<-neg.col

      

      if(nid) segments(x0,y0,x1,y1,col=cols,lwd=wts.rs)

      else segments(x0,y0,x1,y1)

      

    }

    

    else{

      

      if(is.logical(all.in)) all.in<-h.layer

      else all.in<-which(x.names==all.in)

      

      y0<-rep(get.ys(struct[layer1])[all.in],struct[2])

      y1<-get.ys(struct[layer2])

      src.str<-paste('hidden',layer1)

      

      wts<-nnet.vals(mod.in,nid=F,rel.rsc)

      wts<-unlist(lapply(wts[grep(src.str,names(wts))],function(x) x[all.in+1]))

      wts.rs<-nnet.vals(mod.in,nid=T,rel.rsc)

      wts.rs<-unlist(lapply(wts.rs[grep(src.str,names(wts.rs))],function(x) x[all.in+1]))

      

      cols<-rep(pos.col,struct[layer2])

      cols[wts<0]<-neg.col

      

      if(nid) segments(x0,y0,x1,y1,col=cols,lwd=wts.rs)

      else segments(x0,y0,x1,y1)

      

    }

    

  }

  

  bias.lines<-function(bias.x,mod.in,nid,rel.rsc,all.out,pos.col,neg.col,...){

    

    if(is.logical(all.out)) all.out<-1:struct[length(struct)]

    else all.out<-which(y.names==all.out)

    

    for(val in 1:length(bias.x)){

      

      wts<-nnet.vals(mod.in,nid=F,rel.rsc)

      wts.rs<-nnet.vals(mod.in,nid=T,rel.rsc)

      

    	if(val != length(bias.x)){

        wts<-wts[grep('out',names(wts),invert=T)]

        wts.rs<-wts.rs[grep('out',names(wts.rs),invert=T)]

    		sel.val<-grep(val,substr(names(wts.rs),8,8))

    		wts<-wts[sel.val]

    		wts.rs<-wts.rs[sel.val]

    		}

    

    	else{

        wts<-wts[grep('out',names(wts))]

        wts.rs<-wts.rs[grep('out',names(wts.rs))]

      	}

      

      cols<-rep(pos.col,length(wts))

      cols[unlist(lapply(wts,function(x) x[1]))<0]<-neg.col

      wts.rs<-unlist(lapply(wts.rs,function(x) x[1]))

      

      if(nid==F){

        wts.rs<-rep(1,struct[val+1])

        cols<-rep('black',struct[val+1])

      }

      

      if(val != length(bias.x)){

        segments(

          rep(diff(x.range)*bias.x[val]+diff(x.range)*line.stag,struct[val+1]),

          rep(bias.y*diff(y.range),struct[val+1]),

          rep(diff(x.range)*layer.x[val+1]-diff(x.range)*line.stag,struct[val+1]),

          get.ys(struct[val+1]),

          lwd=wts.rs,

          col=cols

        )

      }

      

      else{

        segments(

          rep(diff(x.range)*bias.x[val]+diff(x.range)*line.stag,struct[val+1]),

          rep(bias.y*diff(y.range),struct[val+1]),

          rep(diff(x.range)*layer.x[val+1]-diff(x.range)*line.stag,struct[val+1]),

          get.ys(struct[val+1])[all.out],

          lwd=wts.rs[all.out],

          col=cols[all.out]

        )

      }

      

    }

  }

  

  #use functions to plot connections between layers

  #bias lines

  if(bias) bias.lines(bias.x,mod.in,nid=nid,rel.rsc=rel.rsc,all.out=all.out,pos.col=alpha(pos.col,alpha.val),

                      neg.col=alpha(neg.col,alpha.val))

  

  #layer lines, makes use of arguments to plot all or for individual layers

  #starts with input-hidden

  #uses 'all.in' argument to plot connection lines for all input nodes or a single node

  if(is.logical(all.in)){  

    mapply(

      function(x) layer.lines(mod.in,x,layer1=1,layer2=2,nid=nid,rel.rsc=rel.rsc,

        all.in=all.in,pos.col=alpha(pos.col,alpha.val),neg.col=alpha(neg.col,alpha.val)),

      1:struct[1]

    )

  }

  else{

    node.in<-which(x.names==all.in)

    layer.lines(mod.in,node.in,layer1=1,layer2=2,nid=nid,rel.rsc=rel.rsc,all.in=all.in,

                pos.col=alpha(pos.col,alpha.val),neg.col=alpha(neg.col,alpha.val))

  }

  #connections between hidden layers

  lays<-split(c(1,rep(2:(length(struct)-1),each=2),length(struct)),

              f=rep(1:(length(struct)-1),each=2))

  lays<-lays[-c(1,(length(struct)-1))]

  for(lay in lays){

    for(node in 1:struct[lay[1]]){

      layer.lines(mod.in,node,layer1=lay[1],layer2=lay[2],nid=nid,rel.rsc=rel.rsc,all.in=T,

                  pos.col=alpha(pos.col,alpha.val),neg.col=alpha(neg.col,alpha.val))

    }

  }

  #lines for hidden-output

  #uses 'all.out' argument to plot connection lines for all output nodes or a single node

  if(is.logical(all.out))

    mapply(

      function(x) layer.lines(mod.in,x,layer1=length(struct)-1,layer2=length(struct),out.layer=T,nid=nid,rel.rsc=rel.rsc,

                              all.in=all.in,pos.col=alpha(pos.col,alpha.val),neg.col=alpha(neg.col,alpha.val)),

      1:struct[length(struct)]

      )

  else{

    node.in<-which(y.names==all.out)

    layer.lines(mod.in,node.in,layer1=length(struct)-1,layer2=length(struct),out.layer=T,nid=nid,rel.rsc=rel.rsc,

                pos.col=pos.col,neg.col=neg.col,all.out=all.out)

  }

  

  #use functions to plot nodes

  for(i in 1:length(struct)){

    in.col<-bord.col<-circle.col

    layer.name<-'H'

    if(i==1) { layer.name<-'I'; in.col<-bord.col<-circle.col.inp}

    if(i==length(struct)) layer.name<-'O'

    layer.points(struct[i],layer.x[i],layer.name)

    }



  if(bias) bias.points(bias.x,bias.y,'B')

  

}
library(devtools)

library(reshape2)

plot.nnet(final.age.ann)
print(paste("sum_of_errors:",sum(abs(final.age.ann.p - age.data.test$Age))))

print(paste("average_errors:",sum(abs(final.age.ann.p - age.data.test$Age))/178))
NAindex <- which(is.na(data$Age))

age.NA.data <- data[NAindex,]

age.NA.solved <- predict(final.age.ann, newdata=age.NA.data)

for (i in 1:177){

  data$Age[NAindex[i]] <- age.NA.solved[i]

}
summary(data)