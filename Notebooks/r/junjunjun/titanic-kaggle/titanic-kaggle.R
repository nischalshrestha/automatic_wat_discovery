# 色々パッケージがあるみたいなので読み込みます。

library('ggplot2')

library('ggthemes')

library('scales')

library('dplyr')

library('mice')

library('randomForest')

library('xgboost')

library('caret')

library('glmnet')

library('kernlab')

library('C50')

library('pROC')

library('Matrix')

library('tidyverse')

library('stringr')

library('ranger')

library('e1071')

library('corrplot')

# データをダウンロードします。

train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)
# 「bind_rows」関数を使って,「full」データを作ります。

full  <- bind_rows(train, test) 
# 「full」データをチェックしてみます。

str(full)

summary(full)
# 列「Name」に「Mlle」,「Ms」,「Mme」などの昔の男女に対する称号,

#「sir」等の位が高いと思われる称号が含まれているので「gsub」関数を使って新しく,列「Title」を作ります。



full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
# 新しく家族の姓（Surname)というfeature（変数,列）を作成してみます。

full$Surname <- sapply(full$Name,

                      function(x) strsplit(x, split = '[,.]')[[1]][1])
# Titleに対しての生存者について「table」関数を使って確認してみます。０が死亡で１が生存になります。

table(full$Title,full$Survived)
# Titleに対しての生存割合について,「prop.table」関数を使って確認してみます。

prop.table(table(full$Title,full$Survived),1)
# Titleに対しての生存割合について,「ggplot」を使って可視化してみます。

ggplot(full[1:891,],aes(x = Title,fill= factor(Survived)))+geom_bar(stat='count',position='dodge')+theme_few()
# Titleの種類が多いので,少人数のTitleをまとめ,列「Title」に再代入します。

officer <- c('Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev')

royalty <- c('Dona', 'Lady', 'the Countess','Sir', 'Jonkheer')

full$Title[full$Title == 'Mlle']        <- 'Miss'

full$Title[full$Title == 'Ms']          <- 'Miss'

full$Title[full$Title == 'Mme']         <- 'Mrs'

full$Title[full$Title %in% royalty]  <- 'Royalty'

full$Title[full$Title %in% officer]  <- 'Officer'

full$Title = as.factor(full$Title)
# Titleが少なくなったので,可視化するとスッキリします。

ggplot(full[1:891,],aes(x = Title,fill= factor(Survived)))+geom_bar(stat='count',position='dodge')+theme_few()
# SibSp(兄弟または配偶者)の数と生存者数を確認してみます。

table(full$SibSp,full$Survived)
# SibSp(兄弟または配偶者)の数と生存確率を確認してみます。

prop.table(table(full$SibSp,full$Survived),1)
# SibSp(兄弟または配偶者)の数と生存者を可視化してみます。

ggplot(full[1:891,],aes(x = SibSp,fill= factor(Survived)))+geom_bar(stat='count',position='dodge')+theme_few()
# Parch（親または子供）と生存者数の関係を確認してみます。

table(full$Parch,full$Survived)
# Parch（親または子供）と生存確率を確認してみます。

prop.table(table(full$Parch,full$Survived),1)
# Parch（親または子供）と生存について可視化してみます。

ggplot(full[1:891,],aes(x = Parch,fill= factor(Survived)))+geom_bar(stat='count',position='dodge')+theme_few()
# 家族の数について新しいfeatuer(変数,列）を作ってみます。

# SibSp(兄弟または配偶者)＋Parch（親または子供）＋１（自分）

full$Fsize <- full$SibSp + full$Parch + 1
# 家族の数と生存者について確認してみます。

table(full$Fsize,full$Survived)
# 家族の数と生存確率について確認してみます。

prop.table(table(full$Fsize,full$Survived),1)
# 家族の数と生存確率について可視化してみます。

ggplot(full[1:891,],aes(x = Fsize,fill= factor(Survived)))+geom_bar(stat='count',position='dodge')+theme_few()
# 家族のサイズが1のとき（つまり単独のとき）は圧倒的に生存確率が低いため,Aloneという列を追加します。

full$Alone[full$Fsize == 1] = 1

full$Alone[full$Fsize>=2] = 2
# 家族の姓（Surname）と家族の数（Fsize)を足し合わせて,家族ごとのIDを作成してみます。

# 同じ家族での生存可否が高いと思われるためです。

full$FamilyID <- paste(full$Surname, full$Fsize, sep='_')
# 乗船していたAge(年齢）について可視化してみます。

hist(full$Age)
# 乗船していたAge(年齢）と生存可否について可視化してみます。

ggplot(full[1:891,],aes(x = Age,fill= factor(Survived)))+geom_bar(stat='count',position='dodge')+theme_few()
# 乗船していたAge(年齢）と生存可否について線形回帰と散布図を使って可視化してみます。

ggplot(full,aes(x=full$Age,y=full$Survived))+geom_point()+geom_smooth(method = lm, se=TRUE)
# Pclassと生存確率をみてみます。

prop.table(table(full$Pclass,full$Survived),1)
# Pclassと生存者を可視化してみます。

ggplot(full[1:891,],aes(x = Pclass,fill= factor(Survived)))+geom_bar(stat='count',position='dodge')+theme_few()
# Embarked(乗船した場所）には空欄があります。どの乗客が空欄（””）なのかwhich関数を使って調べてみます。

# ID62とID830の方がEmbarked(乗船した場所）について空欄なのがわかります。

which(full$Embarked =="")
# ID62の方とID830の方の情報を見てみましょう。

full[c(62,830),]
# Embarked(乗船した場所）について,Fare(運賃）と,Pclass(船室）から推測してみます。

embark_fare = full %>% filter(PassengerId !=62 & PassengerId !=830)

ggplot(embark_fare,aes(x = Embarked,y = Fare,fill = factor(Pclass)))+geom_boxplot()+geom_hline(aes(yintercept=80),colour='red',linetype='dashed',lwd=2)+theme_few()
# ID62の方とID830の方は,Pclass「1」で運賃が「80」なので、Embarked(乗船した場所）について「S」を

# 代入してみます。。

full$Embarked[c(62,830)] = 'S'
# Fareには「na」（非数値）が含まれています。subset（関数）でnaの方を調べてみます。

subset(full,is.na(Fare))
# PclassごとのFareの平均値と中央値を調べてみます。group_by関数によってPclassごとにグルーピングをし,

# summarise関数によって,mean(平均値）とmedian(中央値）を調べます。

#（na.rm=TRUWは,naは含めないで計算するという意味です）

full%>%group_by(Pclass)%>%dplyr::summarise(mean(Fare,na.rm=TRUE))

full%>%group_by(Pclass)%>%dplyr::summarise(median(Fare,na.rm=TRUE))
# ID1044の方はPclass「３」なので、中央値8.05をFareに代入します。

full$Fare[c(1044)] = 8.05
# 263個のNa（欠損値）について補完していきます。まず、後ほど比較できるよう現在のAgeを別の列に代入しておきます。

full$Agebefore = full$Age
# Ageの欠損値に対する予測です。

predicted_age <- train(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + Fsize,

                       tuneGrid = data.frame(mtry = c(2, 3, 7)),

                       data = full[!is.na(full$Age), ],

                       method = "ranger",

                       trControl = trainControl(method = "cv", number = 10,repeats = 10, verboseIter = TRUE),importance = 'impurity')
# 予測したAgeを欠損値に補完します。

full$Age[is.na(full$Age)] <- predict(predicted_age, full[is.na(full$Age),])
# summaryでみるとNa（欠損値が無くなりました）。

summary(full)
vimp <- varImp(predicted_age)

ggplot(vimp, 

        top = dim(vimp$importance)[1])
# 補完前と補完後を比較してみます。

par(mfrow=c(1,2)) 

 hist(full$Agebefore, freq=F, main='Age: Original Data',

      col='darkgreen', ylim=c(0,0.04))

 hist(full$Age, freq=F, main='Age: After Data', 

      col='lightgreen', ylim=c(0,0.04))
# Ageが補完できたので,もう一度,Ageと生存率の関係を見てみます。また,今回はPclassごとの生存の回帰直線も加えました。

ggplot(full,aes(x=full$Age,y=full$Survived,group=full$Pclass))+geom_point()+geom_smooth(method = lm, se=TRUE,aes(colour=full$Pclass))+theme_bw()
# チケット番号について,文字部分（アルファベット部分）と数値部分に分割します。

full$Ticket_Pre <- sapply(full$Ticket, FUN=function(x) {ifelse(str_detect(x, " "),str_split(x, " ")[[1]][1],'None')})
# 乗船していたAge(年齢）と生存可否について可視化してみます。16歳程度以下の生存率が比較的高くなっています。16歳ごとに新しいfeatura(変数・列）Age2を作成したいと思います。

ggplot(full[1:891,],aes(x = Age,fill= factor(Survived)))+geom_bar(stat='count',position='dodge')+theme_few()

full$Age2[full$Age<=16] = 1

full$Age2[full$Age>16& full$Age<=32] = 2

full$Age2[full$Age>32& full$Age<=48] = 3

full$Age2[full$Age>48& full$Age<=64] = 4

full$Age2[full$Age>64] = 5
full$Ticket_Pre <- as.factor(str_to_upper(str_replace_all(full$Ticket_Pre, "[./]", "")))
# チケット番号の数値部分の処理です。

full$Ticket_Num <- sapply(full$Ticket, FUN=function(x) {ifelse(str_detect(x, " "),str_split(x, " ")[[1]][2], as.character(x))})
# 可視化してみますと,チケット番号が低いほうが,Pclass(客室）が高く（1号客室）

# また生存率が高いことがわかります。

# 一方で,チケット番号が高くなっていくと,Pclass(客室）のクラスが低くなり（3号客室）、

# 生存率も低くなっていることがわかります。

ggplot(full, aes(x =Ticket_Num, y =Fare)) + geom_point(aes(colour = Pclass,shape = factor(Survived)))
# 上図をみると比較的Fare(運賃）が低いところで生存率が低く,ある一定程度のFareを超えると生存率が高くなっていくので,新しく（Fare2)を作ります。

summary(train$Fare)

full$Fare2[full$Fare<=7.91] = 1

full$Fare2[full$Fare>7.91& full$Fare<=14.454] = 2

full$Fare2[full$Fare>14.454& full$Fare<=31] = 3

full$Fare2[full$Fare>31] = 4
# キャビンについての処理です。キャビンについては,データが少ないので,データが空欄（"")については「N」を当てました。

full$Cabin2 = substr(full$Cabin,1,1)

full$Cabin2[full$Cabin2 == ""] = "N"

full$Cabin2 = as.factor(full$Cabin2)

levels(full$Cabin2)
# 可視化してみると,キャビンについての情報がある方は比較的生存率が高く,

# キャビンについて情報がない方、つまり「N」は生存率が低いことがわかります。

ggplot(full[1:891,],aes(x = Cabin2,fill= factor(Survived)))+geom_bar(stat='count',position='dodge')+theme_few()
# １度必要な各データをfactor化しnumberに返します。



full$Pclass= as.numeric(full$Pclass)-1

full$Sex = as.numeric(as.factor(full$Sex))-1

full$SibSp = as.numeric(as.factor(full$SibSp))-1

full$Parch = as.numeric(as.factor(full$Parch))

full$Embarked = as.numeric(as.factor(full$Embarked))-1

full$Title = as.numeric(as.factor(full$Title))-1

full$Fsize = as.numeric(as.factor(full$Fsize))-1

full$Age2 = as.numeric(as.factor(full$Age2))-1

full$Fare2 = as.numeric(as.factor(full$Fare2))-1

full$FamilyID = as.numeric(as.factor(full$FamilyID))-1

full$Alone = as.numeric(as.factor(full$Alone))-1

full$Cabin2 = as.numeric(as.factor(full$Cabin2))-1

full$Ticket_Pre = as.numeric(as.factor(full$Ticket_Pre))-1

full$Ticket_Num = as.numeric(as.factor(full$Ticket_Num))-1



full$Ticket = as.numeric(as.factor(full$Ticket))-1

full$Cabin = as.numeric(as.factor(full$Cabin))-1

full$Surname = as.numeric(as.factor(full$Surname))-1

full$Name = as.numeric(as.factor(full$Name))-1

full$PassengerId = as.numeric(as.factor(full$PassengerId))

full$Survived = as.numeric(as.factor(full$Survived))-1
# fullデータをtrain2とtest2に区分します。

train2 = full[1:891,]

test2 = full[892:1309,]
# 各データの相関関係です。

cor.train2 = train2 %>% cor

corrplot(train2 %>% cor,addCoefcol =TRUE)

cor.train2.l = cor.train2 %>% as.data.frame %>% mutate(item1 = rownames(.)) %>% gather(item2,corr,-item1)
# 各データの相関関係2です。

ggplot(data=cor.train2.l,aes(x = item1,y=item2,fill=corr))+geom_tile()+scale_fill_gradient(low="white",high="red")
# まずは,svmを使ってモデルを作成します。

svm.model <- ksvm(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + Fsize + Age2 + Fare2 + Alone + FamilyID +Cabin2 +Ticket_Pre +Ticket_Num,data = train2,prob.model=TRUE)
# svm.modelをtest2データに当てはめて予測をします。

prediction <- predict(svm.model, test2)
# データの書き出し作業です。

solution <- data.frame(PassengerID = test2$PassengerId, Survived = prediction)

write.csv(solution, file = 'svmmodel.csv', row.names = F)
# 次にランダムフォレストを使って予測してみます。

rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + Fsize + Age2 + Fare2 + Alone + FamilyID+Cabin2+Ticket_Pre +Ticket_Num,data = train2)
#ランダムフォレストのモデルになります。error rate: 15.82%　,Confusion matrixは,「0（死亡）」と予測して「0」だった数が494,「0（死亡）」と予測して「1」だった数が55(間違いが55,つまり10%程度）,「1（生存）」と予測して「0」だった数が86(間違い）,「1（生存）」と予測して「1」だった数が256（正解）となります。

rf_model
# 次にランダムフォレストの各変数の重要度をみてみます。

vi = varImpPlot(rf_model)
# rf.modelをtest2データに当てはめて予測をします。

prediction.rf_model <- predict(rf_model, test2)
# データの書き出し作業です。

solution2 <- data.frame(PassengerID = test2$PassengerId, Survived = prediction.rf_model)

write.csv(solution2, file = 'rfmodel.csv', row.names = F)
# 次にxgboostを使って予測してみます。xgboostを使うには,まず必要な変数を取り出し,as.matrixにて行列化する必要があります。

str(full)

full2 = full[,-c(1,4,9,11,14)]

full2$Survived = as.numeric(full2$Survived)
train2 = full2[1:891,]

test2 = full2[892:1309,]
y=train2[,1]

y = as.integer(y)

x = as.matrix(train2[,2:18])

set.seed(123)

param = list("objective"="multi:softmax","num_class" = 2,"eval_metric" = "mlogloss")

k=round(1+log2(nrow(x)))

cv.nround = 100
bst.cv = xgb.cv(param=param, data = x, label = y, nfold = k,nrounds=cv.nround)
nround =8
model = xgboost(param=param, data = x, label = y, nrounds=nround)

test_x = as.matrix(test2[,2:18])

pred = predict(model,test_x)
# データの書き出し作業です。

solution3 <- data.frame(PassengerID = test$PassengerId, Survived = pred)

write.csv(solution2, file = 'xgboostmodel.csv', row.names = F)