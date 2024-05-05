
# 绘制累计收益增长图 ---------------------------------------------------------------

#读取CSV文件
data <- read.csv("score.csv", header = TRUE)
return <- data+1

# 对数据框做修改
dim(data); colnames(return)
x <- 1:485


num_rows = nrow(data_frame)

# defining new row data frame
new_row = c("New","Row","Added","Dataframe")

# assigning the new row at a new
# index after the original number of rows 
data_frame[num_rows + 1,] = new_row
print ("Modified Data Frame")
print(data_frame)


# 绘制折线图
op <- par(no.readonly = T)
library(dplyr)
library(tidyverse)
par(mar = c(rep(4,4)))
return %>% {
  plot(res_ew ~ x,data = .,type = "l",col = "red",lwd = 2)
}
par(op)
