grid_size <- 4

x <- c(1,1,2,3,4,4,4)
y <- c(1,2,2,2,2,3,4)

plot(1:grid_size,
     1:grid_size,
     type="n",
     xlim=c(1,4),
     ylim=c(4,1),
     xlab="Column",
     ylab="Row",
     main="Agent Path in Grid World")

abline(v=1:4,col="gray")
abline(h=1:4,col="gray")

text(1,1,"S",cex=2,col="blue")
text(4,4,"G",cex=2,col="red")

lines(y,x,lwd=3)
points(y,x,pch=19,cex=1.5)