#---------------------------------------
# Markov Decision Process Demonstration
#---------------------------------------

grid_size <- 4

start <- c(1,1)
goal  <- c(4,4)

state <- start

path <- matrix("",grid_size,grid_size)

path[start[1],start[2]] <- "S"
path[goal[1],goal[2]]   <- "G"

print("Initial Grid")
print(path)

step <- 1

cat("\nStarting Simulation\n\n")

while(!(all(state==goal))){
  
  row <- state[1]
  col <- state[2]
  
  possible <- list()
  
  if(row>1)
    possible$Up <- c(row-1,col)
  
  if(row<grid_size)
    possible$Down <- c(row+1,col)
  
  if(col>1)
    possible$Left <- c(row,col-1)
  
  if(col<grid_size)
    possible$Right <- c(row,col+1)
  
  best_action <- NULL
  best_distance <- 100
  
  for(a in names(possible)){
    
    next_state <- possible[[a]]
    
    distance <- abs(goal[1]-next_state[1]) +
      abs(goal[2]-next_state[2])
    
    if(distance < best_distance){
      
      best_distance <- distance
      best_action <- a
      state_next <- next_state
      
    }
    
  }
  
  reward <- -1
  
  if(all(state_next==goal))
    reward <- 100
  
  cat("Step:",step,"\n")
  
  cat("Current State :",state,"\n")
  cat("Action Taken  :",best_action,"\n")
  cat("Next State    :",state_next,"\n")
  cat("Reward        :",reward,"\n\n")
  
  state <- state_next
  
  step <- step+1
  
}