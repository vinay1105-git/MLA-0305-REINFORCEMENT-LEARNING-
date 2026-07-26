library(DiagrammeR)

grViz("

digraph MDP {

graph [
layout = dot
rankdir = LR
splines = spline
nodesep = 0.6
ranksep = 1.0
]

node [
shape = box
style = 'rounded,filled'
fontname = Arial
fontsize = 14
color = black
penwidth = 2
]

Title [
label='MARKOV DECISION PROCESS'
fillcolor=navy
fontcolor=white
fontsize=22
]

Input [
label='Input Environment'
fillcolor=lightblue
]

States [
label='States (S)'
fillcolor=lightgreen
]

Actions [
label='Actions (A)'
fillcolor=lightgreen
]

Transition [
label='Transition Model\nP(next_state | state, action)'
fillcolor=lightgreen
]

Reward [
label='Reward Function\nR(state, action)'
fillcolor=gold
]

Policy [
label='Policy'
fillcolor=khaki
]

Agent [
label='Decision Agent'
shape=ellipse
fillcolor=orange
]

Value [
label='Value Function'
fillcolor=plum
]

Iteration [
label='Policy / Value Iteration'
fillcolor=mistyrose
]

Optimal [
label='Optimal Policy'
fillcolor=tomato
fontcolor=white
]

Goal [
label='Maximum Cumulative Reward'
shape=ellipse
fillcolor=red
fontcolor=white
fontsize=18
]

Title -> Input
Input -> States
Input -> Actions

States -> Transition
Actions -> Transition

Transition -> Reward
Transition -> Agent

Reward -> Agent

Agent -> Policy
Agent -> Value

Policy -> Iteration
Value -> Iteration

Iteration -> Optimal

Optimal -> Goal

}
")
