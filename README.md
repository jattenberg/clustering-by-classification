# clustering-by-classification
can we cluster using an arbitrary classifier? I think so. \

# Update 7-9-2020
finally got the chance to test some ideas on this. So far it seems that initial conditions are clearly the dominating force in determining the quality of the clustering outcome, at least according to my visual inspection. I've tried a number of approaches. of note:
- kmeans leads to good clustering; the model basically learns to project the way it makes decision boundaries onto kmeans
- model driven approach inspired by kmeans++ initialization: pick a point, find it's neighbors, build a model on this, pick what's unlikely as a next point and iterate. this seems to work as well as kmeans and has the character of the model baked in. 
- random+neighbors for starting clusters. Seems to work ok, not really in the spirit of the approach though. 

# Next step:
- try the model-driven initialization with retries based on evaluating the accuracy. every time a random choice is made, it can be compared with several alternatives based on the accuracy of the outcome. thinking is, more accuracy, the better? 
