# online active learning
## Alg2: use pseudo label
### Setup:
1. gen_x: generate feature vector in R^d
2. true_h: true coefficient in R^d, y = <true_h, x>
3. h: vectors in R^d, discretized, <h,h> \le a constant. N experts
4. r: gamma - |h|, requester function. M gammas

### Data structure:
1. h x gamma, matrix?
2. O: matrix, counts of observations
3. S: slack term? 

### Subroutines:
1. gen_pseudo_label(H, W, x), H: predictions from every expert on x; W: weights from every expert. 
2. find_UCB(H, x, O): find UCB

