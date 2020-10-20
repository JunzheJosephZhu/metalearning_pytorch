- [ ] try make neuronal activity happen on a higher time scale
- [1] do 3 logistic regression. 
    Input: PR cost, FR cost, PR reward, FR reward; Output: action; Receptive Field: [-5, -1]
    Input: PR value(ratio between reward/cost), FR value, Output: action; Receptive Field: [-5, -1]
    Input: Neuronal activation; Output: action & reward & cost & value; Receptive Field: [0, 0] (same trial)
- [2] try to predict the neuronal activation and policy given only behavior history, without prior knowledge of lstm weights
- [3] visualize stuff: show one example block for each type for how network behaves, plot PR choice on top, FR choices on bottom, also plot sigmoidal curve
    For regression 1, 2: For block of each type, plot the average weights, with x-axis being trial number in history, y-axis being weight
    For regression 3, x-axis: cell(rank from weight value), y-axis: weight, make a separate plot for action & reward & cost & value
- [ ] definition of computational logic: how cost & reward affect activation of neurons
    definition of functional connectivity: for example group A decreases, group B decreases, more like correlation between groups.
- [0] turn pickle into txt