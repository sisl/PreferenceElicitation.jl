### What is preference elicitation for engineering design optimization?
In engineering design optimization, one must inevitably make trade-offs between multiple conflicting design objectives, such as cost, reliability, and performance. Often, this trade-off must be made explicitly by specifying a tradeoff ratio between the objectives. But setting these trade-off ratios is *hard*.  You might have a ballpark intuition for what they should be, but often the optimization is very sensitive to the choice of trade-offs. 

Preference elicitation relieves some of this burden. Instead of deciding the trade-off ratios directly, you make a series of pairwise comparisons between designs: do you prefer design A, design B, or are you indifferent? Research has shown that these pairwise comparisons are far easier to make and *justify* than explicitly setting the trade-off ratio yourself.  This preference elicitation algorithm then takes your decisions and uses them to calculate an optimal set of trade-off ratios based on your choices. It can even suggest which comparisons you should make to get the most accurate trade-off ratios. 

### Installation
To install, simply run 
```julia
Pkg.clone("https://github.com/sisl/PreferenceElicitation.jl.git")
```
Note: if you are behind a proxy server, you’ll need to configure git differently. The following commands should do it:
```julia
run(`git config --global http.proxy $http_proxy`) # where $http_proxy is your proxy server
run(`git config --global url."https://".insteadOf git://`) # forces git to use https
```

### Example 
Once installed, begin by creating your preference elicitation object:
```julia
using preferenceElicitation # load package
p = prefEl([1 0;   # first design
            0 1;   # second design
            0 0], # third design
            priors = [Normal(0,1), Normal(0,1)]) # Specify priors for each variable
                                                  # to be Guassian with mean 0 and variance 1
```
Preferences are put in with the ```@addPref``` macro:
```julia
@addPref p 1 > 3  # prefer design 1 to design 3
@addPref p 3 < 2  # prefer design 2 to design 3
@addPref p 1 == 2 # indifferent between designs 1 and 2
```
To learn your ideal weights, use the ```infer()``` function after putting in your preferences:
```julia
infer(p)
```
By default, this calculates the MAP estimate of the posterior. To get the posterior mean estimate instead, you can use  ```infer(p, method = "MCMC")```. 

Finally, preference elicitation can tell you which comparision will tell it the most by using the ```suggest()``` method:
```julia
suggest(p)
```
