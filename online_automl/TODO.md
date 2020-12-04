# Dataset
1. Find a dataset 
2. Test basic VW


# Can Ray tune be used for VW

- Problems
1. VW cannot be pickle.dumped
(One possible way to work around is to use save model in vw. But I guess we are not doing it)

Ask Eudorious whether VW can be dumped?

## About Ray tune


-  What it needs to dump?
trainable, results, whatever you want at check points

-  When it dumps objects?
    1. _Registry in registry.py (dump trainable)
    2. save_checkpoint in trainable function (customzed)
    3. the result returned (the results returned )

Conclusion: ray tune cannot be used
# If Ray tune cannot be used, need to write from scrtch 

- Write a BlendSearch Run
    - How should the data be allocated
- How to write a trial runner.
    - Trial 
    - Trial runner