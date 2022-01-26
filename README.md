# CellsDrugsGenes

I will be keeping log of the set of experiments we have run for this project.

1. Tensor Completion with Matrix Slices(To be added)

2. Tensor Completion with External information

  - External Information as Correlation
  
  - External Information as Equality

2.5 How to access to String Database?
  
   - Which correlation did we use exactly?
   
   - What is our threshold?

3. Tensor Completion with Clustering

4. Tensor Completion with External Information and Clustering

5. Code snippet that shows the connectivity of the genes which are selected based on experimental evidence. (To be added)

6. The experiments that showed the best that we can reach if we used a weighted average.. (To be added)
  - Recall what was the weighted average? 
  - Do we need to mention this in the paper?

7. Mention the parameter tuning performed for selecting the best optimization algorithm for Tensor Completion Algorithms. As well as for selecting the best parameter for the selected algorithm. 
  - Maybe we should have used SPPA here and cite our own work?
  - Work on what needs to be adjusted in SPPA to run the experiments

8. Have a part in the paper that compares the clock time for the tensor completion with fiber update and without fiber update. Emphasize that we could do the fiber update due to the property of the dataset.
  - Which time should we report? Should we report the time that it takes to perform one update? Or should we report the time that it takes to perform 120 updates while calculating the test loss at every iteration? In that case it would not be a pure comparison of the fiber update. Hence I just decided to time the pure update with and without the fiber. 
  - The code for non-fiber update is not in the directory. (To be added)
  
