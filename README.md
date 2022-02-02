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
  
9. Number of neighbors and the error curve

  - This plot would show that as we have more number of neighbors the error gets smaller hence the method is biased. 

10. Mention about our hypothesis of if two genes are connected then we expect their transcription values to be correlated? or we expect their transcription values to be equal? IF BY ANY EXTERNAL RESOURCES (we selected this to be experimental evidence in order to have a strong evidence) two genes are known to be connected how would their transcription values would be correlated? Would they be equal ? Would they be correlated? Emphasize on the biological factors and find some supporting ideas and mention them in the paper. One classic way to add is via a regularization term. However the classical regularizers penalizes the inequality whereas we need the penalization of the discorrelation. 

11. Further analysis on how did clustering two genes effected their prediction value analysis can be done.

12. Force the decomposition to be non negative. Why did we do that? Did we do it in the current implementation? How to do it experimentally?

13. Combine the external information with clustering.
    
    **How to add the external information such that we benefit from it the most?**
        
      - Check the genes in each cluster.
      - Maybe make sure that the genes that are connected are in the same cluster. 
      - Another option is to put all the  genes that are correlated in the same cluster and then add the external information. If there are 70 genes that are correlated and the 908 genes are yet to be clustered.
 
14. Do not forget to add the computational complexity analysis.
15. Calculate accuracy, loss value for each gene.
  
** Data **
- Important detail that I should discuss in the paper is the fact that test samples and training samples are pairs of cell line and drugs instead of i,j,k values all over the tensor. It is an important detail. This way it is possible to treat that for any selected test samples pair of cell line and drugs we make sure that there is no gene that share the same cell line and drug combination and that is in the training. So this is done to avoid cheating. We made sure of this.
