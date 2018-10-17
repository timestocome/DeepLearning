
http://github.com/timestocome
Linda MacPhee-Cobb


Dataset source and information
https://archive.ics.uci.edu/ml/datasets/arrhythmia

<b> Dataset notes: </b>
 The aim is to determine the type of arrhythmia from 
      the ECG recordings.
 This database contains 279 attributes, 206 of which are linear
     valued and the rest are nominal. 

     Concerning the study of H. Altay Guvenir: "The aim is to distinguish
     between the presence and absence of cardiac arrhythmia and to
     classify it in one of the 16 groups. Class 01 refers to 'normal'
     ECG classes 02 to 15 refers to different classes of arrhythmia
     and class 16 refers to the rest of unclassified ones. For the
     time being, there exists a computer program that makes such a
     classification. However there are differences between the
     cardiolog's and the programs classification. Taking the
     cardiolog's as a gold standard we aim to minimise this difference
     by means of machine learning tools."

     The names and id numbers of the patients were recently 
     removed from the database.


    
<b> Notes on training unbalanced, multi-class problem with a small dataset</b>     
There are only 452 samples of data

There are 256 features

There are 16 classes, 13 of which have sample data. Classes are extrememly unbalanced



<b> Ways to handle similar problems</b>
This problem can be solved using augmented data

 - create a bigger training set by duplicating data and adding noise

 - rebalance the output classes to be somewhat equal in number


<b> Results</b>
With out rebalancing data 100% accuracy can be obtained using a minimum of 2 - 256 dense layers + 1 output layer and about 20 training epochs


<b> Possible problems</b>

The sample set is so small it might not be representative of larger data in a real situation. This assumes the data given in the problem is representative: ( has same class distribution, other samples will be similar except for 1% noise)
