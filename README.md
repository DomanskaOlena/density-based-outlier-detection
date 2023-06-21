# density-based-outlier-detection
*A supervised approach based on Virtual points for outlier detection inspired by Decision Tree algorithm*

Outlier detection is an important task in data analysis and decision making because it helps identify and analyze data points or observations that deviate significantly from the rest of the data. These outliers may indicate potential fraud, equipment failures, disease outbreaks, or other anomalies that require attention or corrective action.

The unsupervised approach to outlier detection is often preferred because it does not require labeled data, meaning that the algorithm can identify outliers in a dataset without having to know beforehand which observations are anomalous. This is particularly useful when there is limited prior knowledge about the data, or when the anomalous observations are rare or unknown. 

We propose a novel approach to outlier detection which doesn’t require labeled data but is based on the supervised learning technique similar to decision tree construction. Our method overcomes many of the limitations of traditional outlier detection techniques by partitioning the space into cluster and empty regions using a decision tree. We accomplish this by introducing virtual data points and modifying the decision tree algorithm accordingly. The results of our experiments on synthetic and real-world datasets show that the method is both highly efficient and scalable.

**Repo contents**

Files in the repository:

*DBOD_avenga.py* - the implementation of the method

*DBOD_uml.png* -  UML (Unified Modeling Language) diagram which visually represents an implementation in *DBOD_avenga.py* along with its classes, in order to better understand it.

*comparison.ipynb* - Jupyter notebook with the comparison of the performance of our density-based outlier detection method (DBOD) and other existing approaches






