# CLASSIFIER:	DECISION TREE ALGORITHM

## Introduction

An implementation of a classifier using the Decision Tree algorithm. The algorithm is tested on data and result metrics are provided. The model used performed suitable on the acting an expert system for determining decision to buy a vehicle.

## Data

The data used is car evaluation data collected from the University of California Irvine (UCI) machine database. The data was originally developed for a presentation of DEX, M. Bohanec, V. Rajkovic, titled ^["Expert system for decision making". Sistemica 1(1), pp. 145-157, 1990.). The model evaluates cars according to the following concept structure ]

### Data Structure

The structure of the data is as follows:

Seven columns containing discrete data values of the number of the buying price of the vehicle, maintenance cost, number of doors, number of persons/passengers, luggage boot capacity, safety rating, and expert decision to buy a vehicle or not. All of which contain 1728 data records, see image below.

The multiple features and several values within each feature make this dataset appropriate for classification using the decision trees algorithm.

![Data Structure]("./img/two.png")

In my evaluation of this dataset, I would be using the decision as my label / predicted values, while all other attributes of the vehicles would be used as features.

### Data Collection

The data is downloaded through a link to the UCI database available on Kaggle. The downloaded data when unzipped contains a Comma Seperated Value file containing all the data records respectively. However, the column heads are missing. In the data cleaning Section, column heads retrieved from the data description are placed into the data.

### Data Cleaning

The Pandas framework is used for data cleaning. The framework uses unique data objects called DataFrame and Series. Pandas Series represents one column/field of data, While a DataFrame represents a collection of several Pandas series.

First, the data is imported and column heads are attached to the data using the command.
pd.read_csv(file = "", names = "")
Column heads are passed as the names argument.

Next, the data is inspected using the command:
data.info()

Then, it is converted to a python dictionary that the model uses as a way to circumvent the use of data wrappers.

Finally, the data can be passed into the model.

## Algorithm

Decision Tree is a supervised learning technique that represents the decision and decision making process graphically. This predictive modelling technique is applied in statistics, data mining and machine learning. The tree designed in this project is a CART (Classification and Regression Tree) which allows its leaves to evaluate discrete and continous data.

### Information Gain

Several mathematical and computation technique come to play in the development of this algorithm. One very important principle is information gain. Information gain refers to the ammount of useful information a variable can gain by observing another variable.

Mathematically,

$$ Gain*{buying price,boot size}{(buying price,boot size)}=F*{\text{KL}}{\left(P*{buying price}{(buying price|boot size)}\|P*{buying price}{(x|I)}\right)},}$$

Information gain is deduced from information entropy which refers the ammount of impurity within a sample population. The gain is used to select what features would be next choosen as criteria for the decision.
Mathematically,

$$Entropy = -sum_{i=1}^{unique\:values} (-Pr(a_i)log_{2}(a_i))$$

### Depth first search

Thus, a recursive depth first search is implemented to look at nodes within the tree and evaluate information gain for next set of possible nodes/features. Once selected, another search is performed passing information of the previous for computaion
Mathematically,

$$gain = gain_{parent} - gain_{child\:or\:node}$$

$$ f({search}) = max\_{feature} (Compute\:gain) $$

$$ f*{recursive fit}({parent\:node}) = f*{recursive\:fit}({child\:node|parent\:node}) $$

### Implementation

    class DecisionTree

- Created a Decison tree object that holds several functionalities

  def fit

- Trains the by accepting data as argument, initiating attributes of the tree e.g features, label, unique label. The function searchs features (provided categorical features have not been searched before) computes a gain value for each feature and compares. The maximum gain is choosen and its feature is stored. Data is the split based on this feature and a recursive call to search data is made again. Finally, the function return a list, result, containing feature, value (relevant for numerical / continous features), and gain. At the top-most call, a list containing the result and dictionary containing the results and dictionary of lower-level nodes is added to the tree representation.

  def "**str**"()

- The function is used to print the tree representation

  def predict

- Given a dictionary of records feature, the tree predicts (makes a decision) on its label.

### Testing

The unit test is prepared for all functions used in implementing the algorithm. Thus, the Pytest framework is used to implement these tests. The implementation passed all unit tests modules as described in the image below.

![unit test]("./img/one.png")

## Results

Using a tree depth of 6 and 10% of the data as test data and the rest as training data,
An accuracy of 91.33% was obtained on the dataset, as described in the image below:

![Training Result]("./img/three.png")

## References

- [Performance Classification and Evaluation of J48 Algorithm and Kendall's Based J48 Algoritthm](https://www.periyaruniversity.ac.in/ijcii/issue/marnew/2_mar_18.pdf)
- [j48 Classification c4-5 Algorithm In a Nutshell](https://medium.com/@nilimakhanna1/j48-classification-c4-5-algorithm-in-a-nutshell-24c50d20658e)
- [Book Review: C4.5: Programs for Machine learning by J. Ross Quinlan, Morgan Kaufmann Publishers Inc. (1993)]()
- [Car Evaluation Database was derived from a simple hierarchical decision model originally developed for the demonstration of DEX, M. Bohanec, V. Rajkovic: Expert system for decision making. Sistemica 1(1), pp. 145-157, 1990.)] (http://archive.ics.uci.edu/ml/datasets/machine-learning-databases/car/)
- [Information Gain on Decision Trees](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)
  $$
