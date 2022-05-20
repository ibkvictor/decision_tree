from copy import deepcopy
import math
from pprint import pformat, pprint

class DecisionTree():
    """
        crude decision tree implemetation using native python
        classification decision tree for descrete data
        functions:
            str()
            recursive_fit()
    """
    def __init__(self, depth):
        self.count = depth
        self.data = None
        self.features = None
        self.num_records = None
        self.representation = []
        self.unique_labels = None
        self.label = None
        return None

    def __str__(self, representation = None, level = 0):
        """
            Implemetation of the special function __str__
            Allow objects of tree values to be printed using print() and converted to string using str()
            level represents number of parents leaf node has

        """
        return pformat(self.representation)

    def fit(self, features, labels):
        """
            trains the model on data provided
            arguments
            features: selected features (i.e x)
            labels: selected label (i.e y)
        """
        self.data = features
        self.features = list(features.keys())
        self.data.update(labels)
        self.label = list(labels.keys()).pop()
        self.unique_labels = set(labels[self.label])
        self.recursive_fit(self.data, self.count)
        self.num_records = len(self.data[self.label])


    def recursive_fit(self, data, level, tested = [], prev_gain = 1):
        """
            Performs recursive fit over training data using information gain theory
            arguments:
            data: dict containing features and labels
            tested: list containing names of discrete features already tested
            prev_again: returns the information gain of parent node

            return:
            max(data[self.label], key = data[self.label].count) (str) the most occuring label value in the label column

            result store: list of list and dict containing:
                list: 
                    feature with most gain selected as node feature,
                    value of feature which is None for discrete feature and selected feature divisor for continuous features.
                
                dict:
                    unique values of features
                    which contain result_store at child / lower-level withing the tree
        """
        if level < 0: # if data is empty or depth is exceeded
            return max(data[self.label], key = data[self.label].count) # return most occuring label value
        if len(tested) == len(self.features): #if all features have been tested
            return max(data[self.label], key = data[self.label].count)
        store = [None, None, -math.inf] # contains feature, val (continuous), gain, correspondingly

        for f in self.get_features(data):
            feature_val_gain = None
            if self.discrete_check(f):
                if not f in tested:
                    feature_val_gain = self.search(f, {f: data[f], self.label: data[self.label]}, prev_gain) # has no val
                else:
                    continue
            else:
                feature_val_gain = self.n_search(f, {f: data[f], self.label: data[self.label]}, prev_gain) # has a val for binary split
            if store[2] <= feature_val_gain[2]:
                store = feature_val_gain


        if (not store[0]) or (store[2]<0): # if no feature is selected or no information was gain i.e negative gain
            return max(data[self.label], key = data[self.label].count) # convert node to leaf

        if self.discrete_check(store[0]):
            result_store = [store, {i:[] for i in set(self.data[store[0]])}]
            tested.append(store[0])
        else:   # for continious values divide by value into less than and greater than left and right correspondingly
            result_store = [store, {i:[] for i in ["l", "r"]}] # l,r representing less than and greater than

        # if len(tested) == len(self.features): #if the number of featues 
        #     return
        
        t = self.split_data(data, store[0], store[1])
        for i in t: # splits data into for leaves to begin searching
            label_value = self.recursive_fit(t[i], level -1, tested.copy(), store[2])
            result_store[1][i].append(label_value)
        for i in result_store[1]:
            if not result_store[1][i]: # if node was empty
                result_store[1][i].append(max(data[self.label], key = data[self.label].count)) # convert to leaf
        if level == self.count:
            self.representation = result_store
        return result_store

    def split_data(self, data, f, value = None):
        """
            splits data according to selected feature and its value
            arguments:
                data: data to be split
                f: feature criteria
                value: value of feature to split based on. It is None for categorical features and a float value for discrete features.
            returns:
                data_large: 
                    refers to dict containing data of feature provided in argument split by value.
                    each of the values of the dict contains records not that do not possess attributes described by the key.
        """
        if value:
            # evaluates for numerical features
            data_l = deepcopy(data)
            data_r = deepcopy(data)
            for index in reversed(range(len(data[f]))):
                if data[f][index] < value:
                    for all_f in self.features:
                        del data_r[all_f][index]
                    data_r[self.label].pop(index)
                else:
                    for all_f in self.features:
                        del data_l[all_f][index]
                    data_l[self.label].pop(index)
            return {"l": data_l, "r": data_r}
        else:
            # evaluates for categorical / discrete features
            data_large = {i: deepcopy(data) for i in set(data[f])}
            f_keys = list(data_large.keys())
            occurences = {k: [] for k in f_keys}
            for index in range(len(data[f])):
                for i in f_keys:
                    if data_large[i][f][index] != i:
                        occurences[i].append(index)
            for i in occurences:
                for index in sorted(occurences[i], reverse=True):
                    for all_f in self.features:
                        del data_large[i][all_f][index]
                    del data_large[i][self.label][index]
            return data_large

    def get_features(self, data):
        """
            The function gets all the names of all the features left in the data.
            arguments:
                data: data provided as dict containing records left.
            return:
                features: a list of unique features of provided data.
        """
        features = list(data.keys())
        features.remove(self.label)
        return features

    def search(self, feature, data, prev_entropy):
        """
            Computes determine information gain on each feature using information theory formulae
            arguments:
                feature: (str) feature (column / field) to be evaluated 
                data: (dict) provided data
                prev_entropy: (float) previous gain value for precedding parent node.
            return
                list: contains feature, value (only for numerical data), and information gain value
        """
        unique_values = list(set(data[feature]))
        holder = {
            i: {j: 0 for j in self.unique_labels} for i in unique_values
        }
        for j in range(len(data[feature])):
            if data[self.label][j] not in holder[data[feature][j]]:
                holder[data[feature][j]][data[self.label][j]] = 1
            else: 
                holder[data[feature][j]][data[self.label][j]] += 1
        can = [self.sum_entropy(list(holder[val].values())) * (sum(list(holder[val].values())) / len(data[feature])) for val in unique_values]
        return [feature, None, (prev_entropy - sum(can))]


    def n_search(self, feature, data, prev_imp_val):
        """
            Numerical feature search for best split value (most gain).
            Computes determine information gain on each numerical feature using information theory formulae
            arguments:
                feature: (str) feature (column / field) to be evaluated 
                data: (dict) provided data
                prev_entropy: (float) previous gain value for precedding parent node.
            return
                list: contains feature, value to split by (only for numerical data), and information gain value
        """
        unique_values = self.n_generate(data[feature])
        holder = {
            val: [{label: 0  for label in self.unique_labels}, {label: 0  for label in self.unique_labels}] for val in unique_values
        }
        max_v_e = [None, None, -math.inf]
        for val in unique_values:
            for j in range(len(data[feature])):
                if data[feature][j] <= val:
                    holder[val][0][data[self.label][j]] += 1
                else:
                    holder[val][1][data[self.label][j]] += 1

            l_sums = list(holder[val][0].values())
            r_sums = list(holder[val][1].values())
            l_entropy = self.sum_entropy(l_sums) * sum(l_sums)/len(data[feature])
            r_entropy = self.sum_entropy(r_sums) * sum(r_sums)/len(data[feature])

            gain = prev_imp_val - (l_entropy + r_entropy)
            if max_v_e[2] <= gain:
                max_v_e = [feature, val, gain]
        return max_v_e

    def n_generate(self, feature_data):
        """
            Generated average between two adjacent unique values in sorted feature column.
            argument: 
                feature_data: list contains an entire column (feature) data.
            return:
                list: containing averages.
        """
        # the sorted unique continuous values in each feature
        unique_vals = list(set(feature_data))
        return [(unique_vals[i] + unique_vals[i+1])/2 for i in range(len(unique_vals) - 1)]
    
    def sum_entropy(self, n_list):
        """
            computes entropy value for provided sample list with frequecies of values in each label

            arguments
            n_list: sample size list for each label

            variables
            n_t: total size of 
        """
        sum_entropy = 0
        n_t = sum(n_list) # total population
        for n_s in n_list:
            
            if n_s == 0: # 0 log 0 is undefined, hence reduce to zero.
                sum_entropy += 0
            else:
                sum_entropy += -(n_s/n_t) * math.log(n_s/n_t, 2)
        return sum_entropy

    def discrete_check(self, feature):
        """
            argument
            feature: the feature to the categories as numerical or categorical
            
            return
            boolean: True or False to represent if a feauture (data field) is categorical (discrete) or numerical (continuous)
        """
        if isinstance(self.data[feature][0], str):
            return True
        return False

    def predict(self, record_features, value = None):
        """
            Give a record of just features, provide its corresponding label
            arguments: 
                record_features: (dict) a slice /record of test data with provided features excluding the label column.
                
            return:
                label: (str) corresponding label value for the record.
        """

        def search_predict(record_features, representation = self.representation):
            """
                depth first search for label given features
            """
            if len(representation) == 1 and isinstance(representation, list):
                return search_predict(record_features, representation[0])
            elif len(representation) == 1:
                return representation[0]
            elif isinstance(representation, str):
                return representation
            else:
                if self.discrete_check(representation[0][0]):
                    return search_predict(record_features,representation[1][record_features[representation[0][0]]])
                else:
                    print(record_features[representation[0][0]])
                    print(representation[0][1])
                    print(representation[1].keys())
                    if record_features[representation[0][0]] <= representation[0][1]:
                        print(record_features[representation[0][0]])
                        print(representation[1]["l"])
                        return search_predict(record_features,representation[1]["l"])
                    else:
                        print(record_features[representation[0][0]])
                        print(representation[1]["r"])
                        return search_predict(record_features,representation[1]["r"])
        return search_predict(record_features)
