import numpy as np
import graphviz as gv
from math import ceil

import pandas as pd


def class_counts(data):
    """Counts the number of each type of example in a dataset."""
    counts = {'Y': len(data.r_data), 'N': data.v_data}  # a dictionary of label -> count.

    return counts


class Cut:
    """A Cut is used to partition a dataset.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = ">="
        return "Is %s col \\%s %s?" % (
            self.column, condition, str(round(self.value, 2)))

    def get_cut_values(self, data):
        """Returns unique values in lower density data after previous split"""
        if self.value is not None:
            lower_density_set = data.get_lower_density_set(self)
            return np.unique(lower_density_set.r_data) if lower_density_set is not None else None
        else:
            return np.unique(data.r_data)  # unique values in the column


def entropy(data):
    counts = class_counts(data)
    entro = []
    for lbl in counts.keys():
        prob_of_lbl = counts[lbl] / float(len(data.r_data) + data.v_data)
        entro.append(prob_of_lbl * np.sqrt(prob_of_lbl))
    return sum(entro) * -1


def info_gain(lhs, rhs, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    pl = float(len(lhs.r_data) + lhs.v_data) / (len(lhs.r_data) + len(rhs.r_data) + lhs.v_data + rhs.v_data)
    pr = float(len(rhs.r_data) + rhs.v_data) / (len(lhs.r_data) + len(rhs.r_data) + lhs.v_data + rhs.v_data)
    info = current_uncertainty - pl * entropy(lhs) - pr * entropy(rhs)
    return info


class Data:
    def __init__(self, r_data, v_data=None):
        self.r_data = r_data
        self.v_data = len(r_data) if v_data is None else v_data

    def get_dim_cut(self, cut, current_uncertainty, third_cut_values=None):
        """Returns a Question which gives a region with lower density in a particular dimension"""
        best_dim_gain = -1
        best_dim_cut = None

        if third_cut_values is None:
            values = cut.get_cut_values(self)
        else:
            values = third_cut_values

        col = cut.column

        if values is None or len(values) == 1:
            return None

        for val in values:  # for each value
            new_cut = Cut(col, val)

            # try splitting the dataset
            lhs_data, rhs_data = self.partition(new_cut)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(lhs_data.r_data) == 0 or len(rhs_data.r_data) == 0:
                continue

            # Calculate the information gain from this split
            dim_gain = info_gain(lhs_data, rhs_data, current_uncertainty)

            if dim_gain > best_dim_gain:
                best_dim_gain, best_dim_cut = dim_gain, new_cut

        return best_dim_cut

    def get_third_cut(self, first_cut, second_cut, current_uncertainty):
        """Returns third split or None according weather it can find it or not"""

        # try splitting the dataset
        lhs_data, rhs_data = self.partition(second_cut)

        # Skip this split if it doesn't divide the
        # dataset.
        if len(lhs_data.r_data) == 0 or len(rhs_data.r_data) == 0:
            return None

        def get_distance_from_value(data, value):
            d1 = abs(value - min(data))
            d2 = abs(value - max(data))
            return min(d1, d2)

        def find_third_cut(interim_region, oth_reg):
            if interim_region.check_density() < oth_reg.check_density():
                lower_density_set = self.get_lower_density_set(second_cut)
                if lower_density_set is None:
                    return None

                return self.get_dim_cut(second_cut, current_uncertainty, np.unique(lower_density_set.r_data))
            else:
                return None

        lhs_distance = get_distance_from_value(lhs_data.r_data, first_cut.value)
        rhs_distance = get_distance_from_value(rhs_data.r_data, first_cut.value)

        if lhs_distance < rhs_distance:
            third_cut = find_third_cut(lhs_data, rhs_data)
        else:
            third_cut = find_third_cut(rhs_data, lhs_data)

        return third_cut

    def find_best_cut(self):
        best_cut = None  # keep train of the feature / value that produced it
        current_uncertainty = entropy(self)
        n_features = len(self.r_data.T)  # number of columns

        for col in range(n_features):  # for each feature
            data_dim = Data(self.r_data[:, col], self.v_data)

            # split 1
            first_cut = Cut(col, None)
            first_cut = data_dim.get_dim_cut(first_cut, current_uncertainty)

            if first_cut is None:
                continue

            # split 2
            second_cut = data_dim.get_dim_cut(first_cut, current_uncertainty)

            if second_cut is None:
                best_cut = self.get_lower_density_cut(first_cut, best_cut)
                continue

            # split 3
            third_cut = data_dim.get_third_cut(first_cut, second_cut, current_uncertainty)

            if third_cut is None:
                best_cut = self.get_lower_density_cut(second_cut, best_cut)
            else:
                best_cut = self.get_lower_density_cut(third_cut, best_cut)

        return best_cut

    def partition(self, cut):
        """Splits the dataset by question"""

        def v_points_split(data, value, col=None):
            """Splits virtual points in order of splitting the dataset"""
            if col is not None:
                values = np.unique(data.r_data[:, col])
            else:
                values = np.unique(data.r_data)

            min_val = np.min(values)
            max_val = np.max(values)
            rhs_v_p = round(abs(data.v_data * (value - min_val) / (max_val - min_val)))
            rhs_v_p = 1 if rhs_v_p == 0 else rhs_v_p
            lhs_v_p = data.v_data - rhs_v_p

            return lhs_v_p, rhs_v_p

        def add_v_points(lhs_r_p, rhs_r_p, lhs_v_p, rhs_v_p):
            if lhs_v_data < len(lhs_r_p):
                lhs_v_p = len(lhs_r_p)
            if rhs_v_data < len(rhs_r_p):
                rhs_v_p = len(rhs_r_p)

            return lhs_v_p, rhs_v_p

        if len(self.r_data.shape) == 1:
            lhs_r_data, rhs_r_data = self.r_data[self.r_data >= cut.value], self.r_data[self.r_data < cut.value]
            lhs_v_data, rhs_v_data = v_points_split(self, cut.value)
        else:
            col = cut.column
            lhs_r_data, rhs_r_data = self.r_data[self.r_data[:, col] >= cut.value], \
                                     self.r_data[self.r_data[:, col] < cut.value]
            lhs_v_data, rhs_v_data = v_points_split(self, cut.value, col=col)

            lhs_v_data, rhs_v_data = add_v_points(lhs_r_data, rhs_r_data, lhs_v_data, rhs_v_data)

        return Data(lhs_r_data, lhs_v_data), Data(rhs_r_data, rhs_v_data)

    def get_lower_density_cut(self, first_cut, second_cut):
        """Returns question which gives us better region (with lower density)"""
        if first_cut is None: return second_cut
        if second_cut is None: return first_cut

        cuts_list = [first_cut, second_cut]
        lower_density_cut, lower_density = None, None
        for cut in cuts_list:
            lhs_data, rhs_data = self.partition(cut)
            density = min(lhs_data.check_density(), rhs_data.check_density())
            if lower_density is None or lower_density > density:
                lower_density_cut, lower_density = cut, density

        return lower_density_cut

    def get_lower_density_set(self, cut):
        """Returns data which has lower density after split"""
        lhs_data, rhs_data = self.partition(cut)
        if lhs_data.v_data == 0 or rhs_data.v_data == 0:
            return None
        if lhs_data.check_density() < rhs_data.check_density():
            return lhs_data
        else:
            return rhs_data

    def check_density(self):
        """Computes relative density in the region"""
        return len(self.r_data) / self.v_data


class Node:
    def __init__(self,
                 data,
                 cut,
                 true_branch,
                 false_branch):
        self.data = data
        self.cut = cut
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.stop = False
        self.min_y = None

    def params_(self, min_y):
        min_y = int(round(min_y * len(self.data.r_data)))
        return min_y

    def prune(self, min_y, min_rd, score=None):
        self.min_y = self.params_(min_y)
        self.prune_(self.min_y, min_rd)
        return self

    def prune_(self, min_y, min_rd):
        """Pruning algorithm for CLTree"""
        if isinstance(self, Leaf):
            return self
        else:
            if len(self.true_branch.data.r_data) < min_y:
                self.true_branch = Leaf(self.true_branch.data)
            else:
                if not isinstance(self.true_branch, Leaf):
                    self.true_branch = self.true_branch.prune_(min_y, min_rd)

            if len(self.false_branch.data.r_data) < min_y:
                self.false_branch = Leaf(self.false_branch.data)
            else:
                if not isinstance(self.false_branch, Leaf):
                    self.false_branch = self.false_branch.prune_(min_y, min_rd)

            if isinstance(self.true_branch, Leaf):
                if isinstance(self.false_branch, Leaf):
                    if len(self.false_branch.data.r_data) / self.false_branch.data.v_data > min_rd:
                        return Leaf(self.data)
                    elif self.true_branch.data.v_data > len(self.true_branch.data.r_data):
                        return Leaf(self.data)
                    else:
                        return self
                else:
                    return self
            else:
                return self

    def get_clusters(self):
        cluster_list = self.get_clusters_(self.min_y)
        cl_with_labels = np.hstack((self.data.r_data, np.full((len(self.data.r_data), 1), -1)))
        c = 0
        for cl in cluster_list:
            for a in cl:
                arr = (self.data.r_data == a).all(axis=1)
                i = np.where(arr == True)
                cl_with_labels[i, -1] = c
            c += 1

        return cl_with_labels[:, -1].astype(int)

    def get_clusters_(self, min_y):
        if isinstance(self, Leaf):
            if self.predictions['Y'] / self.predictions['N'] >= 1 and len(self.data.r_data) >= min_y:
                return [self.data.r_data]
            return []

        cl1 = self.true_branch.get_clusters_(min_y)
        cl2 = self.false_branch.get_clusters_(min_y)

        return cl1 + cl2

    def plot(self):
        def trace(node, dot):
            q_true = str(node.true_branch.cut) if isinstance(node.true_branch, Node) else None
            q_false = str(node.false_branch.cut) if isinstance(node.false_branch, Node) else None
            r_data_true = len(node.true_branch.data.r_data)
            v_data_true = node.true_branch.data.v_data
            r_data_false = len(node.false_branch.data.r_data)
            v_data_false = node.false_branch.data.v_data

            if isinstance(node.true_branch, Leaf):
                dot.node(str(id(node.true_branch)),
                         label='Leaf\\nR: {} V: {}'.format(r_data_true, v_data_true))
            else:
                dot.node(str(id(node.true_branch)),
                         label='Node\\n{}\\nR: {} V: {}'.format(q_true, r_data_true, v_data_true))

            if isinstance(node.false_branch, Leaf):
                dot.node(str(id(node.false_branch)),
                         label='Leaf\\nR: {} V: {}'.format(r_data_false, v_data_false))
            else:
                dot.node(str(id(node.false_branch)),
                         label='Node\\n{}\\nR: {} V: {}'.format(q_false, r_data_false, v_data_false))

            dot.edge(str(id(node)), str(id(node.true_branch)), color='green')
            dot.edge(str(id(node)), str(id(node.false_branch)), color='red')

            if not isinstance(node.true_branch, Leaf):
                trace(node.true_branch, dot)
            if not isinstance(node.false_branch, Leaf):
                trace(node.false_branch, dot)

        dot = gv.Digraph(filename='cltree', format='svg', node_attr={'shape': 'record'})
        dot.node(str(id(self)), label='Node\\n{}\\nR: {} V: {}'.format(self.cut, len(self.data.r_data), self.data.v_data))
        trace(self, dot)
        dot.render(directory='doctest-output', view=True)


class Leaf:
    def __init__(self, data):
        self.predictions = class_counts(data)
        self.data = data
        self.stop = False

    def get_clusters_(self, min_y):
        if self.predictions['Y'] / self.predictions['N'] >= 1 and len(self.data.r_data) >= min_y:
            # cluster_list.append(self.data.r_data)
            return [self.data.r_data]
        return []

class CLTree:
    def __init__(self):
        pass

    def build(self, data):
        cut = data.find_best_cut()

        if cut is None:
            return Leaf(data)

        lhs, rhs = data.partition(cut)

        true_branch = self.build(lhs)

        false_branch = self.build(rhs)

        return Node(data, cut, true_branch, false_branch)
