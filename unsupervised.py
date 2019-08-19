import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import distance_matrix
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image,ImageSequence
from sklearn.linear_model import LinearRegression
from sklearn.cluster import MeanShift, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn import tree
import graphviz


##################################################################################
#                    Unsupervised learning: Dimension Reduction                  #
##################################################################################
def dimension_reduction(df, sample_limit, category_feature, method="MDS", n_components=3, n_jobs=2, whiten=True):
    '''
    The function is used to conduct multidimentional scaling
    Inputs:
        df: dataframe
        sample_limit: given restriction for the size of rows
        category_features: feature used as predefined label
        methods: "MDS" for multidimensional scaling, "PCA" for principal component analysis
        n_components: the final dimensions
        n_jobs: parallel computing factor
        whiten: True if removing relative variance between components
    Returns: numpy array with n dinmensions and labels, index for the label
    '''
    if df.shape[0] > sample_limit:
        sub_df = df.sample(n=sample_limit).reset_index()
    else:
        sub_df = df.reset_index()

    used_columns = list(sub_df.columns)
    if category_feature:
        used_columns.remove(category_feature)
    sub_dfm = np.matrix(sub_df[used_columns])

    if method == "MDS":
        similarities = euclidean_distances(sub_dfm)
        mds = manifold.MDS(n_components=n_components, max_iter=3000, eps=1e-9,
                           dissimilarity='precomputed', n_jobs=1)
        pos = mds.fit(similarities).embedding_

    else:
        pca = PCA(n_components=n_components, copy=True, whiten=whiten)
        pos = pca.fit_transform(sub_dfm)
        print(pca.explained_variance_ratio_) 

    if category_feature:
        category_index = {}
        sub_df["label"] = 0
        for i, category in enumerate(sorted(list(sub_df[category_feature].unique()))):
            sub_df.loc[sub_df[category_feature] == category, "label"] = i
            category_index[category] = i

        new_pos = np.zeros((pos.shape[0], n_components+1))
        new_pos[:,:-1] = pos
        new_pos[:,-1] = sub_df["label"]

    return new_pos, category_index


def detecting_outliers(pos, extreme_perc, normalize=False):
    '''
    This function detects the outliers by comparing the distribution of
    the distance and the residual comparing to the fitted line of the
    components.
    Inputs:
        pos: dimensional reduced array
        extreme_perc: the percentage for the threshold of the outliers
        normalized: True if normalizing the coefficient of the fitted line
    Return: 
        df: dataframe for the residuals and distance
        indices excluding the outliers
    '''
    pos = pos[:, :-1]
    X = pos[:, :-1]
    y = pos[:, -1]
    reg = LinearRegression(normalize=normalize)
    reg.fit(X, y)
    pred_y = reg.predict(X)
    resid = abs(pred_y - y)
    dis = abs(reg.intercept_ + sum(np.dot(X, reg.coef_)) - y) / ((sum(reg.coef_ ** 2) + (-1) ** 2) ** 0.5)
    df = pd.DataFrame(data={"resid": resid, "dis": dis})
    ext_indices = df.loc[((df.resid < np.percentile(resid, extreme_perc)) | 
                          (df.resid > np.percentile(resid, 100 - extreme_perc))) & 
                         ((df.dis < np.percentile(dis, extreme_perc)) | 
                          (df.dis > np.percentile(dis, 100 - extreme_perc)))].index
    #df.loc[~ext_indices]
    return (df, set(df.index) - set(ext_indices))
    

def depict_mds_plot(pos, angle):
    '''
    This function is used to depict 3D gif plot for the MDS result with
    three dimensions
    Inputs:
        pos: the datframe with n dimensions
        angle: the angle variation of each png
    '''
    set_dict = {}
    label = pd.DataFrame(pos[:,-1]).rename(columns = {0:"label"})
    for cat_index in list(label["label"].unique()):
        set_dict[cat_index] = set(label[label["label"] == cat_index].index)

    color = ["b", "g", "r", "c", "m", "y", "k"]
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    for i in range(pos.shape[0]):
        x3=pos[i,0]
        y3=pos[i,1]
        z3=pos[i,2]
        for category_index, sets in set_dict.items():
            if i in sets:
                ax.scatter(x3,y3,z3,c=color[int(category_index)])

    for angle in range(0,360, angle):
        ax.view_init(30, angle)
        plt.savefig(r'D:\Project Data\ML project\mds_plot%d'%angle)

    seq=[]
    for i in range(0,360, angle):
        im=Image.open(r'D:\Project Data\ML project\mds_plot%d.png'%i)
        seq.append(im)
        seq[0].save(r'D:\Project Data\ML project\mds_plot.gif',save_all=True,append_images=seq[1:])


##################################################################################
#               Unsupervised learning: Clustering Models & Grid                  #
##################################################################################
def k_mean_analysis(df, clusters_list, label_col, centroids=False, grid=False):
    '''
    This function implement k-mean clustering for given number of clusters
    Inputs:
        df: dataframe
        clusters: number of clusters
        label_col: name of the label column
        centroids: True if including centroids in the result
        grid: True if using all the value in the clusters_list
              False if using the first parameter inthe list
    Returns: used methods, dataframe with clustered label and dictionary
             of centroids information
    '''
    kmean_dict = {}

    # For gird
    if grid:
        for clusters in clusters_list:
            kmeans = KMeans(n_clusters=clusters)
            kmeans.fit(df)
            df[label_col] = pd.Series(kmeans.labels_)
            kmean_dict[clusters] = [kmeans, df]

        if centroids:
            clusters_result = {}
            for kmean, info in kmean_dict.items():
                centroids_dict = {}
                for label in sorted(pd.unique(kmeans.labels_)):
                    centroids_dict[label] = kmeans.cluster_centers_[label]
                clusters_result[kmean] = info + [centroids_dict]

            return clusters_result

        return kmean_dict

    # For single case
    else:
        clusters = clusters_list[0]
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(df)
        df[label_col] = pd.Series(kmeans.labels_)

        if centroids:
            centroids_dict = {}
            for label in sorted(pd.unique(kmeans.labels_)):
                centroids_dict[label] = kmeans.cluster_centers_[label]

            return [kmeans, df, centroids_dict]

        return [kmeans, df]


def mean_shift_analysis(df, label_col, bandwidth_list=[0.6, 0.8, 1, 1.2, 1.4], centroids=False, grid=False):
    '''
    The function provide clustering operation for mean shift method
    Inputs:
        df: dataframe
        label_col: column used to store the label
        bandwidth_list: list of bandwidth for the methods
        centroids: True if including centroids information in the result
        grid: True if using all the parameters in the list
              False if using the first parameter inthe list
    Returns: used methods, dataframe with clustered label and dictionary
             of centroids information
    '''
    bandwidth_dict = {}

    # for grid
    if grid:
        for bandwidth in bandwidth_list:
            ms = MeanShift(bandwidth=bandwidth, n_jobs=2)
            ms.fit(df)
            df[label_col] = pd.Series(ms.labels_)
            bandwidth_dict[bandwidth] = [ms, df]

        if centroids:
            bandwidth_result = {}
            for bandwidth, info in bandwidth_dict.items():
                centroids_dict = {}
                for label in sorted(pd.unique(ms.labels_)):
                    centroids_dict[label] = ms.cluster_centers_[label]
                bandwidth_result[bandwidth] = info + [centroids_dict]

            return bandwidth_result

        return bandwidth_dict

    # for single case
    else:
        bandwidth = bandwidth_list[0]
        ms = MeanShift(bandwidth=bandwidth, n_jobs=2)
        ms.fit(df)
        df[label_col] = pd.Series(ms.labels_)

        if centroids:
            centroids_dict = {}
            for label in sorted(pd.unique(ms.labels_)):
                centroids_dict[label] = ms.cluster_centers_[label]

            return [ms, df, centroids_dict]

        return [ms, df]


def plot_cluster(df, label_col):
    '''
    This function is used to provide 2D plots for clustering methods
    Inputs:
        df: dataframe with columns used for plotting
        label_col: columns for label storage
    '''
    df_col = list(df.columns)
    col1, col2 = df_col[0], df_col[1]
    groups = df.groupby(label_col)

    if len(df_col) > 3:
        print("not allow to plot in 2D")
        return df

    fig, ax = plt.subplots()
    for pred_class, group in groups:
        ax.scatter(group[col1], group[col2], label=pred_class)
    ax.legend()
    plt.show()


def method_grid(df, label_col, method, clusters_list=[2, 5, 10, 100], centroids=True,  
                bandwidth_list=[0.6, 0.8, 1, 1.2, 1.4], grid=False):
    '''
    The function takes the general parameters for clustering methods and update the parameters
    used in the method grid
    Inputs:
        df: dataframe
        label_col: column used as label storage
        method: "kmean" for kmean analysis, "mean_shift" for mean shift analysis
        clusters_list: list of clusters for kmean 
        centroids: True if including centroids in the final results
        bandwidth_list: list of bandwidth used in the mean shift analysis
        grid: True if using all the parameter in the list
              False if using the first parameter inthe list
    Return: methods, dictionary of parameters
    '''
    if method == "kmean":
        met = k_mean_analysis
        kwargs = {"df": df, "clusters_list": clusters_list, "label_col": label_col, "centroids": centroids}

    if method == "mean_shift":
        met = mean_shift_analysis
        kwargs = {"df": df, "label_col": label_col, "bandwidth_list": bandwidth_list}

    if grid:
        kwargs["grid"] = grid

    return met, kwargs
        

def dbscan_analysis(df, label_col, min_sample_list=[5,10,100,1000], eps_list=[0.1, 0.5, 1, 10, 100, 1000], grid=False):
    '''
    The function is used to conduct DBSCAN analysis
    Inputs:
        df: dataframe
        label_col: column used as the storage for labels
        min_sample_list: list of min_sample passed in the method
        eps_list: list of eps used in the method
        grid: True if using all the possible combination of the lists
              False if using the first combination of the lists
    Return: method and labelled dataframe
    '''
    dbs_dict = {}

    # for grid
    if grid:
        for min_sample in min_sample_list:
            for eps in eps_list:
                dbscan = DBSCAN(eps=eps, min_samples=min_sample, n_jobs=2)
                dbscan.fit(df)
                df[label_col] = pd.Series(dbscan.labels_)
                dbs_dict[(min_sample, eps)] = [dbscan, df]

        return dbs_dict

    # for single case
    else:
        eps = eps_list[0]
        min_sample = min_sample_list[0]
        dbscan = DBSCAN(eps=eps, min_samples=min_sample, n_jobs=2)
        dbscan.fit(df)
        df[label_col] = pd.Series(dbscan.labels_)

        return [dbscan, df]


##################################################################################
#                       Summarize Methods for Labels                             #
##################################################################################
def summarize_clusters(df, label_col):
    '''
    This function generate the summary statistics for the clusters
    Inputs:
        df: dataframe
        label_col: the columns denotes the information of the clusters
    Returns: dictionary of dataframes of descriptive statistics
    '''
    label_dict = {}
    for label in list(df[label_col].unique()):
        label_dict["label = " + str(label)] = df[df[label_col] == label].describe()

    return label_dict


def simple_tree(df, features, label_col, max_depth=1, min_samples_split=2):
    '''
    This model is used to provide simple tree analysis to find 
    distinctive features
    Inputs:
        df: dataframe
        features: features list
        label_col: column of the label
        max_depth: max_depth of decision tree
        min_samles_split: the minimun number of sample required to split a node
    Returns: a trained tree model
    '''
    DT = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    DT.fit(df[features], df[label_col])

    return DT


def map_feature_importances(model, features):
    '''
    This function is used to provide feature importances of the
    festures used in the decision tree
    Inputs:
        tree: decision tree object
        features: features used in the decision tree
    Returns: feature importances dictionary
    '''
    feature_dict = {}
    importances = model.feature_importances_
    for i, val in enumerate(features):
        feature_dict[val] = importances[i]

    return feature_dict


#############################################################
# Source: Plotting Decision Trees via Graphviz, scikit-learn
# Author: scikit-learn developers
# Date: 2007 - 2018
#############################################################
def depict_decision_tree(decision_tree, features, classifier, output_filename):
    '''
    This function is used to generate the decision tree graph which
    contains the tree node and related information. The output will be
    saved as pdf file
    Inputs:
        dataframe
        features: list of feature names used in decision tree
        classifier: name of the classifier used in decision tree
        output_filename: the name of the pdf file
    '''
    dot_data = tree.export_graphviz(decision_tree, out_file="tree.dot", \
                                                   feature_names=features,\
                                                   class_names=classifier,\
                                                   filled=True, rounded=True,\
                                                   special_characters=True)
    file = open('tree.dot', 'r')
    text = file.read()
    graph = graphviz.Source(text)  
    graph.render(output_filename) 


##################################################################################
#                  Combine / Recluster / Split Labels (Simple)                   #
##################################################################################
def cluster_combiner(df, label_col, old_labels, new_label_name):
    '''
    This function allow merging clusters to a new cluster
    Inputs:
        df: dataframe
        label_col: column denoted the label
        old_labels: a list of old labels considered as a single cluster
        new_label_name: the name for the new label created
    Return: a dataframe with new labels assigned
    '''
    for label in old_labels:
        df.loc[df[label_col] == label, label_col] = new_label_name

    return df


def recluser(df, label_col, clusters):
    '''
    This function is used to recluster the dataframe with new clusters
    Inputs:
        df: dataframe
        label_col: column of the labels
        clusters: the number of the clusters
    '''
    if len(list(df[label_col].unique())) == clusters:
        return df
    
    df = df.drop([label_col], axis=1)
    df = k_mean_analysis(df, clusters, label_col)

    return df


def cluster_splitter(df, label_col, old_label, clusters):
    '''
    This function is used to split the cluster to several clusters
    Inputs:
        df: dataframe
        label: column denoted the label
        old_label: label used to split
        clusters: number of clusters for the new label
    Return: dataframe with new label assigned
    '''
    sub_df = df[df[label_col] == old_label]
    sub_df = sub_df.drop([label_col], axis=1)
    sub_df = k_mean_analysis(sub_df, clusters, "tmp_" + label_col)
    sub_df["tmp_" + label_col] *= 10
    sub_df["tmp_" + label_col] += old_label

    preserve_df = df[df[label_col] != old_label]
    preserve_df["tmp_" + label_col] = preserve_df[label_col]
    preserve_df = preserve_df.drop([label_col], axis=1)

    new_df = pd.concat([sub_df, preserve_df], join="inner")
    new_df["new_" + label_col] = new_df["tmp_" + label_col]
    new_df = new_df.drop(["tmp_" + label_col], axis=1)

    return new_df


##################################################################################
#                       Recursive Splitter and Combiner                          #
##################################################################################

def cluster_splitter_one_step(df, label_col, old_label, clusters, bandwidth, met, centroids_dict):
    '''
    This function is used to split the given cluster to several new clusters for one step
    Inputs:
        df: dataframe
        label_col: column used for the storage of labels
        old_label: label used to split
        clusters, bandwidth: parameters for the clustering models
        met: methods selected for the split
        centroids_dict: dictionary of centroids
    Return: updated dataframe and updated dictionary of centroids
    '''
    sub_df = df #[df[label_col] == old_label]
    print("the sub_df of ", old_label," in cluster_splitter_one_step function has shape: ", sub_df.shape[0])
    sub_df = sub_df.drop([label_col], axis=1)
    sub_df = sub_df.reset_index()

    if "index" in sub_df.columns:
        sub_df = sub_df.rename(columns={"index": "old_index"})
    old_index = sub_df["old_index"]
    # old index should not be passed in as the variable in the model

    method, kwargs = method_grid(sub_df[sub_df.columns[1:]], label_col, method=met, 
                                 clusters_list=[clusters], centroids=True, 
                                 bandwidth_list=[bandwidth], grid=False)
    methodn, sub_df, current_dict = method(**kwargs)
    sub_df = sub_df.set_index([old_index], drop=True)

    for label in pd.unique(sub_df[label_col]):
        centroids_dict[str(old_label) + "-" + str(label)] = current_dict[label]
        sub_df.loc[sub_df[label_col] == label, label_col] = str(old_label) + "-" + str(label)

    return sub_df, centroids_dict


def cluster_combiner_one_step(df, label_col, old_labels, new_label, centroids_dict):
    '''
    This function allow merging clusters to a new cluster
    Inputs:
        df: dataframe
        label_col: column denoted the label
        old_labels: a list of old labels considered as a single cluster
        new_label: the value for the new label created
        centroids_dict: dictionary of centroids
    Return: a dataframe with new labels assigned, updated centroids_dict
            and name of new label
    '''
    new_centroids = {}
    for label, centroid in centroids_dict.items():
        if label not in old_labels:
            new_centroids[label] = centroid

    new_label = str(label) + "-" + str(new_label)
    new_centroids[new_label] = df[df[label_col].isin(old_labels)].mean()

    for label in old_labels:
        df.loc[df[label_col] == label, label_col] = new_label

    return df, new_centroids, new_label

class unsupervised_tree:
    '''
    Tree object used to store the information of different layers in unsupervised analysis
    Attributes:
        label: label of the node
        parent_label: label of the parent node
        indices: indices of the dataframe corresponding to this node
        centroid: centroid of the node
        layer: the layer for the node in the whole tree
    '''
    def __init__(self, label, parent_label, indices, centroid, layer):
        self.label = label
        self.parent_label = parent_label
        self.indices = indices
        self.centroid = centroid
        self.children = []
        self.layer = layer

    def extend_node(self, df, clusters, bandwidth, label_col, met, layer, centroids_dict):
        '''
        This method is used to extend the children node of the given tree
        Inputs:
            df: dataframe
            clusters, bandwidth: parameters for clustering models
            label_col: column for the storage of labels
            met: method used for clustering
            layer: the layer for the node in the whole tree
            centroids_dict: dictionary of centroids
        '''
        #for par_label in pd.unique(df[label_col]):
        print("the label passed in extend_node function is: ", self.label)
        sub_df, updated_centroids = cluster_splitter_one_step(df, label_col, self.label, 
                                               clusters, bandwidth, met, centroids_dict)
        for new_label in pd.unique(sub_df[label_col]):
            print("new_label ", new_label, " is processed")
            sub_indices = sub_df[sub_df[label_col] == new_label].index
            self.children.append(unsupervised_tree(new_label, self.label, sub_indices, 
                                             updated_centroids[new_label], layer + 1))


def grow_tree(tree, df, clusters, bandwidth, label_col, met, max_layers, 
              group_thres, centroids_dict):
    '''
    This is a recursive function used to create the tree for unsupervised learning method
    Inputs:
        tree: tree node used to extend
        df: dataframe
        clusters, bandwidth: parameters for the clustering methods
        label_col: column for the storage of labels
        met: method used for clustering
        max_layers: max layer for the tree
        group_thres: the min number of indices for a node
        centroids_dict: dictionary of centroids
    '''
    if tree.layer == max_layers:
        print("the tree stop growing since the layer limit is reached")
        return

    if df.shape[0] < group_thres:
        print("the tree stop growing since the minimum size of group is reached")
        return

    else:
        tree.extend_node(df, clusters, bandwidth, label_col, met, tree.layer, centroids_dict)
        print("the current layer is: ", tree.layer, " and the current size of df is: ", len(tree.indices))
        for child in tree.children: # tree children should not be empty
            print ("subtree with label: ", child.label, "is called, and the current size of df is: ", len(child.indices))
            centroids_dict[child.label] = child.centroid
            child_df = df.loc[child.indices]
            print("the size of the child dataframe is: ", child_df.shape[0])
            # child_df is not returned modified in the process
            grow_tree(child, child_df, clusters, bandwidth, label_col, met, 
                                   max_layers, group_thres, centroids_dict)


def get_shortest_distance(centroids_dict):
    '''
    This function calculate the shortest distance of the centroids
    Input: dictionary of centroids
    Return: shortest distance and labels corresponding to them
    '''
    shortest_distances = []
    distance_dict = {}
    for label1, cent1 in centroids_dict.items():
        inner_dis = {}
        distance_dict[label1] = inner_dis
        shortest_distance = float("inf")
        for label2, cent2 in centroids_dict.items():
            #shortest_label = label1
            if label1 != label2:
                distance = sum((np.array(cent1) - np.array(cent2)) ** 2) ** 0.5
                distance_dict[label1][label2] = distance
                if distance < shortest_distance: 
                    shortest_distance = distance
                    shortest_label = label2
        shortest_distances.append((label1, shortest_label, shortest_distance))

    shortest = float("inf")
    shortest_distance_info = tuple()
    for distance_info in shortest_distances:
        if distance_info[2] < shortest:
            shortest = distance_info[2]
            shortest_distance_info = distance_info

    return shortest_distance_info


def combine_node(df, label_col, new_label, node_dict, centroids_dict, layer):
    '''
    This function combines the node and define new parents node in the unsupervised tree
    Inputs:
        df: dataframe
        label_col: column for the storage of labels
        new_label: new_label created
        node_dict: dictionary of the nodes
        centroids_dict: dictionary of the centroids
        layer: the layer for the node in the whole tree
    Returns: updated dataframe, dictionaries of centroids and nodes
    '''
    print("the length of the centroids_dict is: ", len(centroids_dict))
    l1, l2, shortest_distance = get_shortest_distance(centroids_dict)
    old_labels = [l1, l2]
    df, centroids_dict, new_label = cluster_combiner_one_step(df, label_col, old_labels, 
                                                              new_label, centroids_dict)
    new_node_dict = {}
    all_indices = []
    children_nodes = []
    for label, node in node_dict.items():
        if node.label in old_labels:
            all_indices += list(node.indices)
            children_nodes.append(node)
        else:
            new_node_dict[label] = node

    parent_node = unsupervised_tree(new_label, None, all_indices, 
                                centroids_dict[new_label], layer)
    parent_node.children = children_nodes
    new_node_dict[new_label] = parent_node

    return df, new_node_dict, centroids_dict


def cut_tree(df, label_col):
    '''
    The function is used to cut the node from the unsupervised tree object, and merge
    all children nodes to one parent node
    Inputs:
        df: dataframe
        label_col: column used to store the label
    Return: dictionary of nodes (contains the single parent node)
    '''
    node_dict = {}
    centroids_dict = {}
    layer = 0
    for label in pd.unique(df[label_col]):
        indices = df[df[label_col] == label].index
        centroids = df[df[label_col] == label].mean()
        centroids_dict[label] = centroids
        node_dict[label] = unsupervised_tree(label, None, indices, centroids, layer)
    
    new_label = 0
    nodes_set = set(node_dict.keys())
    while len(nodes_set) > 1 :
        print("the current length of the nodes set is: ", len(nodes_set))
        df, node_dict, centroids_dict = combine_node(df, label_col, new_label, 
                                         node_dict, centroids_dict, layer + 1)
        nodes_set = set(node_dict.keys())
        new_label += 1
        if len(centroids_dict) == 1:
            return node_dict

    return node_dict


def tree_summary(tree, df, sum_df):
    '''
    This recursive function is used to get the summary statistics for the unsupervised tree
    Inputs:
        tree: unsupervised tree object
        df: dataframe
        sum_df: summary dataframe
    Return: updated summary dataframe with column means of the nodes
    '''
    new_sum = pd.DataFrame({tree.label: df.loc[tree.indices].mean()})
    sum_df = sum_df.join(new_sum)

    if tree.children == []:
        print("the stopping condition for the function tree_summary is reached")
        return sum_df

    for sub_tree in tree.children:
        sum_df = tree_summary(sub_tree, df, sum_df)

    return sum_df


def tree_label_assignment(tree, df, label_col):
    '''
    The function is used to assign labels to the dataframe according to the information
    of unsupervised tree
    Inputs:
        tree: unsupervised tree
        df: dataframe
        label_col: column for the storage of label
    Return: labelled dataframe
    '''
    df.loc[tree.indices, label_col] = tree.label

    if tree.children == []:
        print("the stopping condition for tree label assignment is reached")
        return df

    for sub_tree in tree.children:
        df = tree_label_assignment(sub_tree, df, label_col)

    return df


def top_down_analysis(df, clusters=2, bandwidth=0.5, label_col="label", met="kmean",
                      max_layers=5, group_thres=50, tree_sum=True):
    '''
    The function conducts the top down analysis for the a tree and get summarized 
    statistics for the nodes
    Inputs:
        df: dataframe
        clusters, bandwidth: parameters for the clustering methods
        label_col: column for the storage of labels
        met: clustering methods
        max_layer: max layer for the tree
        group_thres: the minimum number of indices for a single node
        tree_sum: if True, provide the summarized statistics for tree node;
                  if False, provide the labelled dataframe
    Return: summarized dataframe
    '''
    train_tree = unsupervised_tree(0, 0, df.index, df.mean(), 0)
    centroids_dict = {0:np.array(train_tree.centroid)}
    grow_tree(train_tree, df, clusters, bandwidth, label_col, met,
              max_layers, group_thres, centroids_dict)
    
    sum_df = pd.DataFrame({"full_data": df.mean()})

    if tree_sum:
        sum_df = pd.DataFrame({"full_data": df.mean()})
        return tree_summary(train_tree, df, sum_df)

    labelled_df = tree_label_assignment(train_tree, df, "label")
    return labelled_df


def bottom_up_analysis(df, label_col, tree_sum=True):
    '''
    The function conducts the bottom up analysis for the a tree and get summarized 
    statistics for the nodes
    Inputs:
        df: dataframe
        label_col: column for the storage of labels
        tree_sum: if True, provide the summarized statistics for tree node;
                  if False, provide the labelled dataframe
    Return: summarized dataframe
    '''
    sum_df = pd.DataFrame({"full_data":df.mean()})
    full_tree_node = cut_tree(df, label_col)

    for key, tree_node in full_tree_node.items():
        if tree_sum:
            return tree_summary(tree_node, df, sum_df)

        return tree_node




