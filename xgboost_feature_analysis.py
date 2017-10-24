'''
xgboost_analysis

author: Jair Taylor

A set of tools to help analyze XGBoost models by 
looking at the structure of the underlying trees.
In particular, it can be used to get the F-score 
for certain variables when other variables are fixed.  

'''

from copy import deepcopy

def get_forest(model, ntrees = None):
    '''
    Given an XGBoost model, output a list of trees.
    Each tree is a dictionary, with each element a node of the tree
    that with the following possible keys:
        'condition': a string describing the condition, e.g., 'x<1.004'
        'parent': the parent of the node, if the node is not the root
        'yes': the child corresponding to the condition being True
        'no': the child corresponding to the condition being False
        'missing': the child corresponding to the relevant variable missing
                    (seemingly always the same as 'yes')
        'leaf': the value of the node if the node is a leaf.
    '''

    if ntrees is None:
        ntrees = model.n_estimators
    
    booster = model.booster()
    forest = []

    for i in range(ntrees):
        tree = {}
        lines = booster.get_dump()[i].split('\n')
        for i in range(len(lines)):
            
            s = lines[i]
            s = s.replace('\t', '')

            node = {}

            if 'leaf' in s:
                node['leaf'] = float( s.split('=')[-1]  )
            elif '[' in s and ']' in s:
                split_node = s.replace('[', '***').replace(']', '***').split('***')
                node['condition'] = split_node[1]
                node.update( str_to_dict(split_node[2].strip()) )
                
            if ':' in s:
                index_of_node = int(s.split(':')[0])

            else:
                continue
                
            for key in node.keys():
                if key in ['yes', 'no', 'missing', 'index']:
                    node[key] = int(node[key])
            tree[index_of_node] = node
        forest.append(tree)
    return [identify_parents(tree) for tree in forest]

def str_to_dict(astr):
    output_dict = {}
    for s in astr.split(','):
        output_dict[s.split('=')[0]] = s.split('=')[1]
    return output_dict

def str_to_condition(astr):
    if '<' in astr:
        (var,value) = astr.split('<')
        return lambda x: x[var] < float(value)
    
    
def evaluate_tree(tree, x):
    node = tree[0]
    while 'leaf' not in node.keys():
        try:
            truth_condition = str_to_condition(node['condition'])(x)
        except:
            goto_index = node['missing']
        if truth_condition:
            
            goto_index  = node['yes']
        else:
            goto_index = node['no']
        node = tree.get(goto_index)
    return float(node['leaf'])
        
    
def list_conditions(tree,x):
    conditions = []
    node = tree[0]
    while 'leaf' not in node.keys():
        #print node
        try:
            truth_condition = str_to_condition(node['condition'])(x)
        except:
            goto_index = node['missing']
            
        condition_pair = [node['condition']]
        if truth_condition:
            goto_index  = node['yes']
            condition_pair.append(True)
        else:
            goto_index = node['no']
            condition_pair.append(False)
        conditions.append(condition_pair)
        node = tree[int(goto_index)]
    return conditions  

def evaluate_model(model, x):
    base_score = model.get_params()['base_score']
    forest = get_forest(model)
    output = base_score
    for tree in forest:
        s = evaluate_tree(tree,x)
        output += s
    return output


def evaluate_forest(forest, x, base_score):
    output = base_score
    for tree in forest:
        s = evaluate_tree(tree,x)
        output += s
    return output

def fetch_relevant_data(X, forest, x):
    for tree in forest:
        conditions = list_conditions(tree,x)
        for c in conditions:
            if c[1]:
                mask = [ str_to_condition(c[0])(X.loc[i]) for i in X.index]
            else:
                mask = [~str_to_condition(c[0])(X.loc[i]) for i in X.index]
            X = X[mask]
    return X
                
    
def get_feature_importance(forest, features = []):
    if not type(forest) is list: #if not a list (e.g. forest), assuming it is an xgb model
        forest = get_forest(forest)

    feature_importance_dict = {'Fscore_' + var:0 for var in features}
    for tree in forest:
        for i in tree.keys():
            node = tree[i]
            var = var_of_node(node)
            if not var is None:
                if 'Fscore_' + var in feature_importance_dict.keys():
                    feature_importance_dict['Fscore_' + var] += 1
                else: 
                    feature_importance_dict['Fscore_' + var] = 1
    return feature_importance_dict

def get_feature_importance_at_points(forest, points, features = []):
    if not type(forest) is list:  #if not a list (e.g. forest), assuming it is an xgb model
        forest = get_forest(forest)

    feature_importance_list = []

    for x in points:
        contracted_forest = contract_forest(forest,x)
        #features = [var for var in model.booster().feature_names if var not in x.keys()]
        feature_importance_dict = get_feature_importance(contracted_forest, features = features)
        feature_importance_dict.update(x)
        feature_importance_list.append(feature_importance_dict)

    return feature_importance_list

def var_of_node(node):
    if 'condition' in node.keys():
        return node['condition'].split('<')[0]
    else:
        return None
    
def contract_tree(tree, x):
    '''
    Given a tree and a single data point x, which may not have all the features of tree,
    returns the tree contracted_tree so that
        - if x has its missing features filled in in any way, contracted_tree
          will evaluate the same as tree.
        - for any feature of x, contracted_tree will 
          not have any nodes that depend on this feature.
    
    We find contracted_tree results from deleting all nodes that have 
    the features of x, and replacing them with the appropriate child
    based on the values of x.
    '''
    contracted_tree = deepcopy(tree)
    for n in sorted(contracted_tree.keys()):
        if n in contracted_tree.keys():
            some_contracted = False
            node = contracted_tree[n]
            if var_of_node(node) in x.keys():
                parent = node.get('parent', None)
                try:
                    truth_condition = str_to_condition(node['condition'])( x  )
                    if truth_condition:
                        child = node['yes']
                    else:
                        child = node['no']
                except:
                    child = node['missing']

                siblings = [i for i in [node['yes'], node['no'], node['missing']] if i != child]

                for s in siblings:
                    for d in descendants(tree,s):
                        if d in contracted_tree.keys():
                            del contracted_tree[d]

                if parent is None:
                    del contracted_tree[child]['parent']
                    del contracted_tree[n]
                else:
                    for key in ['yes', 'no', 'missing']:
                        if contracted_tree[parent][key] == n:
                            contracted_tree[parent][key] = child #contracted_tree[child]['index']
                            contracted_tree[child]['parent'] = parent
                    del contracted_tree[n]

    #Relabel root as 0 if necessary
    root = min(contracted_tree.keys())
    if root != 0:
        for key in ['yes', 'no', 'missing']:
            if key in contracted_tree[root].keys():
                child = contracted_tree[root][key]
                contracted_tree[child]['parent'] = 0
        contracted_tree[0] = contracted_tree[root]
        del contracted_tree[root]
        

    return contracted_tree

def contract_forest(forest, x):
    return [contract_tree(tree,x) for tree in forest]

def descendants(tree, n):
    node = tree[n]
    
    if 'leaf' in node.keys():
        return [n]
    else:
        gens = [[n]]
        while True:
            gen = gens[-1]
            nextgen = []
            for i in gen:
                node = tree[i]
                
                nextgen += list(set([node[key] for key in ['yes', 'no', 'missing'] if key in node.keys()]))
            if len(nextgen) == 0:
                break
            else:
                gens.append(nextgen)
        return sum(gens, [])
    
def identify_parents(tree):
    gens = [[0]]
    while True:
        gen = gens[-1]
        nextgen = []
        for i in gen:
            node = tree[i]
            for j in list(set([node[key] for key in ['yes', 'no', 'missing'] if key in node.keys()])):
                nextgen.append(j)
                tree[j]['parent'] = i
        if len(nextgen) == 0:
            break
        else:
            gens.append(nextgen)
    return tree
    