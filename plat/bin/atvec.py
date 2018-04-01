import argparse
import sys
import json
import numpy as np
from sklearn import metrics
from sklearn import svm
import os
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# Force matplotlib to not use any Xwindows backend.
# from plat.fuel_helper import get_dataset_iterator
from plat.utils import json_list_to_array, get_json_vectors, offset_from_string

def filter_attributes(attribs, which_attribs):
    if which_attribs is None:
        return attribs
    attrib_range = list(map(int, which_attribs.split(",")))
    new_attribs = attribs[:,:,attrib_range]
    print("attribs filtered from {} to {}".format(attribs.shape, new_attribs.shape))
    return new_attribs

def get_averages(attribs, encoded):
    num_items, _, num_attribs = attribs.shape
    print("Splitting {} items across {} attributes ({}, {})".format(num_items, num_attribs, attribs.shape, encoded.shape))

    with_attr = [[] for x in xrange(num_attribs)]
    without_attr = [[] for x in xrange(num_attribs)]
    for i in tqdm(range(num_items)):
        for m in range(num_attribs):
            if attribs[i][0][m] == 1:
                with_attr[m].append(encoded[i])
            else:
                without_attr[m].append(encoded[i])

    print("With: {}".format(map(len, with_attr)))
    print("Without: {}".format(map(len, without_attr)))

    with_attr = map(np.array, with_attr)
    without_attr = map(np.array, without_attr)
    return with_attr, without_attr

def get_class_averages(attribs, encoded, num_classes):
    with_attr = [[] for x in xrange(num_classes)]
    without_attr = [[] for x in xrange(num_classes)]
    for i in range(len(encoded)):
        if i % 10000 == 0:
            print("iteration {}".format(i))
        for m in range(num_classes):
            if attribs[i][0] == m:
                with_attr[m].append(encoded[i])
            else:
                without_attr[m].append(encoded[i])

    print("With: {}".format(map(len, with_attr)))
    print("Without: {}".format(map(len, without_attr)))

    with_attr = map(np.array, with_attr)
    without_attr = map(np.array, without_attr)
    return with_attr, without_attr

def get_balanced_averages2(attribs, encoded, a1, a2):
    just_a1 = []
    just_a2 = []
    both = []
    neither = []
    for i in range(len(encoded)):
        if i % 10000 == 0:
            print("iteration {}".format(i))
        if attribs[i][0][a1] == 1 and attribs[i][0][a2] == 1:
            both.append(encoded[i])
        elif attribs[i][0][a1] == 1 and attribs[i][0][a2] == 0:
            just_a1.append(encoded[i])
        elif attribs[i][0][a1] == 0 and attribs[i][0][a2] == 1:
            just_a2.append(encoded[i])
        elif attribs[i][0][a1] == 0 and attribs[i][0][a2] == 0:
            neither.append(encoded[i])
        else:
            print("DANGER: ", attribs[i][0][a1], attribs[i][0][a2])

    len_both = len(both)
    len_just_a1 = len(just_a1)
    len_just_a2 = len(just_a2)
    len_neither = len(neither)
    topnum = max(len_both, len_just_a1, len_just_a2, len_neither)

    print("max={}, both={}, a1={}, a2={}, neither={}".format(
        topnum, len_both, len_just_a1, len_just_a2, len_neither))

    just_a1_bal = []
    just_a2_bal = []
    both_bal = []
    neither_bal = []

    for i in range(topnum):
        both_bal.append(both[i%len_both])
        just_a1_bal.append(just_a1[i%len_just_a1])
        just_a2_bal.append(just_a2[i%len_just_a2])
        neither_bal.append(neither[i%len_neither])

    with_attr = [ (just_a1_bal + both_bal), (just_a2_bal + both_bal)  ]
    without_attr = [ (just_a2_bal + neither_bal), (just_a1_bal + neither_bal) ]

    print("With: {}".format(map(len, with_attr)))
    print("Without: {}".format(map(len, without_attr)))

    with_attr = map(np.array, with_attr)
    without_attr = map(np.array, without_attr)
    return with_attr, without_attr

# recursive function to initialize nested array
def nested_binary_array_init(level, leaf_value):
    if level == 0:
        if leaf_value == None:
            return []
        else:
            return leaf_value
    else:
        return [
            nested_binary_array_init(level-1, leaf_value),
            nested_binary_array_init(level-1, leaf_value)
        ]

# recursive function to assign all lengths and return largest
def assign_len_get_max(matrix, lengths):
    if lengths[0] == 0:
        # leaf
        lengths[0] = len(matrix[0])
        lengths[1] = len(matrix[1])
        if lengths[0] > lengths[1]:
            return lengths[0]
        else:
            return lengths[1]
    else:
        # recursive
        len1 = assign_len_get_max(matrix[0], lengths[0])
        len2 = assign_len_get_max(matrix[1], lengths[1])
        if len1 > len2:
            return len1
        else:
            return len2

def replicate_balance_matrix(matrix, lengths, max_len):
    if isinstance(lengths[0], list):
        # recurse
        m1 = replicate_balance_matrix(matrix[0], lengths[0], max_len)
        m2 = replicate_balance_matrix(matrix[1], lengths[1], max_len)
        return [m1, m2]
    else:
        # compute leaves
        m1 = []
        m2 = []
        for z in range(max_len):
            m1.append(matrix[0][z%lengths[0]])
            m2.append(matrix[1][z%lengths[1]])
        return [m1, m2]

def collect_samples(branch, matrix, depth_decide, cur_depth):
    if isinstance(matrix[0][0], list):
        if cur_depth == depth_decide:
            return collect_samples(branch, matrix[branch], depth_decide, cur_depth+1);
        else:
            s1 = collect_samples(branch, matrix[0], depth_decide, cur_depth+1);
            s2 = collect_samples(branch, matrix[1], depth_decide, cur_depth+1);
            return s1 + s2
    else:
        if cur_depth == depth_decide:
            # slice syntax -> return a copy
            return matrix[branch][:]
        else:
            return matrix[0] + matrix[1]

def get_balanced_averages(attribs, encoded, indexes):
    num_ix = len(indexes)

    # initialize 2^n arrays.
    matrix = nested_binary_array_init(num_ix, None)
    lengths = nested_binary_array_init(num_ix, 0)

    # great, now partition the encoded list according to those attribs
    for i in range(len(encoded)):
        a = attribs[i][0]
        m = matrix
        for j in range(num_ix):
            m = m[a[indexes[j]]]
        m.append(encoded[i])

    max_len = assign_len_get_max(matrix, lengths)

    print("max={}, both={}, a1={}, a2={}, neither={}".format(
        max_len, lengths[1][1], lengths[1][0], lengths[0][1], lengths[0][0]))

    matrix_bal = replicate_balance_matrix(matrix, lengths, max_len)

    with_attr = [[] for i in range(num_ix)]
    without_attr = [[] for i in range(num_ix)]

    for i in range(num_ix):
        without_attr[i] = collect_samples(0, matrix_bal, i, 0)
        with_attr[i] = collect_samples(1, matrix_bal, i, 0)

    print("With: {}".format(map(len, with_attr)))
    print("Without: {}".format(map(len, without_attr)))

    with_attr = map(np.array, with_attr)
    without_attr = map(np.array, without_attr)

    return with_attr, without_attr

def averages_to_attribute_vectors(with_attr, without_attr):
    num_encoded_attributes = len(with_attr)
    _, latent_dim = with_attr[0].shape
    atvecs = np.zeros((num_encoded_attributes, latent_dim))
    for n in range(num_encoded_attributes):
        m1 = np.mean(with_attr[n],axis=0)
        m2 = np.mean(without_attr[n],axis=0)
        atvecs[n] = m1 - m2
    return atvecs

def averages_to_svm_attribute_vectors(with_attr, without_attr):
    h = .02  # step size in the mesh
    C = 1.0  # SVM regularization parameter
    num_encoded_attributes = len(with_attr)
    _, latent_dim = with_attr[0].shape
    atvecs = np.zeros((num_encoded_attributes, latent_dim))
    for n in range(num_encoded_attributes):
        X_arr = []
        y_arr = []
        for l in range(len(with_attr[n])):
            X_arr.append(with_attr[n][l])
            y_arr.append(True)
        for l in range(len(without_attr[n])):
            X_arr.append(without_attr[n][l])
            y_arr.append(False)
        X = np.array(X_arr)
        y = np.array(y_arr)
        # svc = svm.LinearSVC(C=C, class_weight="balanced").fit(X, y)
        print("Computing svm {} with {}, {}".format(n, X.shape, y.shape))
        svc = svm.LinearSVC(C=C).fit(X, y)
        print("svm computed")
        # get the separating hyperplane
        w = svc.coef_[0]
        # print(w)

        #FIXME: this is a scaling hack.
        m1 = np.mean(with_attr[n],axis=0)
        m2 = np.mean(without_attr[n],axis=0)
        mean_vector = m1 - m2
        mean_length = np.linalg.norm(mean_vector)
        svn_length = np.linalg.norm(w)

        atvecs[n] = (mean_length / svn_length)  * w
    return atvecs

    X_arr = []
    y_arr = []
    for n in range(len(with_attr)):
        X_arr.append()
    atvecs = np.zeros((num_encoded_attributes, latent_dim))
    for n in range(num_encoded_attributes):
        m1 = np.mean(with_attr[n],axis=0)
        m2 = np.mean(without_attr[n],axis=0)
        atvecs[n] = m1 - m2
    return atvecs

#TODO: switch to plat.save_json_vectors
def save_json_attribs(attribs, filename):
    with open(filename, 'w') as outfile:
        json.dump(attribs.tolist(), outfile)   

def compute_accuracy(y, score_list, threshold):
    l = len(score_list)
    y_pred_list = []
    for i in range(l):
        if score_list[i] < threshold:
            y_pred_list.append(0)
        else:
            y_pred_list.append(1)
    y_pred = np.array(y_pred_list)
    return metrics.accuracy_score(y, y_pred)

# binary search to find optimal threshold to maximize accuracy
def do_thresh(atvecs, encoded, attribs, outfile, isclass=False):
    if outfile is None:
        outfile = "thresh.json"
    l = min(len(encoded), len(attribs))
    print(attribs.shape)
    thresh_list = []
    for j in range(len(atvecs)):
        at_vec = atvecs[j]
        y_list = []
        score_list = []
        y_pred_list = []
        score_true_list = []
        score_false_list = []
        for i in range(l):
            score = np.dot(at_vec, encoded[i])
            score_list.append(score)
            if isclass:
                y_list.append(attribs[i][0])
                if attribs[i][0] == j:
                    score_true_list.append(score)
                else:
                    score_false_list.append(score)
            else:
                y_list.append(attribs[i][0][j])
                if attribs[i][0][j] == 1:
                    score_true_list.append(score)
                else:
                    score_false_list.append(score)
        y = np.array(y_list)
        sorted_scores = sorted(score_list)
        # here is the core loop
        min_index = 0
        min_index_accuracy = compute_accuracy(y, score_list, sorted_scores[min_index])
        max_index = l-1
        max_index_accuracy = compute_accuracy(y, score_list, sorted_scores[max_index])
        while(min_index + 1 < max_index):
            pivot_index = int(min_index + (max_index - min_index)/2)
            pivot_index_accuracy = compute_accuracy(y, score_list, sorted_scores[pivot_index])
            if (min_index_accuracy < pivot_index_accuracy):
                min_index = pivot_index
                min_index_accuracy = pivot_index_accuracy
            else:
                max_index = pivot_index
                max_index_accuracy = pivot_index_accuracy
        if min_index_accuracy > max_index_accuracy:
            best_index = min_index
        else:
            best_index = max_index
        # poor man's unit tests
        # best_index = best_index + 500
        # if best_index > l-1:
        #     best_index = l-1
        # if best_index < 0:
        #     best_indext = 0
        threshold = sorted_scores[best_index]
        best_accuracy = compute_accuracy(y, score_list, threshold)
        thresh_list.append(threshold)
        print("Best accuracy for attribute {:2d} is {:.4f} at {:6d} = {:6.3f}".format(j, best_accuracy, best_index, threshold))

    thresh_array = np.array([thresh_list])
    save_json_attribs(thresh_array, outfile)


def do_roc(chosen_vector, encoded, attribs, attribs_index, threshold, attrib_set, outfile, isclass):
    if outfile is None:
        outfile = "roc"
    if threshold is None:
        threshold = 0
    title = os.path.basename(outfile)

    score_negative = True
    score_positive = True
    if attrib_set == "false":
        score_positive = False
    elif attrib_set == "true":
        score_negative = False

    l = min(len(encoded), len(attribs))
    y_list = []
    score_list = []
    y_pred_list = []
    score_true_list = []
    score_false_list = []
    print(attribs.shape)
    for i in range(l):
        score = np.dot(chosen_vector, encoded[i])
        did_append = False
        if isclass:
            did_append = True
            score_list.append(score)
            if attribs[i][0] == attribs_index:
                y_list.append(1)
            else:
                y_list.append(0)
            if attribs[i][0] == attribs_index:
                score_true_list.append(score)
            else:
                score_false_list.append(score)
        else:
            label = attribs[i][0][attribs_index]
            if label == 1 and score_positive:
                did_append = True
                y_list.append(label)
                score_list.append(score)
                score_true_list.append(score)
            elif label != 1 and score_negative:
                did_append = True
                y_list.append(label)
                score_list.append(score)
                score_false_list.append(score)
        if did_append:
            if score < threshold:
                y_pred_list.append(0)
            else:
                y_pred_list.append(1)


    y = np.array(y_list)
    scores = np.array(score_list)
    y_pred = np.array(y_pred_list)
    scores_false = np.array(score_false_list)
    scores_true = np.array(score_true_list)
    # y = np.array([1, 1, 2, 2])
    # scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(y, y_pred)
    print("{} ({} rows) ROC AUC is {:.03f} and accuracy is {:.03f}".format(title, len(y), roc_auc, accuracy))
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC (acc={:.03f})'.format(title, accuracy))
    plt.legend(loc="lower right")
    plt.savefig('{}_{}_roc.png'.format(outfile, attrib_set), bbox_inches='tight')

    # # the histogram of the data
    # plt.figure()
    # n, bins, patches = plt.hist(scores, 50, facecolor='blue', alpha=0.75)
    # plt.xlabel('Attribute')
    # plt.ylabel('Probability')
    # plt.title('Histogram of {}'.format(title))
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.axvline(threshold, color='b', linestyle='dashed', linewidth=2)
    # plt.savefig('{}_hist_all.png'.format(outfile), bbox_inches='tight')

    # split histogram
    plt.figure()
    plt.hist(scores_true, 100, facecolor='green', alpha=0.5)
    plt.hist(scores_false, 100, facecolor='red', alpha=0.5)
    plt.xlabel('Attribute')
    plt.ylabel('Probability')
    plt.title('Histograms of {}'.format(title))
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.axvline(threshold, color='b', linestyle='dashed', linewidth=2)
    plt.savefig('{}_{}_hist.png'.format(outfile, attrib_set), bbox_inches='tight')

def get_attribs_from_file1(file):
    with open(file) as f:
        lines = f.readlines()
    a = [[[int(l.rstrip())]] for l in lines]
    a = np.array(a)
    print("Read attributes {} from {}".format(a.shape, file))
    return a

def get_attribs_from_file(filelist):
    files = filelist.split(",")
    lists = []
    for filename in files:
        with open(filename) as f:
            lines = f.readlines()
            lists.append(lines)
    num_items = len(lists[0])
    num_lists = len(lists)
    a = []
    for i in range(num_items):
        entry = []
        for j in range(num_lists):
            entry.append(int(lists[j][i].rstrip()))
        a.append([entry])
    a = np.array(a)
    print("Read attributes {} from {}".format(a.shape, filelist))
    # a = [[[int(l.rstrip())]] for l in lines]
    return np.array(a)

def atvec(parser, context, args):
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Source dataset (for labels).")
    parser.add_argument('--labels', dest='labels', default=None,
                        help="Text file with 0/1 labels.")
    parser.add_argument('--split', dest='split', default="train",
                        help="Which split to use from the dataset (train/nontrain/valid/test/any).")
    parser.add_argument("--num-attribs", dest='num_attribs', type=int, default=40,
                        help="Number of attributes (labes)")
    parser.add_argument("--which-attribs", type=str, default=None,
                        help="optional comma separated list of attributes to run")
    parser.add_argument("--num-classes", dest='num_classes', type=int, default=None,
                        help="For multiclass, number of classes (assumed 0 .. n-1)")
    parser.add_argument("--z-dim", dest='z_dim', type=int, default=100,
                        help="z dimension of vectors")
    parser.add_argument("--encoded-vectors", type=str, default=None,
                        help="Comma separated list of json arrays")
    parser.add_argument("--encoded-true", type=str, default=None,
                        help="Comma separated list of json arrays (true)")
    parser.add_argument("--encoded-false", type=str, default=None,
                        help="Comma separated list of json arrays (false)")
    parser.add_argument('--thresh', dest='thresh', default=False, action='store_true',
                        help="Compute thresholds for attribute vectors classifiers")
    parser.add_argument('--svm', dest='svm', default=False, action='store_true',
                        help="Use SVM for computing attribute vectors")
    parser.add_argument("--limit", dest='limit', type=int, default=None,
                        help="Limit number of inputs when computing atvecs")
    parser.add_argument('--roc', dest='roc', default=False, action='store_true',
                        help="ROC curve of selected attribute vectors")
    parser.add_argument("--attribute-vectors", dest='attribute_vectors', default=None,
                        help="use json file as source of attribute vectors")
    parser.add_argument("--attribute-thresholds", dest='attribute_thresholds', default=None,
                        help="use these non-zero values for binary classifier thresholds")
    parser.add_argument("--attribute-set", dest='attribute_set', default="all",
                        help="score ROC/accuracy against true/false/all")
    parser.add_argument('--attribute-indices', dest='attribute_indices', default=None, type=str,
                        help="indices to select specific attribute vectors")
    parser.add_argument("--balanced2", dest='balanced2', type=str, default=None,
                        help="Balanced two attributes and generate atvec. eg: 20,31")
    parser.add_argument("--balanced", dest='balanced', type=str, default=None,
                        help="Balance attributes and generate atvec. eg: 20,21,31")
    parser.add_argument("--avg-diff", dest='avg_diff', type=str, default=None,
                        help="Two lists of vectors to average and then diff")
    parser.add_argument("--svm-diff", dest='svm_diff', type=str, default=None,
                        help="Two lists of vectors to average and then svm diff")
    parser.add_argument('--outfile', dest='outfile', default=None,
                        help="Output json file for vectors.")
    args = parser.parse_args(args)

    if args.avg_diff:
        vecs1, vecs2 = args.avg_diff.split(",")
        encoded1 = json_list_to_array(vecs1)
        encoded2 = json_list_to_array(vecs2)
        print("Taking the difference between {} and {} vectors".format(len(encoded1), len(encoded2)))
        m1 = np.mean(encoded1,axis=0)
        m2 = np.mean(encoded2,axis=0)
        atvec = m2 - m1
        z_dim, = atvec.shape
        atvecs = atvec.reshape(1,z_dim)
        print("Computed diff shape: {}".format(atvecs.shape))
        if args.outfile is not None:
            save_json_attribs(atvecs, args.outfile)
        sys.exit(0)

    if args.svm_diff:
        vecs1, vecs2 = args.svm_diff.split(",")
        encoded1 = json_list_to_array(vecs1)
        encoded2 = json_list_to_array(vecs2)
        print("Taking the svm difference between {} and {} vectors".format(len(encoded1), len(encoded2)))
        h = .02  # step size in the mesh
        C = 1.0  # SVM regularization parameter
        X_arr = []
        y_arr = []
        for l in range(len(encoded1)):
            X_arr.append(encoded1[l])
            y_arr.append(False)
        for l in range(len(encoded2)):
            X_arr.append(encoded2[l])
            y_arr.append(True)
        X = np.array(X_arr)
        y = np.array(y_arr)
        # svc = svm.LinearSVC(C=C, class_weight="balanced").fit(X, y)
        svc = svm.LinearSVC(C=C).fit(X, y)
        # get the separating hyperplane
        w = svc.coef_[0]

        #FIXME: this is a scaling hack.
        m1 = np.mean(encoded1,axis=0)
        m2 = np.mean(encoded2,axis=0)
        mean_vector = m1 - m2
        mean_length = np.linalg.norm(mean_vector)
        svn_length = np.linalg.norm(w)

        atvec = (mean_length / svn_length)  * w
        z_dim, = atvec.shape
        atvecs = atvec.reshape(1,z_dim)
        print("Computed svm diff shape: {}".format(atvecs.shape))
        if args.outfile is not None:
            save_json_attribs(atvecs, args.outfile)
        sys.exit(0)

    print("reading encoded vectors...")
    attribs = None
    if args.encoded_vectors is not None:
        if args.encoded_vectors.endswith("json"):
            encoded = json_list_to_array(args.encoded_vectors)
            print("Read json array: {}".format(encoded.shape))
        else:
            encoded = np.load(args.encoded_vectors)['arr_0']
            print("Read numpy array: {}".format(encoded.shape))
    else:
        if args.encoded_true.endswith("json"):
            encoded_true = json_list_to_array(args.encoded_true)
            print("Read true json array: {}".format(encoded_true.shape))
        else:
            encoded_true = np.load(args.encoded_true)['arr_0']
            print("Read true numpy array: {}".format(encoded_true.shape))
        if args.encoded_false.endswith("json"):
            encoded_false = json_list_to_array(args.encoded_false)
            print("Read false json array: {}".format(encoded_false.shape))
        else:
            encoded_false = np.load(args.encoded_false)['arr_0']
            print("Read false numpy array: {}".format(encoded_false.shape))
        encoded = np.concatenate((encoded_true, encoded_false), axis=0)
        num_true = len(encoded_true)
        num_false = len(encoded_false)
        true_values = np.ones(shape=[num_true,1,1], dtype=np.int)
        false_values = np.zeros(shape=[num_false,1,1], dtype=np.int)
        attribs = np.concatenate((true_values, false_values), axis=0)

    if args.limit is not None:
        encoded = encoded[:args.limit]
    num_rows, z_dim = encoded.shape
    if attribs is None:
        print("reading attributes...")
        if args.dataset:
            attribs = np.array(list(get_dataset_iterator(args.dataset, args.split, include_features=False, include_targets=True)))
            print("Read attributes from dataset: {}".format(attribs.shape))
        else:
            print("Read attributes from file: {}".format(args.labels))
            attribs = get_attribs_from_file(args.labels)

    if args.which_attribs is not None:
        attribs = filter_attributes(attribs, args.which_attribs)
    print("encoded vectors: {}, attributes: {} ".format(encoded.shape, attribs.shape))

    if args.roc:
        atvecs = get_json_vectors(args.attribute_vectors)
        dim = len(atvecs[0])
        chosen_vector = offset_from_string(args.attribute_indices, atvecs, dim)
        if args.attribute_thresholds is not None:
            atvec_thresholds = get_json_vectors(args.attribute_thresholds)
            threshold = atvec_thresholds[0][int(args.attribute_indices)]
        else:
            threshold = None
        do_roc(chosen_vector, encoded, attribs, int(args.attribute_indices), threshold, args.attribute_set, args.outfile, isclass=(args.num_classes is not None))
        sys.exit(0)

    if args.thresh:
        atvecs = get_json_vectors(args.attribute_vectors)
        do_thresh(atvecs, encoded, attribs, args.outfile, isclass=(args.num_classes is not None))
        sys.exit(0)

    if(args.balanced2):
        indexes = map(int, args.balanced2.split(","))
        with_attr, without_attr = get_balanced_averages2(attribs, encoded, indexes[0], indexes[1]);
        num_attribs = 2
    elif(args.balanced):
        indexes = map(int, args.balanced.split(","))
        with_attr, without_attr = get_balanced_averages(attribs, encoded, indexes);
        num_attribs = len(indexes)
    elif args.num_classes is not None:
        with_attr, without_attr = get_class_averages(attribs, encoded, args.num_classes);
        num_attribs = args.num_classes
    elif args.num_attribs is not None:
        with_attr, without_attr = get_averages(attribs, encoded);
        num_attribs = args.num_attribs
    else:
        print("I think we need either num_classes or num_attribs or something")
        sys.exit(0);

    if args.svm:
        atvects = averages_to_svm_attribute_vectors(with_attr, without_attr)
    else:
        atvects = averages_to_attribute_vectors(with_attr, without_attr)
    print("Computed atvecs shape: {}".format(atvects.shape))

    if args.outfile is not None:
        save_json_attribs(atvects, args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot model samples")
    atvec(parser, None, sys.argv[1:])