import argparse
import sys
import json
import numpy as np
from plat.fuel_helper import get_dataset_iterator
from plat.utils import json_list_to_array

def get_averages(attribs, encoded, num_encoded_attributes):
    with_attr = [[] for x in xrange(num_encoded_attributes)]
    without_attr = [[] for x in xrange(num_encoded_attributes)]
    for i in range(len(encoded)):
        if i % 10000 == 0:
            print("iteration {}".format(i))
        for m in range(num_encoded_attributes):
            if attribs[i][0][m] == 1:
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

def averages_to_attribute_vectors(with_attr, without_attr, num_encoded_attributes, latent_dim):
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

def atvec(parser, context, args):
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Source dataset (for labels).")
    parser.add_argument('--split', dest='split', default="train",
                        help="Which split to use from the dataset (train/nontrain/valid/test/any).")
    parser.add_argument("--num-attribs", dest='num_attribs', type=int, default=40,
                        help="Number of attributes (labes)")
    parser.add_argument("--z-dim", dest='z_dim', type=int, default=100,
                        help="z dimension of vectors")
    parser.add_argument("--encoded-vectors", type=str, default=None,
                        help="Comma separated list of json arrays")
    parser.add_argument("--balanced2", dest='balanced2', type=str, default=None,
                        help="Balanced two attributes and generate atvec. eg: 20,31")
    parser.add_argument("--balanced", dest='balanced', type=str, default=None,
                        help="Balance attributes and generate atvec. eg: 20,21,31")
    parser.add_argument("--avg-diff", dest='avg_diff', type=str, default=None,
                        help="Two lists of vectors to average and then diff")
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

    encoded = json_list_to_array(args.encoded_vectors)
    num_rows, z_dim = encoded.shape
    attribs = np.array(list(get_dataset_iterator(args.dataset, args.split, include_features=False, include_targets=True)))
    print("encoded vectors: {}, attributes: {} ".format(encoded.shape, attribs.shape))

    if(args.balanced2):
        indexes = map(int, args.balanced2.split(","))
        with_attr, without_attr = get_balanced_averages2(attribs, encoded, indexes[0], indexes[1]);
        num_attribs = 2
    elif(args.balanced):
        indexes = map(int, args.balanced.split(","))
        with_attr, without_attr = get_balanced_averages(attribs, encoded, indexes);
        num_attribs = len(indexes)
    else:
        with_attr, without_attr = get_averages(attribs, encoded, args.num_attribs);
        num_attribs = args.num_attribs

    atvects = averages_to_attribute_vectors(with_attr, without_attr, num_attribs, z_dim)
    print("Computed atvecs shape: {}".format(atvects.shape))

    if args.outfile is not None:
        save_json_attribs(atvects, args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot model samples")
    atvec(parser, None, sys.argv[1:])