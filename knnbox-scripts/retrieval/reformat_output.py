"""
Read the binary output from KNN-retrieval process and put it into the format:
- src.txt
- mt.txt
- {name}_qe_score.txt

Different way to get sentence-level QE scores:
- Log prob
- KNN distance
- Sentence similarity
- Number of different knn-proposals
- Model prediction equals retrieved knn-tokens

Also can try:
- Different datastore (reduced train set, non train set)
- Different layer

"""
import argparse
import os
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['dummy.en', 'wmt22general.en', 'ted_tst14.en'])
    parser.add_argument('--mt_model', type=str, choices=['deltalm_base_ft_ted', 'ted_new'])
    parser.add_argument('--emb_layer', type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=6)
    parser.add_argument('--datastore', type=str, choices=['ted_new'])
    parser.add_argument('--output_root_dir', type=str)

    args = parser.parse_args()
    print(args)

    bin_out_path = f"data/{args.mt_model}/custom/{args.datastore}_{args.emb_layer}/{args.dataset}.bin"
    src_file = f"files/{args.dataset}"

    with open(src_file, 'r') as f:
        src_sents = f.readlines()
        src_sents = [src_sent.strip() for src_sent in src_sents]

    with open(bin_out_path, 'rb') as f:
        bin_out = pickle.load(f)
    nr_neighbors_max = len(list(bin_out.values())[0].tokens[0].distances)

    mt = []

    # All scores are modified so that the higher the score, the better the MT quality
    # Sentence-level score is the average of the token-level score
    qe_score_dict = {
        'prob': []
    }
    nr_neighbors_list = [1] + list(range(2, nr_neighbors_max + 1, 2))
    for nr_neighbors in nr_neighbors_list:
        qe_score_dict[f'knn_distance_inv_k{nr_neighbors}'] = []
        qe_score_dict[f'sent_similarity_k{nr_neighbors}'] = []
        qe_score_dict[f'nr_diff_knn_inv_k{nr_neighbors}'] = []
        qe_score_dict[f'nr_out_equal_knn_k{nr_neighbors}'] = []

    for src_sent in src_sents:
        bin_out_sent = bin_out[src_sent] if src_sent in bin_out else None
        if bin_out_sent is None:
            mt.append("")
            for k,v in qe_score_dict.items():
                v.append(np.nan)
        else:
            mt.append(bin_out_sent.tgt_str)
            # Average of tokens' output probs
            qe_score_dict['prob'].append(
                np.array([x.chosen_token_prob for x in bin_out_sent.tokens]).mean()
            )

            for nr_neighbors in nr_neighbors_list:
                # Average of tokens' KNN distance (average of average)
                # Change sign so that the higher score means better quality
                qe_score_dict[f'knn_distance_inv_k{nr_neighbors}'].append(
                    - np.concatenate([
                        k_neareast_filter(x.distances, nr_neighbors, x.distances) for x in bin_out_sent.tokens
                    ]).mean()
                )

                # Average of tokens' train sentences similarity (also average of average)
                qe_score_dict[f'sent_similarity_k{nr_neighbors}'].append(
                    np.concatenate([
                        k_neareast_filter(x.cos_sims, nr_neighbors, x.distances) for x in bin_out_sent.tokens
                    ]).mean()
                )

                # Average of tokens' count different retrieved KNN
                # Change sign so that the higher score means better quality
                qe_score_dict[f'nr_diff_knn_inv_k{nr_neighbors}'].append(
                    - np.array([
                        len(set(k_neareast_filter(x.rec_token_id.cpu(), nr_neighbors, x.distances).tolist()))
                        for x in bin_out_sent.tokens
                    ]).mean()
                )

                # Average of count KNN tokens equal out tokens
                qe_score_dict[f'nr_out_equal_knn_k{nr_neighbors}'].append(
                    np.array([
                        np.equal(k_neareast_filter(x.rec_token_id.cpu(), nr_neighbors, x.distances), x.chosen_token_id.cpu()).sum()
                        for x in bin_out_sent.tokens
                    ]).mean()
                )

    # Replace the NaN values with the lowest score
    for k, v in qe_score_dict.items():
        qe_score_dict[k] = np.nan_to_num(v, nan=np.nanmin(v))

    # Output everything
    output_dir = f"{args.output_root_dir}/{args.mt_model}/{args.dataset}/{args.datastore}_{args.emb_layer}"
    os.makedirs(output_dir, exist_ok=True)
    write_text_file(src_sents, f"{output_dir}/src.txt")
    write_text_file(mt, f"{output_dir}/mt.txt")
    for k, v in qe_score_dict.items():
        write_text_file(v, f"{output_dir}/qeScore_{k}.txt")


def k_neareast_filter(l, k_small, distances):
    """
    :param: l: a list of values corresponding to neighbors in the `distances` list
    :param: k_small: number of neareast neighbors to filter
    :param: distances: distance of the k neighbors
    """
    l = np.array(l)
    distances = np.array(distances)
    filtered_indices = top_k_smallest_indices(distances, k_small)
    return l[filtered_indices]


def top_k_smallest_indices(arr, k):
    if k == len(arr):
        return np.argsort(arr)
    elif k < len(arr):
        # Use argpartition to get indices of top k smallest values
        indices = np.argpartition(arr, k)[:k]
        # Sort the indices based on the corresponding values
        sorted_indices = indices[np.argsort(arr[indices])]
        return sorted_indices
    else:
        raise RuntimeError(f"k={k} > len(arr)={len(arr)}")


def write_text_file(lines, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")


if __name__ == '__main__':
    main()
