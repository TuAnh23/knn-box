r"""
This file is copied from fairseq_cli/generate.py
knnbox make slight change about parsing args so that is
can parse the arch specified on the cli instead of directly
using the arch inside checkpoint.
"""

import ast
import logging
import math
import os
import os.path
import sys
import time
import random
from itertools import chain
from collections import namedtuple
from dotenv import load_dotenv
import pickle
from typing import List, Literal
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std
from scipy.stats import spearmanr
from scipy.stats.stats import pearsonr
from sklearn.metrics import matthews_corrcoef
from getkey import getkey
from tqdm import tqdm
import pandas as pd
import sentencepiece as sp
import torch
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from comet import download_model, load_from_checkpoint

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq.sequence_scorer import SequenceScorer

from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever

from token_stat import SentenceStat, Token, Error, need_update, get_data_store, get_knn_layer, get_mt_model_name, WmtError, WmtErrorSeverity, WmtErrorType

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
stats = []

def make_batches(lines, args, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if args.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    if args.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )

def main(args, override_args):
    assert args.path is not None, "--path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(
            args.results_path, "generate-{}.txt".format(args.gen_subset)
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(args, override_args, h)
    else:
        return _main(args, override_args, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(args, override_args, output_file):
    load_dotenv()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(args)

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 12000
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    task.load_dataset("test")
    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    ## knnbox related code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    overrides = ast.literal_eval(args.model_overrides)
    # if override_args is not None:
    #     overrides = vars(override_args)
    #     overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    # else:
    #     overrides = None
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )
    #models[0].decoder.set_beam_size(10)
    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(args)
    bpe = task.build_bpe(args)
    if args.lm_path is not None:
        overrides["data"] = args.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [args.lm_path],
                arg_overrides=overrides,
                task=None,
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({args.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": args.lm_weight}
    generator = task.build_generator(
        models, args, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_pos = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x
    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x
    # TODO remove if necessary
    use_sp = True
    spm_model = sp.SentencePieceProcessor(model_file="bpe.model")
    def encode_str(sentence, dict):
        if use_sp:
            sentencepiece_line = " ".join(spm_model.encode(sentence, out_type=str))
            return dict.encode_line(sentencepiece_line, add_if_not_exist=False).type(torch.int64).cuda()
        return ""
    def encode_pair(src, tgt):
        src = encode_str(src, src_dict)
        tgt = encode_str(tgt, tgt_dict)
        return src, tgt
    def get_lengths(enc):
        return [enc.shape[0]]
    def generate_hypos_from_tokens(src_tokens):
        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": get_lengths(src_tokens),
            },
        }
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=None,
            constraints=None,
        )
        return hypos;
    def generate_hypos_from_string(src_str):
        return generate_hypos_from_tokens(encode_str(src_str, src_dict).unsqueeze(0))
    def enc_prefrix(enc):
        shape = enc.shape
        for i in range(0, shape[0]):
            for j in reversed(range(1, shape[1])):
                enc[i][j] = enc[i][j-1]
            enc[i][0] = tgt_dict.eos()
        #pre = torch.LongTensor([]).cuda()
        #ret = torch.cat((pre, enc[0:enc.shape[0]-1]), 0).unsqueeze(0)
        return enc.cuda()
    def enc_to_str(enc, dict = tgt_dict):
        return decode_fn(dict.string(enc, args.remove_bpe)).replace(" @-@ ", "-")
    def translate(src_str = None):
        if src_str == None:
            src_str = input()
        hypos = generate_hypos_from_string(src_str)
        size = len(hypos[0])
        for i in range(0, size):
            print(enc_to_str(hypos[0][i]["tokens"], tgt_dict))

    def retrieve_by_id(id, ds):
        ret = ds[id]
        src = ret["source"]
        src = src
        tgt = ret["target"]
        return enc_to_str(src, src_dict), enc_to_str(tgt)

    model = models[0]
    ds = task.dataset(args.gen_subset)
    test_ds = task.dataset("test")
    k = 100
    # TODO better solution for datastore
    knn_store_layer = get_knn_layer()
    datastore = None
    name = get_data_store()
    mt_model_name = get_mt_model_name()
    datastore_path = f"../../datastore/vanilla-visual/{mt_model_name}/{name}_{knn_store_layer}"
    print(f"Loading datastore from: {datastore_path}")
    datastore = Datastore.load(datastore_path, load_list=["keys", "vals", "sentence_ids", "token_positions"])
    datastore.load_faiss_index("keys")
    retriever = Retriever(datastore=datastore, k=k)

    def hidden_state(src_encoding, tgt_encoding):
        lenghts = get_lengths(src_encoding)
        forward_res = model.forward(src_encoding, lenghts, tgt_encoding, features_only=True)
        return forward_res[1]["inner_states"][knn_store_layer].permute(1,0,2)
        #return forward_res[0]
        # for i, d in enumerate(data):
        #     forwa rd_res = model.forward(enc_src, lengths, d, features_only=True)
        #     #print(model.forward(enc_src, lengths, d, features_only=True)[0].shape)
        #     print(forward_res[0][0][i])
        #     res[i] = forward_res[0][0][i]
        #     #print("Added decoding")
        # print("Finished")
    def retrieve_similar(embeddings):
        return retriever.retrieve(embeddings, return_list=["vals", "distances", "sentence_ids", "token_positions"])
    def get_info(knn_res, i, max_prob_token, chosen_token_prob):
        src_sentences = []
        tgt_sentences = []
        src_enc = []
        tgt_enc = []
        next_words    = []
        # TODO replace 0 with code that allows for simultaneous retrieval
        for w in map(lambda w: tgt_dict[int(w.cuda().item())] , knn_res["vals"].cuda()[0][i]):
            next_words.append(w)
        for j, id in enumerate(knn_res["sentence_ids"][0][i]):
            #breakpoint()
            src_enc.append(ds[id]["source"])
            src_sentences.append(enc_to_str(ds[id]["source"]))
            # TODO remove 0
            token_pos = knn_res["token_positions"][0][i][j]
            tgt_enc.append(enc_to_str(ds[id]["target"][0:token_pos]))
            tgt_sentences.append(enc_to_str(ds[id]["target"][0:token_pos]))
        return {
            "source_encoding": src_enc,
            "target_encoding": tgt_enc,
            "source_sentences": src_sentences,
            "target_sentences": tgt_sentences,
            "next_words": next_words,
            # TODO remove 0
            "next_tokens": knn_res["vals"][0][i],
            # TODO remove 0
            "distances": list(map(lambda x: x.item(), knn_res["distances"][0][i])),
            "max_prob_token": max_prob_token,
            "chosen_token_prob": chosen_token_prob,
        }
    def whole_sentence_info_from_enc(src_sentence_enc, tgt_sentence_enc):
        embeddings = hidden_state(src_sentence_enc, enc_prefrix(tgt_sentence_enc))
        output = model.output_layer(embeddings[0])
        probs = model.get_normalized_probs((output, None), False)
        max_prob_tokens = []
        chosen_likelihood = []
        for i, likelihoods in enumerate(probs):
            max_prob_tokens.append(torch.argmax(likelihoods).item())
            chosen_likelihood.append(probs[i][tgt_sentence_enc[0][(i+1)% tgt_sentence_enc.shape[1]]].item())
        # for i in range(0, tgt_sentence_enc.shape[1]):
        #     # Strange indexing because of 2-token in tgt_sentence_enc
        #     chosen_likelihood.append(probs[(i+1)% tgt_sentence_enc.shape[0]][tgt_sentence_enc[0][i]].item())
        knn_res = retrieve_similar(embeddings)
        token_info = []
        src_dict[0]
        enc_to_str(max_prob_tokens, tgt_dict)
        for i in range(0, embeddings.shape[1]):
            token_info.append(get_info(knn_res, i, max_prob_tokens[i], chosen_likelihood[i]))
        return token_info
    def whole_sentence_info(src_sentence, tgt_sentence):
        src, tgt = encode_pair(src_sentence, tgt_sentence)
        src = src.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        retrieval =  whole_sentence_info_from_enc(src, tgt)
        return {
            "source": src_sentence,
            "target": tgt_sentence,
            "knn_retrieval": retrieval,
        }

    def analyze_translation(src):
        src_encoding = encode_str(src, src_dict)
        hypos = generate_hypos_from_tokens(src_encoding.unsqueeze(0))
        hypo_encoding = hypos[0][0]["tokens"]
        retrieval =  whole_sentence_info_from_enc(src_encoding.unsqueeze(0), hypo_encoding.unsqueeze(0))
        return {
            "src_enc": src_encoding,
            "tgt_enc": hypo_encoding,
            "source": src,
            "target": enc_to_str(hypo_encoding),
            "knn_retrieval": retrieval,
        }
    def print_single(step):
        source = step["source_sentences"]
        target = step["target_sentences"]
        words = step["next_words"]
        distances = step["distances"]
        for i in range(0, len(source)):
            print("\t\t+" + source[i])
            print("\t\t-" + target[i] + " _" + words[i] + "_(" + str(distances[i]) + ")")
    def print_info(whole_info):
        info = whole_info["knn_retrieval"]
        for i, step in enumerate(info):
            print("\tToken no. " + str(i) + " :" + str(tgt_dict[whole_info["tgt_enc"][i]]))
            print_single(step)
        print(whole_info["source"])
        print(whole_info["target"])

    def show_retrieve(sentence):
        print_info(analyze_translation(sentence))
    def get_grading(display = True, size = k):
        message = "Evaluate each proposal, 0=incorrect, 1=correct, e.g. 01101111\n"
        if display:
            message = ""
        grading = input(message)
        if len(grading) != size:
            print("Enter " + str(size) + " values")
            return get_grading(False, size)
        for c in grading:
            if not (c == '0' or c == '1'):
                print("Only enter 0 or 1")
                return get_grading(False)
        return grading
    def load_anotations(annotations_file_name):
        annotations = {}
        if os.path.exists(annotations_file_name):
            print(f"Found existing anotations at {annotations_file_name}")
            with open(annotations_file_name, "rb") as f:
                annotations = pickle.load(f)
        return annotations
    def save_anotations(annotations, anotations_file_name):
        if annotations != None:
            print(f"Saving annotation to {anotations_file_name}")
            with open(anotations_file_name, "wb") as f: # "wb" because we want to write in binary mode
                pickle.dump(annotations, f)
    def collect_annotation(res):
        size = res.shape[0]
        annotation = [Error.CORRECT]
        i = 0
        while i < size-1:
            before = res[0:i+2]
            after = res[i+2:size]
            print(enc_to_str(before), "___", enc_to_str(after) , end='\r')
            anno = getkey()
            if anno == "-":
                i -= 1
                annotation.pop()
                continue
            if not anno.isnumeric():
                continue
            anno = int(anno)
            if anno >= Error.count():
                continue
            i += 1
            annotation.append(Error.int_to_error(anno))
        print()
        return annotation

    # def anotate(info):
    #     tgt_enc = info["tgt_enc"]
    #     parts = map(lambda t: src_dict[t], tgt_enc)
    #     acc = ""
    #     steps = [acc := acc + x + "__" for x in parts]
    #     print(acc)
    #     correct_tokens = get_grading(False, tgt_enc.shape[0]-1)
    #     info["translation_correct"] = correct_tokens
    #     ret = info["knn_retrieval"]
    #     eval = []
    #     for i, r in enumerate(ret[:-1]):
    #         if (i == 0):
    #             print("(First token)")
    #         else:
    #             print(steps[i])
    #         print_single(r)
    #         grading = get_grading(False)
    #         eval.append(grading)
    #     info["knn_correct"] = eval
    # def analyze_anotation(info):
        # ret = info["knn_retrieval"]
        # knn_correct = info["knn_correct"]
        # #knn_correct = ['10111111', '10011010', '11111011', '11111111']
        # for i in range(0, len(knn_correct)):
        #     correctness = knn_correct[i]
        #     distances = ret[i]["distances"]
        #     correct_dist = []
        #     error_dist = []
        #     amount = len(correctness)
        #     average = 0
        #     correct_average = 0
        #     error_average = 0
        #     for j in range(0, amount):
        #         c = correctness[j]
        #         distance = distances[j]
        #         average += distance
        #         if c == '0':
        #             error_dist.append(distance)
        #         elif c == '1':
        #             correct_dist.append(distance)
        #     if len(correct_dist) > 0:
        #         correct_average = sum(correct_dist) / len(correct_dist)
        #     else:
        #         correct_average = None
        #     if len(error_dist) > 0:
        #         error_average = sum(error_dist) / len(error_dist)
        #     else:
        #         error_average = None
        #     average = sum(distances) / len(distances)
        #     info["knn_retrieval"][i]["correct_average"] = correct_average
        #     info["knn_retrieval"][i]["error_average"] = error_average
        #     info["knn_retrieval"][i]["total_average"] = average
    def print_anotation_statistic(info):
        ret = info["knn_retrieval"]
        knn_correct = info["knn_correct"]
        #knn_correct = ['10111111', '10011010', '11111011', '11111111']
        for i in range(0, len(knn_correct)):
            print("Token no " + str(i) + ", |" + tgt_dict[info["tgt_enc"][i]] + "|")
            print("\tKnn-data: " + knn_correct[i])
            print("\tAverage distance of knn values: " + str(ret[i]["total_average"]))
            print("\tAverage distance of correct knn values: " + str(ret[i]["correct_average"]))
            print("\tAverage distance of incorrect knn values: " + str(ret[i]["error_average"]))
    def knn_statistics_batch(length_batch, sentence_size):
        batch_size = 32
        current_length = len(length_batch)
        while True:
            actual_size = max(current_length, batch_size)
            if actual_size == 0:
                break
            tensor = torch.zeros(actual_size, sentence_size)
    def count_occurence(search, target):
        hits = 0
        for t in target:
            if search == t:
                hits += 1
        return hits
    def occurence_in_n(search, target, n):
        for i in range(0, n):
            if search == target[i]:
                return 1
        return 0
    def retrieval(g1, g2):
        count = 0
        for i in g1:
            if i in g2:
                count += 1
        return count

    import spacy
    import de_core_news_sm
    nlp = spacy.load("de_core_news_sm")
    nlp = de_core_news_sm.load()

    # For invididual tokens of sentence
    def token_stat_init(stat: SentenceStat, src, tgt, knn, annotation = None):
        src_length = src.shape[0]
        tgt_length = tgt.shape[0]
        stat.generated_token = tgt
        stat.tgt_count = tgt_length
        stat.src_str = enc_to_str(src)
        stat.tgt_str = enc_to_str(tgt)
        doc = nlp(stat.tgt_str)
        tags = [(w.text, w.pos_) for w in doc]
        for i in range(0, tgt.shape[0]):
            t = Token()
            # +1 is distorted as first token is 2, either change it (but also watch out for last token)
            # or use original encoding as tgt
            current_token = tgt[(i + 1)  % tgt_length]
            current_knn = knn[i]
            t.chosen_token_id = current_token
            t.chosen_token_prob = current_knn["chosen_token_prob"]
            next_tokens = current_knn["next_tokens"]
            knn_sources = current_knn["source_encoding"]
            knn_targets = current_knn["target_encoding"]
            t.rec_token_str = current_knn["target_encoding"]
            t.rec_token_id = current_knn["next_tokens"]
            t.max_prob_token = current_knn["max_prob_token"]
            t.distances = current_knn["distances"]
            #stat.max_prob_token.append(current_knn["max_prob_token"])
            #stat.distances.append(current_knn["distances"])
            correct_knn_predicitons = count_occurence(current_token, next_tokens)
            t.sum_1 = occurence_in_n(current_token, next_tokens, 1)
            t.sum_5 = occurence_in_n(current_token, next_tokens, 5)
            t.sum_k = occurence_in_n(current_token, next_tokens, k)
            #stat.sum1_values.append(occ1)
            #stat.sum5_values.append(occ5)
            #stat.sumk_values.append(occk)
            # stat.total_correct_predicitions += correct_knn_predicitons
            # stat.sum_at_least_one += 1 if correct_knn_predicitons > 0 else 0
            # stat.retrieved_token.append(next_tokens)
            t.total_correct_predicitions = correct_knn_predicitons
            #stat.precision.append([])
            #stat.recall.append([])
            recalls = []
            precisions = []
            # stat.retrieved.append([])
            for j in range(0, k):
                #stat.retrieved[i].append(enc_to_str(knn_sources[j]))
                t.retrieved_sentences.append(enc_to_str(knn_sources[j]))
                true_pos = retrieval(src, knn_sources[j].cuda())
                retrieved_count = knn_sources[j].shape[0]
                relevant_count = src_length
                #stat.precision[i].append((true_pos, retrieved_count))
                #stat.recall[i].append((true_pos, relevant_count))
                recalls.append(true_pos / relevant_count)
                precisions.append(true_pos / retrieved_count)
            #stat.recall.append(math.fsum(recalls) / len(recalls))
            #stat.precision.append(math.fsum(precisions) / len(precisions))
            t.different_count = len(set(t.rec_token_id.tolist()))
            t.recall = recalls
            t.precision = precisions
            t.avg_recall = math.fsum(recalls) / len(recalls)
            t.avg_precision = math.fsum(precisions) / len(precisions)
            stat.tokens.append(t)
        if annotation != None:
            stat.add_annotation(annotation)
        # src_dict
        # test = list(map(lambda x: x.chosen_token_id.item(), stat.tokens))
        # ref = list(map(lambda x: src_dict[x], test))
        stat.sentence_stat_init()
        stat.add_pos(tgt_dict)
    def filter_token(stat: SentenceStat, i, prediction_correct = None, retrieved_correct = None, min_dist = 0, max_dist = 10000000,
                     min_sim = 0, max_sim = 1):
        if prediction_correct is not None and prediction_correct == (stat.max_prob_token[i] != stat.generated_token[i+1]):
            return False
        look_at = 1
        if retrieved_correct is not None and retrieved_correct == (stat.generated_token[i+1] not in stat.retrieved_token[i][0:look_at]):
            return False
        if stat.distances[i][0] < min_dist or stat.distances[i][0] > max_dist:
            return False
        if not min_sim <= stat.token_avg_cos_sim[i] <= max_sim:
            return False
        return True # TODO
    def add_annotations(filename="data/standard_annotation", src="ted", overwrite=False):
        os.makedirs(filename, exist_ok=True)
        if src == "custom":
            custom_file_name = os.getenv('CUSTOM_FILE_NAME').lower()
            filename += "/" + custom_file_name + ".bin"
        model_path = download_model("Unbabel/wmt20-comet-qe-da")
        model = load_from_checkpoint(model_path)
        comet_data = []

        annotations = load_anotations(filename)
        update = False
        for src_str, value in annotations.items():
            #value.knn_store_layer = -1
            #continue
            if not need_update() and value.knn_store_layer == knn_store_layer:
                continue
            update = True
            print("Found wrong datastore layer:", src_str)
            res = analyze_translation(src_str)
            src_enc = res["src_enc"]
            tgt = res["tgt_enc"]
            knn = res["knn_retrieval"]
            annotation = list(map(lambda x: x.annotation, value.tokens))
            statistic = SentenceStat()
            token_stat_init(statistic, src_enc, tgt, knn, annotation)
            annotations[src_str] = statistic
            comet_data.append({"src": enc_to_str(src_enc),"mt": enc_to_str(tgt)})
        if need_update():
            res = model.predict(comet_data, batch_size=8, gpus=1)[0]
            c = 0
            for src_str, value in annotations.items():
                annotations[src_str].score = res[c]
                c += 1
        i = 0
        last = -1
        if src != "custom":
            last = int(input("Stop after ... sentences"))
        else:
            src_list = open("files/" + custom_file_name, "r").readlines()
            last = len(src_list)
        if src == "ted":
            print("Using ted data")
            #src_list = ds
            # sentences with less than 200 chars
            src_list = open("files/ted.txt", "r").readlines()
        elif src == "wmt/xinhua.txt":
            print("Using news data")
            src_list = open(src, "r").readlines()
        for _ in tqdm(range(len(src_list))):
                if len(annotations) == len(src_list):
                    break
                if i == last:
                    break
                if src != "custom":
                    print(last - i, "sentences remaining")
                    print("Annotation no. ", len(annotations))
                src_str = src_list[i]
                src_str = src_str.replace("\n", "")
                if src != "custom":
                    print(src_str)
                i += 1
                if src_str in annotations and not overwrite:
                    last += 1
                    continue
                if src_str == "q":
                    break
                res = analyze_translation(src_str)
                src_enc = res["src_enc"]
                tgt = res["tgt_enc"]
                knn = res["knn_retrieval"]
                if src != "custom":
                    annotation = collect_annotation(tgt)
                else:
                    annotation = [Error.CORRECT] * len(res['knn_retrieval'])
                statistic = SentenceStat()
                #breakpoint()
                token_stat_init(statistic, src_enc, tgt, knn, annotation)
                annotations[src_str] = statistic
        if src != "custom":
            delete = ""
            while last != 0:
                delete = input("")
                if delete == "q":
                    break
                if delete in annotations:
                    del annotations[delete]
        if update or last != 0:
            save_anotations(annotations, filename)
            print("Saved annotations")
        return list(annotations.values())
    def analyze(list):
        length = len(list)
        sum_1 = []
        sum_5 = []
        sum_k = []
        sum_at_least_one = []
        total_correct_predicitions = []
        avg_recalls = []
        avg_precisions = []
        for l in list:
            sum_1.append(l.sum_1 / l.tgt_count)
            sum_5.append(l.sum_5 / l.tgt_count)
            sum_k.append(l.sum_k / l.tgt_count)
            sum_at_least_one.append(l.sum_at_least_one / l.tgt_count)
            total_correct_predicitions.append(l.total_correct_predicitions / (l.tgt_count * k))
            avg_recalls.append(l.avg_recall)
            avg_precisions.append(l.avg_precision)
        sum_1 = math.fsum(sum_1) / len(sum_1)
        sum_5 = math.fsum(sum_5) / len(sum_5)
        sum_k = math.fsum(sum_k) / len(sum_k)
        sum_at_least_one = math.fsum(sum_at_least_one) / len(sum_at_least_one)
        total_correct_predicitions = math.fsum(total_correct_predicitions) / len(total_correct_predicitions)
        total_avg_recall = math.fsum(avg_recalls) / len(avg_recalls)
        total_avg_precision = math.fsum(avg_precisions) / len(avg_precisions)
        breakpoint()
        return None

    def save_plot(folder, name):
        file_name = folder + "/" + name + ".pdf"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name)
        print("\tCreated:", file_name)
        plt.close()

    def create_graphs(filter_by):
        statfile = "filter_stats.bin"
        if os.path.exists(statfile):
            print("Found stat file")
            with open(statfile, "rb") as f:
                filter_stats = pickle.load(f)
        else:
            return
        x = []
        r1 = []
        r5 = []
        rk = []
        pc = []
        for _, stat in filter_stats.items():
            x.append(stat["min_" + filter_by])
            r1.append(stat["avg_sum1"])
            r5.append(stat["avg_sum5"])
            rk.append(stat["avg_sumk"])
            pc.append(stat["avg_pred_corrrect"])
        plt.step(x, r1, "r", label="Retrieve in Top1")
        plt.step(x, r5, "g", label="Retrieve in Top5")
        plt.step(x, rk, "b", label="Retrieve in Topk")
        plt.step(x, pc, "k", label="Predicted by Model")
        plt.legend(loc="upper right")
        plt.ylabel("Percentage of correct tokens")
        plt.xlabel("Cos Similarity")
        #plt.xticks(np.arange(x[0], x[-1], step=x[1] - x[0]))
        save_plot(folder_name, "avg_sum")

    def create_sorted_graph(stats: SentenceStat, sort_by, folder_name, mode, show: Literal["no_label", "correct", "error"], step_size = 1600):
        sorted_list = None
        buckets = math.floor(len(stats.tokens) / step_size)
        if len(stats.tokens) < 100:
            print("Fewer than 100 tokens, no graph will be drawn")
            return
        if buckets < 1:
            return
        left_over = len(stats.tokens) % step_size
        inc = math.ceil(left_over / buckets)
        step_size += inc
        print("Sorting", len(stats.tokens), "tokens")
        if sort_by == "cos":
            sorted_list = sorted(stats.tokens, key=lambda x: x.avg_cos_sim)
        elif sort_by == "distance":
            sorted_list = sorted(stats.tokens, key=lambda x: x.distances[0])
        elif sort_by == "different_count":
            sorted_list = stats.tokens
            step_size = 1
        sentences = []
        average_correct = stats.annotation_correct
        current = SentenceStat()
        last = len(sorted_list)-1
        if sort_by == "different_count":
            for i in range(0, k):
                sentences.append(SentenceStat())
            for token in sorted_list:
                sentences[token.different_count-1].tokens.append(token)
            for i in range(0, k):
                if len(sentences[i].tokens) == 0:
                    print("No tokens in sub category")
                    return
                sentences[i].sentence_stat_init()
        else:
            for i, token in enumerate(sorted_list):
                current.tokens.append(token)
                if len(current.tokens) == step_size or i == last:
                    current.sentence_stat_init()
                    sentences.append(current)
                    current = SentenceStat()

        if (len(sentences) < 4):
            print("Too few datapoints:", len(sentences))
            return
        print("Added", len(sentences), "data points")
        x = []
        r1 = []
        r5 = []
        rk = []
        eq = []
        lb = []
        for data_point in sentences:
            if sort_by == "cos":
                x.append(data_point.total_avg_cos_sim)
            elif sort_by == "distance":
                x.append(data_point.avg_most_similar_dist)
            elif sort_by == "different_count":
                x.append(data_point.tokens[0].different_count)

            r1.append(data_point.sum_1)
            r5.append(data_point.sum_5)
            rk.append(data_point.sum_k)
            eq.append(data_point.retriev_eq_most_likely)
            lb.append(data_point.annotation_correct)
        incorrect = list(map(lambda x: x.error_count[Error.INCORRECT.value], sentences))
        order = list(map(lambda x: x.error_count[Error.ORDER.value], sentences))
        ending = list(map(lambda x: x.error_count[Error.ENDING.value], sentences))
        extra = list(map(lambda x: x.error_count[Error.EXTRA.value], sentences))
        name = list(map(lambda x: x.error_count[Error.NAME.value], sentences))
        source = list(map(lambda x: x.error_count[Error.SOURCE.value], sentences))
        incomplete = list(map(lambda x: x.error_count[Error.INCOMPLETE.value], sentences))
        if show == "no_label":
            plt.plot(x, r1, "r", label="Most likely Token in Top1")
            plt.plot(x, r5, "g", label="Most likely Token in Top5")
            plt.plot(x, rk, "b", label="Most likely Token in Top8")
            plt.plot(x, eq, "k", label="Most likely Token equals model output")
            plt.ylabel("Percentage of tokens")
        elif show == "correct":
            #plt.plot(x, eq, "--ro", label="Chosen Token equals model output")
            plt.plot(x, lb, '--bo', label="Label correct")
            plt.axhline(average_correct, label="total correct tokens")
            plt.ylabel("Percentage of correct tokens")
        elif show == "error":
            plt.plot(x, incorrect, label="Incorrect words")
            plt.plot(x, order, label="Word order")
            plt.plot(x, ending, label="Word ending")
            plt.plot(x, extra, label="Additional word")
            plt.plot(x, name, label="Entity name")
            plt.plot(x, source, label="Source words")
            plt.plot(x, incomplete, label="Incomplete words")
            plt.ylabel("Percentage of tokens")
        left = True
        xlabel = ""
        if sort_by == "cos":
            left = left
            xlabel = "Sentence similarity"
        elif sort_by == "distance":
            left = not left
            xlabel = "KNN-distance"
        elif sort_by == "different_count":
            left = not left
            xlabel = "Different KNN-recommendations"
        if show == "correct":
            left = left
        if show == "error":
            left = not left
        if left:
            plt.legend(loc="upper left")
        else:
            plt.legend(loc="upper right")
        plt.xlabel(xlabel)
        filename = sort_by + str(step_size)+ "_" + show
        save_plot(folder_name, filename)

    def error_type_plot(stats: SentenceStat, folder_name, mode):
        print("Print error plot")
        if mode == 1 or mode == 2:
            return
        x = ["Wrong word", "Word order", "Word ending", "Additional word", "Entity Name", "Source word", "Incomplete word"]
        y = stats.error_count[1:8]
        #_, ax = plt.subplots()
        #ax.invert_yaxis()
        plt.subplots_adjust(left=0.25)
        plt.barh(x, y)
        plt.xlabel("Percentage of total")
        save_plot(folder_name, "error_type")
        print()

    def analyze_fixed_data_range(stats: List[SentenceStat], filter_setting):
        datapoints = []
        for stat in stats:
            for i in range(1, stat.tgt_count-1):
                if not filter_token(stat, i, **filter_setting):
                    continue


    def analyze_with_filter(stats: List[SentenceStat], filter_setting):
        print(filter_setting)
        filter_stats = {}
        statfile = "filter_stats.bin"
        if os.path.exists(statfile):
            print("Found stat file")
            with open(statfile, "rb") as f:
                filter_stats = pickle.load(f)
        data_points = 0
        token_number = 0
        analysed_number = 0
        avg_sum1 = []
        avg_sum5 = []
        avg_sumk = []
        avg_prediction_correct = []
        avg_prec = []
        avg_rec = []
        avg_cos_sim = []
        avg_distances = []
        all_distance = []
        all_cos_sim = []
        for stat in stats:
            # Skip to first token to reduce influence of individual style
            token_number += stat.tgt_count-1
            sum1 = 0
            sum5 = 0
            sumk = 0
            prediction_correct = 0
            prec = []
            rec = []
            cos_sim = []
            distances = []
            tokens = 0
            enc_to_str(stat.max_prob_token)
            # tgt_count-1 to prevent iterating over EOS
            for i in range(1, stat.tgt_count-1):
                if filter_token(stat, i, **filter_setting):
                    # Passed filter
                    tokens += 1
                else:
                    continue
                sum1 += stat.sum1_values[i]
                sum5 += stat.sum5_values[i]
                sumk += stat.sumk_values[i]
                prediction_correct += 1 if stat.max_prob_token[i] == stat.generated_token[i+1] else 0
                prec.append(stat.precision[i])
                rec.append(stat.recall[i])
                cos_sim.append(stat.token_avg_cos_sim[i])
                all_cos_sim += stat.all_cos_sims[i]
                distances.append(sum(stat.distances[i]) / len(stat.distances[i]))
                all_distance += stat.distances[i]
            if tokens == 0:
                continue
            avg_sum1.append(sum1 / tokens)
            avg_sum5.append(sum5 / tokens)
            avg_sumk.append(sumk / tokens)
            avg_prediction_correct.append(prediction_correct / tokens)
            avg_prec.append(math.fsum(prec) / tokens)
            avg_rec.append(math.fsum(rec) / tokens)
            avg_cos_sim.append(math.fsum(cos_sim) / tokens)
            avg_distances.append(math.fsum(distances) / tokens)
            data_points += 1
            analysed_number += tokens
        if data_points == 0:
            print("No datapoints")
            return None
        print("Datapoints", data_points)
        collected = {
            "analysed": analysed_number,
            "total": token_number,
            "min_dist": filter_setting["min_dist"],
            "max_dist": filter_setting["max_dist"],
            "min_sim": filter_setting["min_sim"],
            "max_sim": filter_setting["max_sim"],
            "avg_pred_corrrect": math.fsum(avg_prediction_correct) / data_points,
            "avg_sum1": math.fsum(avg_sum1) / data_points,
            "avg_sum5": math.fsum(avg_sum5) / data_points,
            "avg_sumk": math.fsum(avg_sumk) / data_points,
            "avg_prec": math.fsum(avg_prec) / data_points,
            "avg_rec": math.fsum(avg_rec) / data_points,
            "avg_cos_sim": math.fsum(avg_cos_sim) / data_points,
            "avg_distances": math.fsum(avg_distances) / data_points,
            "sim_dist_cor": spearmanr(all_cos_sim, all_distance),
        }
        filter_stats[str(filter_setting)] = collected
        with open(statfile, "wb") as f: # "wb" because we want to write in binary mode
            pickle.dump(filter_stats, f)
        save_readable(filter_stats)
        return collected
    def save_readable(filter_stats):
        with open('readable_stats.txt', 'w') as f:
            for setting, stats in filter_stats.items():
                f.write("Setting:" + "\n")
                f.write("\t" + setting + "\n")
                analysed = stats["analysed"]
                total = stats["total"]
                f.write("\tLooking at " + str(analysed) + "/" + str(total) + "=" + str(analysed/total*100) + "\% data points\n")
                f.write("Values:" + "\n")
                f.write("\tPrediction correct: " + str(stats["avg_pred_corrrect"]))
                f.write("\tAppearance in top1: " + str(stats["avg_sum1"]) + "\n")
                f.write("\tAppearance in top5: " + str(stats["avg_sum5"]) + "\n")
                f.write("\tAppearance in topk: " + str(stats["avg_sumk"]) + "\n")
                f.write("\tAverage precision of recalled words: " + str(stats["avg_prec"]) + "\n")
                f.write("\tAverage recall of recalled words: " + str(stats["avg_rec"]) + "\n")
                f.write("\tAverage cos distance between recalled sentences: " + str(stats["avg_cos_sim"]) + "\n")
                f.write("\tAverage knn distance " + str(stats["avg_distances"]) + "\n")
                f.write("\tCorrelation between distance and similarity " + str(stats["sim_dist_cor"]) + "\n")
                f.write("\n")
    def knn_statistics(use_target = True, comet_use_target = False):
        before = time.perf_counter()
        total = len(test_ds)
        size = 500
        size_map = [None] * size
        res = []
        for i in range (0, size):
            size_map[i] = []
        stats = []
        # todo implement baseline, ideally of similar length
        file_name = "knn_stats_target" if use_target else "ted_stat_hyp"
        file_name += "_comet_target_" + str(comet_use_target)
        file_name = "data/" + file_name + "_layer_" + str(knn_store_layer)
        if os.path.exists(file_name):
            print("Found existing file")
            with open(file_name, "rb") as f:
                stats = pickle.load(f)
        else:
            comet_data = []
            for i, sample in enumerate(test_ds):
                src = sample["source"].cuda().unsqueeze(0)
                tgt = sample["target"].cuda().unsqueeze(0) if use_target else generate_hypos_from_tokens(src)[0][0]["tokens"].unsqueeze(0)
                knn = whole_sentence_info_from_enc(src, tgt)
                statistic = SentenceStat()
                #comet_data.append({"src": enc_to_str(src),"mt": enc_to_str(tgt),"ref": enc_to_str(sample["target"])})
                comet_data.append({"src": enc_to_str(src),"mt": enc_to_str(tgt)})
                token_stat_init(statistic, src[0], tgt[0], knn)
                stats.append(statistic)
                size_map[src.shape[0]].append((src, tgt))
                print(str(i) + "/" + str(total) + " " + str(100 * i/total) + "%" , end='\r')
            model_path = download_model("Unbabel/wmt22-comet-da") if comet_use_target else download_model("Unbabel/wmt20-comet-qe-da")
            model = load_from_checkpoint(model_path)
            scores = model.predict(comet_data, batch_size=8, gpus=1)[0]
            for i in range(0, len(scores)):
                stats[i].score = scores[i]
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "wb") as f: # "wb" because we want to write in binary mode
                pickle.dump(stats, f)
        dur = time.perf_counter() - before
        print("\n" + str(dur) + " seconds")
        return stats
    def load_mt_gender():
        file_name = "data/mt_gender_bin"
        if os.path.exists(file_name):
            print("Found existing file")
            with open(file_name, "rb") as f:
                mt = pickle.load(f)
                return mt
        else:
            f = open("en.txt", "r")
            analysis = []
            lines = f.readlines()
            line_count = len(lines)
            for i, line in enumerate(lines):
                seperated = line.split("\t")
                src = seperated[2]
                info = analyze_translation(src)
                stat = SentenceStat()
                token_stat_init(stat, info["src_enc"], info["tgt_enc"], info["knn_retrieval"])
                analysis.append((seperated[0], int(seperated[1]), seperated[3], stat))
                print(str(i) + "/" + str(line_count) + " " + str(100 * i/line_count) + "%" , end='\r')
            with open(file_name, "wb") as f:
                pickle.dump(analysis, f)
            return analysis
    def analyze_gender_set(stats):
        categories = [[[], [], []], [[], [], []], [[], [], []]]

        for s in stats:
            correct = s[0]
            hint = s[1]
            target = s[2]
            token_info: SentenceStat= s[3]

            should = 0
            if correct == "male":
                should = 0
            elif correct == "female":
                should = 1
            else:
                should = 2
            actual = 0
            if "Der" in token_info.tgt_str:
                actual = 0
            elif "Die" in token_info.tgt_str:
                actual = 1
            else:
                actual = 2
            if not ((should == 1 and actual == 1) or (should == 2 and actual == 2)):
                print(token_info.src_str, "____", token_info.tgt_str)
            categories[should][actual].append(token_info.avg_most_similar_dist)
        for i in range(0, 3):
            for j in range(0, 3):
                print(i, j, len(categories[i][j]) )
                if len(categories[i][j]) > 50:
                    categories[i][j] = math.fsum(categories[i][j]) / len(categories[i][j])
                else:
                    categories[i][j] = None
        breakpoint()
    def load_anotation_set():
        model_path = download_model("Unbabel/wmt20-comet-qe-da")
        model = load_from_checkpoint(model_path)
        comet_data = []

        file_name = "data/premade_anotations_layer_" + str(knn_store_layer) + ".bin"
        if os.path.exists(file_name):

            print("Found existing file")
            with open(file_name, "rb") as f:
                data = pickle.load(f)
                for d in data:
                    comet_data.append({"src": (d.src_str),"mt": (d.tgt_str)})
                res = model.predict(comet_data, batch_size=8, gpus=1)[0]
                for i in range(0, len(res)):
                    data[i].score = res[i]
                return data
        print("No file found, calculating data")
        stat_list = []
        def save(src, tgt, annotation):
            if not 3 in src and (tgt.shape[0] != len(annotation)+1):
                print("Error: annotation size not equal to tokens")
                exit()
            annotation.append(Error.CORRECT)
            knn = whole_sentence_info_from_enc(src.unsqueeze(0), tgt.unsqueeze(0))
            statistic = SentenceStat()
            token_stat_init(statistic, src, tgt, knn, annotation)
            comet_data.append({"src": enc_to_str(src),"mt": enc_to_str(tgt)})
            stat_list.append(statistic)
        print("Started read annotation data set")
        sources = open("files/qa.src","r").readlines()
        print("Read sources")
        translations = open("files/qa.mt","r").readlines()
        print("Read translations")
        sources = list(map(lambda s: encode_str(s, src_dict), sources))
        print("Source to token")
        translations = list(map(lambda s: encode_str(s, src_dict), translations))
        print("Translation to token")
        token_labels = open("files/qa.tags","r").readlines()
        labels = []
        encode_str("test", tgt_dict)
        enc_to_str(translations[4])
        prev_sentence_id = 0
        actual_token_counter = 0
        total = len(token_labels)
        has_unk = False
        added = 0
        for i, label in enumerate(token_labels):
            print(str(i) + "/" + str(total) + " " + str(100 * i/total) + "%" , end='\r')
            columns = label.split("\t")
            sentence_id = int(columns[3])
            tag = Error.CORRECT if columns[6] == "OK\n" else Error.INCORRECT
            token = columns[5]
            rem_token_length = len(token)
            #print("Next word: |", token, "| len: " , rem_token_length)
            tgt_dict[1]
            if sentence_id != prev_sentence_id:
                actual_token_counter = 0
                if not has_unk:
                    save(sources[prev_sentence_id], translations[prev_sentence_id], labels)
                    added += 1
                prev_sentence_id = sentence_id
                has_unk = False
                labels = []
            while not has_unk:
                if (translations[sentence_id][actual_token_counter] == 3):
                    has_unk = True
                    print("Unknown Token generated")
                    break
                next_token = tgt_dict[translations[sentence_id][actual_token_counter]]
                next_token = next_token.replace("&amp;", "'")
                if next_token == '",' or next_token == '".':
                    # This is one token in sentencepiece, but two "words"
                    has_unk = True
                    print("\nEncountered bad token")
                    break
                if use_sp:
                    next_token = next_token.replace("▁", "")
                else:
                    next_token = next_token.replace("@@", "")
                rem_token_length -= len(next_token)
                #print("SUB " + str(len(next_token)) + "\n")
                labels.append(tag)
                #print("Read |", next_token, "|(", actual_token_counter, ")remaining:", rem_token_length)

                actual_token_counter += 1
                prev_sentence_id = sentence_id
                if rem_token_length == 0:
                    break
                elif rem_token_length < 0:
                    print("Error ")
                    print(sentence_id, ":", enc_to_str(translations[sentence_id], tgt_dict))
                    has_unk = True
                    break
        if not has_unk:
            added += 1
            save(sources[-1], translations[-1], labels)
        print("\Read", added+1, "of", prev_sentence_id, "sentences")

        print("Adding score")
        res = model.predict(comet_data, batch_size=8, gpus=1)[0]
        for i in range(0, len(res)):
            data[i].score = res[i]

        print("Saving data")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(stat_list, f)
        return stat_list
    def load_wmt_mqm_human_evaluation():
        #judge_name = "rater1"
        judge_name = "rater4"
        #modelname = "Tencent_Translation.1520"
        modelname = "Online-A.1574"
        file_name = "data/wmt_human_" + judge_name + "_" + modelname + ".bin"
        #file_name = "data/wmt_human_eval.bin"
        if os.path.exists(file_name):
            print("Found existing file")
            with open(file_name, "rb") as f:
                mt = pickle.load(f)
                return mt
        else:
            f = open("files/qe.tsv", "r")
            lines = f.readlines()[1:]
            added = {}
            total_lines = 0
            count = 0
            print("Starting adding error annotations")
            for line in lines:
                count += 1
                line = line.split("\t")
                id = int(line[3])
                if line[4] != judge_name:
                    continue
                if line[0] != modelname:
                    continue
                total_lines += 1
                source = line[5]
                mt = line[6]
                # Change for better parsing
                mt = mt.replace("</v>", "▁").replace("<v>", "▁")
                enc = encode_str(mt.replace("▁", ""), tgt_dict)
                if not source in added:
                    type_array = [WmtErrorType.Noerror] * enc.shape[0]
                    severity_array = [WmtErrorSeverity.noerror] * enc.shape[0]
                    added[source] = (enc, type_array, severity_array)
                current = added[source]

                error = line[7]
                severity = line[8]
                current_error = False

                token_id = 0
                current_token = tgt_dict[enc[0]].replace("▁", "")
                rem_length = len(current_token)
                part_token = ""
                if len(enc) != len(current[1]):
                    print("Error in ", count, ", skipping...")
                    continue
                for c in mt:
                    if c == ' ':
                        continue
                    if c == "▁":
                        current_error = not current_error
                        continue
                    part_token += c
                    rem_length -= 1
                    if current_error:
                        # Dont care about sub errors
                        # print("ERROR:")
                        # print(line)
                        # print(error)
                        e_error = error.split("/")[0].replace(" ", "").replace("-", "").replace("!", "")
                        current[1][token_id] = WmtErrorType[e_error]
                        current[2][token_id] = WmtErrorSeverity[severity.strip()]
                    if rem_length == 0:
                        if token_id >= len(current[1])-1:
                            break
                        token_id += 1
                        current_token = tgt_dict[enc[token_id]].replace("▁", "")
                        #print("current_token", current_token)
                        rem_length = len(current_token)
                        #print("Rem_length", rem_length)
                        count = 0
                        part_token = ""

            print("Finished adding error annotations")
            print("Starting creating knn data")
            stats = []
            i = 0
            total = len(added)
            for src, (tgt_enc, errors, severities) in added.items():
                print(str(i) + "/" + str(total) + " " + str(100 * i/total) + "%" , end='\r')
                i += 1
                src_enc = encode_str(src, src_dict)
                knn = whole_sentence_info_from_enc(src_enc.unsqueeze(0), tgt_enc.unsqueeze(0))
                stat = SentenceStat()
                annotation = list(map(lambda t: WmtError(t[0], t[1]), zip(errors, severities)))
                token_stat_init(stat, src_enc, tgt_enc, knn, annotation)
                stats.append(stat)
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "wb") as f: # "wb" because we want to write in binary mode
                pickle.dump(stats, f)
            print("\nFinished creating knn data")
            return stats
    def collapse(stats: List[SentenceStat], filters):
        token_lists = map(lambda x: x.tokens, stats)
        res = []
        for i in token_lists:
            res += i
        stat = SentenceStat()
        stat.tokens = res
        stat.sentence_stat_init()
        return stat
    def create_distribution_plot(stats: SentenceStat, folder_name, mode, measure: Literal['knn', 'cos'], top=1):
        if mode == 1 or mode == 5:
            return
        has_error = []
        major = []
        minor = []
        neutral = []
        categories = []
        if mode == 6:
            categories = [[], [], [], []]
            for token in stats.tokens:
                categories[token.annotation.severity.value].append(token)
        else:
            categories = [[], []]
            for token in stats.tokens:
                categories[not token.annotation.is_correct()].append(token)
                # if token.annotation.is_correct():
                #     no_error.append(token)
                # else:
                #     has_error.append(token)
        # correct_stat = SentenceStat()
        # correct_stat.tokens = no_error
        # correct_stat.sentence_stat_init()
        # error_stat = SentenceStat()
        # error_stat.tokens = has_error
        # error_stat.sentence_stat_init()

        categorized_stats = []
        for c in categories:
            stat = SentenceStat()
            stat.tokens = c
            stat.sentence_stat_init()
            categorized_stats.append(stat)

        correct_measure = None
        error_measure = None
        measures = []
        label = ""
        if measure == "knn":
            for c in categorized_stats:
                measures.append(list(map(lambda x: sum(x.distances[0:top]) / top, c.tokens)))
            # correct_measure = list(map(lambda x: sum(x.distances[0:top]) / top, correct_stat.tokens))
            # error_measure = list(map(lambda x: sum(x.distances[0:top]) / top, error_stat.tokens))
            label = "KNN-Distance"
        elif measure == "cos":
            for co in categorized_stats:
                measures.append(list(map(lambda x: (sum(x.cos_sims[0:top]) / top), co.tokens)))
            # correct_measure = list(map(lambda x: sum(x.cos_sims[0:top]).item() / top, correct_stat.tokens))
            # error_measure = list(map(lambda x: sum(x.cos_sims[0:top]).item() / top, error_stat.tokens))
            label = "Cos-similarity"
        else:
            print("Measure", measure, "does not exist")
            exit()
        d = None
        if mode == 6:
            d = {'distances': measures[0] + measures[1] + measures[2] + measures[3],
             'Severity': ["Correct"] * len(measures[0]) + ["Neutral"] * len(measures[1])
                + ["Minor"] * len(measures[2]) + ["Major"] * len(measures[3])}
        else:
            d = {'distances': measures[0] + measures[1],
             'Severity': ["Correct"] * len(measures[0]) + ["Incorrect"] * len(measures[1])}
        frame = pd.DataFrame(data=d)
        data_wide = frame.pivot(columns="Severity", values="distances")
        #data_wide = frame.pivot(columns="annotation", values="distances")
        figure = data_wide.plot.density()
        figure.set_xlabel(label)
        plt.subplots_adjust(left=0.2)
        file_name =  "distribution/" + measure + "_top" + str(top)
        #os.makedirs(os.path.dirname(file_name), exist_ok=True)
        #figure.get_figure().savefig(file_name)
        save_plot(folder_name, file_name)

    def create_error_by_knn_match(stats: SentenceStat, folder_name, mode):
        l = []
        for i in range(0, k+1):
            l.append(SentenceStat())
        for token in stats.tokens:
            l[token.total_correct_predicitions].tokens.append(token)
        for i in range(0, k+1):
            l[i].sentence_stat_init()
        y = list(map(lambda x: x.annotation_correct, l))
        x = list(range(0, 9))
        x.reverse()
        y.reverse()
        plt.bar(x, y)
        plt.ylabel("Percentage of correct tokens")
        plt.xlabel("Retrived KNN-tokens that equal token chosen by model")
        save_plot(folder_name, "topk_to_annotation")

    def create_rarity_graph(stats: SentenceStat, folder_name, mode, top = 1):
        tokens = list(map(lambda x: x.chosen_token_id.item() - 4, stats.tokens))
        distances = list(map(lambda x: sum(x.distances[0:top]) / top, stats.tokens))
        dict_lines = open("dict.de.txt", "r").readlines()
        rarities = list(map(lambda x: int(x.split(" ")[1].replace("\n", "")), dict_lines))
        token_rarities = list(map(lambda x: rarities[x], tokens))
        print("Max", max(token_rarities))
        plt.yscale("log")
        plt.plot(distances, token_rarities, "o", markersize=0.1)
        plt.xlabel("KNN-distance")
        plt.ylabel("Occurences")
        save_plot(folder_name, "rarity_occurences")
        plt.yscale("log")
        plt.plot(distances, tokens, "o", markersize=0.1)
        plt.xlabel("KNN-distance")
        plt.ylabel("Rarity rank")
        save_plot(folder_name, "rarity_rank")

    def analyze_part_of_speech(error_types, mode, folder):
        print("Analyse parts of speech distribution")
        y = list(map(lambda x: x.avg_most_similar_dist, error_types.values()))
        x = list(error_types.keys())
        for i in range(0, len(x)):
            x[i] += " (" + str(len(error_types[x[i]].tokens)) + ")"
        plt.subplots_adjust(left=0.25)
        plt.barh(x, y)
        plt.xlabel("KNN-Distance")
        save_plot(folder, "pos_to_distance")
        error_percentage = list(map(lambda x: x.annotation_correct, error_types.values()))
        plt.subplots_adjust(left=0.25)
        plt.barh(x, error_percentage)
        plt.xlabel("Percentage of correct tokens")
        save_plot(folder, "pos_to_error")
        print("Finished parts of speech distribution")

    def threshold_prediction(single: SentenceStat, mode, folder_name, criteria = 0):
        print("Beginning treshold analysis")
        name = "Probability" if criteria == 0 else "KNN-Distance"
        x = []
        y_precision = []
        y_recall = []
        y_f = []
        y_mcc = []
        for i in range(1, 200):
            true_pos = 0
            false_pos = 0
            true_neg = 0
            false_neg = 0
            threshold = i * 0.001 if criteria == 0 else i * 10
            preds = []
            actual = []
            probs = []
            for token in single.tokens:
                probs.append(token.chosen_token_prob)
                pred = not ( token.chosen_token_prob >= threshold if criteria == 0 else token.distances[0] <= threshold )
                target = not token.annotation.is_correct()
                preds.append(pred)
                actual.append(target)
                if pred:
                    if target:
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    if target:
                        false_neg += 1
                    else:
                        true_neg += 1
            #print("Treshhold:", threshold, true_pos, false_pos, true_neg, false_neg)
            if true_pos == 0:
                #print("Treshhold:", threshold, "No found", true_pos, false_pos, true_neg, false_neg)
                continue
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            f = 2 / ( (1/precision) + (1/recall))
            average_mcc = matthews_corrcoef(actual, preds)
            x.append(threshold)
            y_precision.append(precision)
            y_recall.append(recall)
            y_f.append(f)
            y_mcc.append(average_mcc)

            #print("TMCC: reshhold:", average_mcc)
            #print("Treshhold:", threshold, "Precision:", precision, "Recall:", recall, "F:", f)
        print("\t", name, "Max mcc", max(y_mcc))
        print("\t", name, "Max f-score", max(y_f))
        if folder_name == "":
            return
        plt.plot(x, y_precision, label="Precision")
        plt.plot(x, y_recall, label="Recall")
        plt.plot(x, y_f, label="f-score")
        plt.plot(x, y_mcc, label="MCC-score")

        plt.xlabel("Treshold with " + name)
        plt.ylabel("Value")
        plt.legend()
        save_plot(folder_name, name + "_treshold")
        print("Finished treshold analysis")

    def analyze_comet_score(stats: List[SentenceStat]):
        import statistics
        scores = list(map(lambda x: x.score, stats))
        distances = list(map(lambda s: list(map(lambda x: statistics.mean(x.distances[0:8]), s.tokens)), stats))
        #distances = list(map(lambda s: list(map(lambda x: statistics.mean(x.cos_sims[0:8].tolist()), s.tokens)), stats))
        average = list(map(lambda x: statistics.mean(x), distances))
        median = list(map(lambda x: statistics.median(x), distances))
        m = list(map(lambda x: max(x), distances))
        person_maxk = []
        person_mink = []
        for i in range(1, 11):
            max_i = list(map(lambda x: statistics.mean(sorted(x)[-i:]), distances))
            person_maxk.append(pearsonr(scores, max_i))
            min_i = list(map(lambda x: statistics.mean(sorted(x)[:i]), distances))
            person_mink.append(pearsonr(scores, min_i))

        average_pearson = pearsonr(scores, average)
        median_pearson = pearsonr(scores, median)
        max_pearson = pearsonr(scores, m)
        print("Analyzing the pearson correlation to the comet score")
        print("\tAverage:", average_pearson)
        print("\tmedian:", median_pearson)
        print("\tMax1:", person_maxk[0])
        print("\tMax5:", person_maxk[4])
        print("\tMax10:", person_maxk[9])
        print("\tMin1:", person_mink[0])
        print("\tMin5:", person_mink[4])
        print("\tMin10:", person_mink[9])
        print("Finished comet score analysis")

    def print_annotation(stats: List[SentenceStat], folder_name):
        file_name = "text/custom/layer_6/ted_annotations.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as f:
            for s in stats:
                f.write(s.src_str)
                f.write("\t")
                for t in s.tokens:
                    f.write(tgt_dict[t.chosen_token_id].replace("▁", " ") + "[" + str(t.annotation) + "]")
                f.write("\n")

    def analyze_correlation(single, tops = [1,2,4,8]):
        print("Analysing pearson correlation")
        for use in ["dis", "cos"]:
            print("\tAnalysing the", use, "metric")
            for top in tops:
                annotation = None
                if use == "dis":
                    annotation = list(map(lambda x: sum(x.distances[0:top]) / top, single.tokens))
                elif use == "cos":
                    annotation = list(map(lambda x: sum(x.cos_sims[0:top]) / top, single.tokens))
                corrects = list(map(lambda x: not x.annotation.is_correct(), single.tokens))
                prob = list(map(lambda x: x.chosen_token_prob, single.tokens))
                anno_correct = pearsonr(annotation, corrects)
                prob_correct = pearsonr(prob, corrects)
                prob_anno = pearsonr(prob, annotation)
                print("\t\tLooking at average of top", top, "metric:", use)
                print("\t\tanno-coorect:", anno_correct, "prob-correct:", prob_correct, "prob-anno:", prob_anno)
        print("Finished analysis")
    def random_baseline(mode = 0, measure = "cos"):
        mode_name = "random data" if mode == 0 else "training data" if mode == 1 else "non-training data"
        file_name = "data/demonstrate_" + mode_name + ".pyobject"
        data = None
        if os.path.exists(file_name):
            print("Found existing file")
            with open(file_name, "rb") as f:
                mt = pickle.load(f)
                data = mt
                print("Read from file")
        else:
            print("Building new File")
            stats = []
            total = 1000
            for c in range(0, total):
                src_e = []
                tgt_e = []
                if mode == 0:
                    for i in range(0, 15 + int(random.random() * 10)):
                        src_e.append(int(random.random() * 9800))
                        tgt_e.append(int(random.random() * 9800))
                elif mode == 1:
                    src_e = ds[c * 100]["source"][:-1]
                    tgt_e = ds[c * 100]["target"][:-1]
                else:
                    src_e = test_ds[c * 4]["source"][:-1]
                    tgt_e = test_ds[c * 4]["target"][:-1]

                src_e = torch.tensor(src_e).cuda().unsqueeze(0)
                tgt_e = torch.tensor(tgt_e).cuda().unsqueeze(0)
                knn = whole_sentence_info_from_enc(src_e, tgt_e)
                statistic = SentenceStat()
                token_stat_init(statistic, src_e[0], tgt_e[0], knn)
                stats.append(statistic)
                print(str(c) + "/" + str(total) + " " + str(100 * c/total) + "%" , end='\r')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "wb") as f: # "wb" because we want to write in binary mode
                pickle.dump(stats, f)
                data = stats
                print("Saved file")
        single = collapse(data, "")
        folder_name = "plots/display"
        top = 1
        d = None
        label = ("knn-distance" if measure == "dis" else "sentence similarity") + " of " + mode_name
        l = None
        if measure == "dis":
            l = list(map(lambda x: sum(x.distances[0:top]) / top, single.tokens))
            d = {label: l}
        else:
            l = list(map(lambda x: sum(x.cos_sims[0:top]) / top, single.tokens))
            d = {label: l}
        print("Variance:", np.var(l))
        print("Average:", np.mean(l))
        print("Median:", np.median(l))
        frame = pd.DataFrame(data=d)
        figure = frame.plot.density()
        figure.legend().remove()
        figure.set_xlabel(label)
        file_name =  mode_name + "_" + measure + "_top" + str(top)
        #os.makedirs(os.path.dirname(file_name), exist_ok=True)
        #figure.get_figure().savefig(file_name)
        save_plot(folder_name, file_name)
        create_sorted_graph(single, "cos", folder_name + "/" +mode_name, 4, "no_label", step_size=800)

    #info = analyze_translation("Who called you this morning?")
    #print_info(info)
    def pos_analysis(single: SentenceStat):
        print("Beginning part of speech analysis")
        print("All parts of speech:")
        analyze_correlation(single, [1])
        threshold_prediction(single, 2, "", 1)

        by_pos = single.sort_by_pos()
        count = 0
        for pos, stat in by_pos.items():
            if (len(stat.tokens) < 500):
                continue
            count += 1
            print("POS:", pos, "(", len(stat.tokens), "):")
            analyze_correlation(stat, [1])
            threshold_prediction(stat, 2, "", 1)
        print("Analyzed", count, "parts of speech")
        print("Finished part of speech analysis\n\n")

    stats = None
    mode = os.getenv('DATASET').lower()
    if mode == "full_ted":
        mode = 1
    elif mode == "ted":
        mode = 3
    elif mode == "news":
        mode = 4
    elif mode == "qa":
        mode = 2
    elif mode == "qe":
        mode = 6
    elif mode == "display":
        mode = 7
    elif mode == "custom":
        mode = 8
    else:
        print("Error: Non existing mode")
        exit()
    for mode in [mode]:
        show = ""
        folder_name = f"plots/{get_mt_model_name()}/"
        name = get_data_store() +  "_layer_" + str(knn_store_layer) + "/"
        if mode == 1:
            stats = knn_statistics(use_target=False, comet_use_target=False)
            folder_name += "full_ted/" + name
        elif mode == 2:
            stats = load_anotation_set()
            #analyze_comet_score(stats)
            show = ["correct"]
            #show = ["no_label"]
            folder_name += "qa/" + name
        elif mode == 3:
            stats = add_annotations(src="ted")
            show = ["correct", "error"]
            #show = ["correct"]
            folder_name += "ted/" + name
            #print_annotation(stats, folder_name)
        elif mode == 4:
            stats = add_annotations(filename="data/out_of_domain", src="files/news.txt")
            show = ["error", "correct"]
            folder_name += "news/" + name
        elif mode == 5:
            stats = load_mt_gender()
            analyze_gender_set(stats)
            breakpoint()
            continue
        elif mode == 6:
            stats = load_wmt_mqm_human_evaluation()
            # def test(sen):
            #     for t in sen.tokens:
            #         print(src_dict[t.chosen_token_id], ":", t.annotation.type)
            # test(stats[0])
            folder_name += "qe/layer_" + str(knn_store_layer) + "/"
            show = ["correct"]
        elif mode == 7:
            print("Analysing baseline datasets")
            folder_name += "display/"
        elif mode == 8:
            custom_file_name = os.getenv('CUSTOM_FILE_NAME').lower()
            stats = add_annotations(filename=f"data/{get_mt_model_name()}/custom/{os.getenv('DATASTORE_NAME')}_{os.getenv('LAYER')}", src="custom")
            folder_name += "custom/" + custom_file_name + "/layer_" + str(knn_store_layer) + "/"


        from pathlib import Path
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        import sys
        print("Creating file", folder_name + "/log.txt")
        log = int(os.getenv("LOG_TO_FILE"))
        with (open(folder_name + "/log.txt", 'w') if log else sys.stdout) as sys.stdout:
            if mode == 7:
                for i in [0,1,2]:
                    for m in ["cos", "sin"]:
                        random_baseline(i, m)
                continue
            if mode == 8:
                analyze_comet_score(stats)
                continue

            single = collapse(stats, folder_name)
            analyze_comet_score(stats)
            if mode == 1:
                continue
            pos_analysis(single)
            by_pos = single.sort_by_error()
            analyze_correlation(single)

            threshold_prediction(single, mode, folder_name, 1)
            threshold_prediction(single, mode, folder_name, 0)
            print()
            #create_rarity_graph(single, folder_name, mode)
            #create_error_by_knn_match(single, folder_name, mode)
            print("Creating distribution graphs")
            for measure in ["knn", "cos"]:
                for i in range(1, 9):
                    create_distribution_plot(single, folder_name, mode, measure, i)
            print()

            error_type_plot(single, folder_name, mode)

            print("Print bucket graphs")
            for sort_criteria in ["different_count", "distance", "cos"]:
                for i in range(1, 6):
                    for s in show:
                        create_sorted_graph(single, sort_criteria, folder_name, mode, s, step_size=100 * (2**i))
            print()
    return 1

def get_knn_generation_parser(interactive=False, default_task="translation"):
    parser = options.get_parser("Generation", default_task)
    options.add_dataset_args(parser, gen=True)
    options.add_distributed_training_args(parser, default_world_size=1)
    ## knnbox related code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # compared to options.get_generation_parser(..), knnbox only add one line code below
    options.add_model_args(parser)
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
    options.add_generation_args(parser)
    if interactive:
        options.add_interactive_args(parser)
    return parser


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args, None)

if __name__ == "__main__":
    cli_main()