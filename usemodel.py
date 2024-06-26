#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from itertools import chain
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils, data
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq.sequence_scorer import SequenceScorer

print("OUt of main")

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")

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

def main(args):
    print("second main")
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
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}
    

def _main(args, output_file):
    print("Calling generate")
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

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(args.model_overrides)

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

        # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(args)
    bpe = task.build_bpe(args)

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

    # Remove line comment
    # def encode_pair(src, tgt):
    #     src = src_dict.encode_line(encode_fn(src), add_if_not_exist=False).type(torch.int64).cuda()
    #     tgt = tgt_dict.encode_line(encode_fn(tgt), add_if_not_exist=False).type(torch.int64).cuda()
    #     return src, tgt
    # enc_src, ref_enc = encode_pair("Hallo Welt!", "Hello World!")
    # prev = torch.LongTensor([tgt_dict.eos() for _ in ref_enc]).unsqueeze(0).cuda()
    # pre = torch.LongTensor([tgt_dict.eos()]).cuda()
    # model = models[0]
    # src2 = enc_src.unsqueeze(0)
    # src_lengths = [enc_src.shape[0]]
    # res = model.forward(src2, src_lengths, prev)

    # for i in range(0, ref_enc.shape[0]):
    #     pre = torch.cat((pre, ref_enc[i].reshape(1)), 0)
    #     res = model.forward(src2, src_lengths, pre.unsqueeze(0), features_only=True)
    #     print((res[0][0][i]))
    #     print(pre)
    # exit()
    #print(torch.argmax(res[0][0][0]))

    ##res = model.forward(tensor([[  93, 4397,  573,  111,    2]], device='cuda:0'), src_lengths, prev)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_pos,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )


    # test code start
    # 
    # print(enc)
    # print(batch)
    # if enc.dim() == 1:
    #     enc = enc.unsqueeze(0)
    # if enc.size(-1) > min(models[0].max_positions()):
    #     raise ValueError(
    #         "tokens exceeds maximum length: {} > {}".format(
    #             enc.size(-1), models[0].model.max_positions()
    #         )
    #     )
    # prev_output_tokens = enc.clone()
    # print(prev_output_tokens)
    # prev_output_tokens[:, 0] = enc.gather(
    #     1,
    #     (enc.ne(task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1),
    # ).squeeze()

    # prev_output_tokens[:, 1:] = enc[:, :-1]
    # print(prev_output_tokens)
    # prev_output_tokens = utils.move_to_cuda(prev_output_tokens)
    # preds = models[0](**batch["net_input"], prev_output_tokens = prev_output_tokens)
    # print(preds)

    # test code end

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": args.lm_weight}
    generator = task.build_generator(
        models, args, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )
    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(args)
    bpe = task.build_bpe(args)

    scorer = scoring.build_scorer(args, tgt_dict)

    # sentence = "Gibt das Array mit den entfernten Elementen zurück"
    # #sentence = input('\nInput: ')
    # enc = src_dict.encode_line(encode_fn(sentence), add_if_not_exist=False).type(torch.int64)
    # #enc = src_dict.encode_line(sentence, add_if_not_exist=False,)
    # print(enc)
    # batch = data.language_pair_dataset.collate(
    #     samples=[{'id': -1, 'source': enc}],  # bsz = 1
    #     pad_idx=task.source_dictionary.pad(),
    #     eos_idx=task.source_dictionary.eos(),
    #     left_pad_source=False,
    #     input_feeding=False,
    # )
    # batch = utils.move_to_cuda(batch)
    # over


    inputs = ["Translating single words causes errors."]
    for batch in make_batches(inputs, task=task, args=args, max_positions=max_pos, encode_fn=encode_fn):
        src_tokens = batch.src_tokens.cuda()
        src_lengths = batch.src_lengths.cuda()
        constraints = batch.constraints
        if constraints is not None:
            constraints = constraints.cuda()
        sample = {
                    "net_input": {
                        "src_tokens": src_tokens,
                        "src_lengths": src_lengths,
                    },
                }
        hyposs = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=None,
            constraints=constraints,
        )
        for hypos in hyposs:
            for h in hypos:
                tokens = h["tokens"]
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=tokens.int().cpu(),
                    src_str="Wer kann die Maus bedienen",
                    alignment=h["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                print(hypo_str)
    exit()

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample["target"][:, : args.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]
        gen_timer.start()
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target"] is not None

            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )
            print("Source tokens")
            print(src_tokens)
            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        args.remove_bpe,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )
            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not args.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            print("Length:" + str(len(hypos[i])))
            for j, hypo in enumerate(hypos[i][: args.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not args.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                    if args.print_alignment:
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )

                    if args.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if getattr(args, "retain_iter_history", False):
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )

                # Score only the top hypothesis
                if has_target and j == 0:
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )
                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    if has_target:
        if args.bpe and not args.sacrebleu:
            if args.remove_bpe:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            "Generate {} with beam={}: {}".format(
                args.gen_subset, args.beam, scorer.result_string()
            ),
            file=output_file,
        )

    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
