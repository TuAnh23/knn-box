from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from fairseq.models.fairseq_encoder import EncoderOut
from .deltalm import (
    DeltaLMModel,
    DeltaLMEncoder,
    DeltaLMDecoder,
    base_architecture_deltalm
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner


@register_model("vanilla_knn_mt_visual_deltalm")
class VanillaKNNMTVisualDeltaLMModel(DeltaLMModel):
    r"""
    The vanilla knn-mt model with visualization.
    """
    @staticmethod
    def add_args(parser):
        """
        add knn-mt related args here
        """
        DeltaLMModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "inference"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter k of vanilla knn-mt")
        parser.add_argument("--knn-lambda", type=float, metavar="D", default=0.7,
                            help="The hyper-parameter lambda of vanilla knn-mt")
        parser.add_argument("--knn-temperature", type=float, metavar="D", default=10,
                            help="The hyper-parameter temperature of vanilla knn-mt")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the DeltaLMDecoder with VanillaKNNMTVisualDecoderDeltaLM
        """
        return VanillaKNNMTVisualDecoderDeltaLM(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


class VanillaKNNMTVisualDecoderDeltaLM(DeltaLMDecoder):
    r"""
    The vanilla knn-mt Decoder with visualization, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)
            self.datastore = global_vars()["datastore"]

        elif args.knn_mode == "inference":
            self.datastore = Datastore.load(args.knn_datastore_path, \
                    load_list=["keys", "vals", "sentence_ids", "token_positions"])
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.combiner = Combiner(lambda_=args.knn_lambda,
                     temperature=args.knn_temperature, probability_dim=len(dictionary))

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs, ## we need get dynamic k from kwargs, so add it here
    ):
        r"""
        we overwrite this function to do something else besides forward the DeltaLMDecoder.

        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if self.args.knn_mode == "build_datastore":
            layer_number = 6
            layer = extra["inner_states"][layer_number].permute(1,0,2)
            keys = select_keys_with_pad_mask(layer, self.datastore.get_pad_mask())
            self.datastore["keys"].add(keys.half())

        elif self.args.knn_mode == "inference":
            # use the k from visual web interface
            k = int(kwargs["knn_parameter"]["k"])
            # retriever should return a lot of information here
            self.retriever.retrieve(x, k=k,
                return_list=["keys", "vals", "query", "distances", "sentence_ids", "token_positions"])

        if not features_only:
            x = self.output_layer(x)
        return x, extra


    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1.
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability

        compared to vanilla knn-mt, the visual version `get_noramlized_probs` need to return some extra infomation
        when do inference
        """
        if self.args.knn_mode == "inference":
            ## we get dynamic lambda and temperature from web UI
            ## we pass them from `knnbox-scripts/visual/src/knn_sequence_generator.py`
            ## original `fairseq/sequence_generator.py` dont pass sample to this function
            ## but inside `knnbox-scripts/visual/src/knn_sequence_generator.py` we pass it indeed.
            extra = {}
            lambda_ = sample["knn_parameter"]["lambda"]
            temperature = sample["knn_parameter"]["temperature"]

            knn_prob = self.combiner.get_knn_prob(**self.retriever.results,
                                                temperature=temperature, device=net_output[0].device)
            combined_prob, extra_combiner_info = self.combiner.get_combined_prob(knn_prob,
                                                net_output[0], lambda_=lambda_, log_probs=log_probs)

            # we return following infomation
            # knn-scripts/visual/src/knn_sequence_generator.py will record them when doing beam-search
            extra["neural_probs"] =  extra_combiner_info["neural_probs"]
            extra["combined_probs"] = extra_combiner_info["unlog_combined_probs"]
            extra["query_point"] =  self.retriever.results["query"]
            extra["knn_neighbors_values"] = self.retriever.results["vals"]
            extra["knn_neighbors_keys"] = self.retriever.results["keys"]
            extra["knn_l2_distance"] = self.retriever.results["distances"]
            extra["knn_sentence_ids"] = self.retriever.results["sentence_ids"]
            extra["knn_token_positions"] = self.retriever.results["token_positions"]

            return combined_prob, extra
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)


r""" Define some visual vanilla knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("vanilla_knn_mt_visual_deltalm", "vanilla_knn_mt_visual@deltalm_base")
def base_architecture(args):
    base_architecture_deltalm(args)