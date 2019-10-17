import pprint
import sys
from typing import List, Dict, Tuple, Iterable

import torch
import torch.nn as nn

from utils import util
from model.decoder import Decoder
from model.attentional_recurrent_subtoken_decoder \
    import AttentionalRecurrentSubtokenDecoder
from model.encoder import Encoder
from model.hybrid_encoder import HybridEncoder
from utils.dataset import Batcher, Example


class RenamingModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(RenamingModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config: Dict = None

    @property
    def batcher(self):
        if not hasattr(self, '_batcher'):
            _batcher = Batcher(self.config)
            setattr(self, '_batcher', _batcher)
        return self._batcher

    @property
    def device(self):
        return self.encoder.device

    @property
    def vocab(self):
        return self.encoder.vocab

    @staticmethod
    def default_params():
        return {
            'train': {
                'unchanged_variable_weight': 1.0,
                'max_epoch': 30,
                'patience': 5
            },
            'decoder': {
                'type': 'SimpleDecoder'
            }
        }

    @classmethod
    def build(cls, config):
        params = util.update(RenamingModel.default_params(), config)
        encoder = globals()[config['encoder']['type']].build(config['encoder'])
        decoder = globals()[config['decoder']['type']].build(config['decoder'])

        model = cls(encoder, decoder)
        params = util.update(params, {'encoder': encoder.config,
                                      'decoder': decoder.config})
        model.config = params
        # give the decoder a reference to the encoder
        model.decoder.encoder = encoder

        # assign batcher to sub-modules
        encoder.batcher = model.batcher
        decoder.batcher = model.batcher

        if False:
            print('Current Configuration:', file=sys.stderr)
            pp = pprint.PrettyPrinter(indent=2, stream=sys.stderr)
            pp.pprint(model.config)

        return model

    def forward(self,
                source_asts: Dict[str, torch.Tensor],
                prediction_target: Dict[str, torch.Tensor]):
        # type: (...) ->  Tuple[torch.Tensor, Dict]
        """Given a batch of decompiled abstract syntax trees, and the
        gold-standard renaming of variable nodes, compute the log-likelihood of
        the gold-standard renaming for training

        Arg:
            source_asts: a list of ASTs
            variable_name_maps: mapping of decompiled variable names to its
                renamed values

        Return:
            tensor of size batch_size denoting the log-likelihood of renamings

        """
        context_encoding = self.encoder(source_asts)
        var_name_log_probs = self.decoder(context_encoding, prediction_target)
        result = self.decoder.get_target_log_prob(var_name_log_probs,
                                                  prediction_target,
                                                  context_encoding)
        tgt_var_name_log_prob = result['tgt_var_name_log_prob']
        tgt_weight = prediction_target['variable_tgt_name_weight']
        weighted_log_prob = tgt_var_name_log_prob * tgt_weight
        tgt_var_encoding_indices_mask = \
            prediction_target['target_variable_encoding_indices_mask']
        ast_log_probs = \
            weighted_log_prob.sum(dim=-1) \
            / tgt_var_encoding_indices_mask.sum(-1)
        result['batch_log_prob'] = ast_log_probs
        return result

    def decode_dataset(self, dataset, batch_size=4096):
        # type: (...) -> Iterable[Tuple[Example, Dict]]
        with torch.no_grad():
            data_iter = dataset.batch_iterator(batch_size=batch_size,
                                               train=False,
                                               progress=False,
                                               config=self.config)
            self.eval()
            for batch in data_iter:
                rename_results = \
                    self.decoder.predict([e.ast for e in batch.examples],
                                         self.encoder)
                for example, result in zip(batch.examples, rename_results):
                    yield example, result

    def predict(self, examples: List[Example]):
        return self.decoder.predict(examples, self.encoder)

    def save(self, model_path, **kwargs):
        params = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }
        torch.save(params, model_path)

    @classmethod
    def load(cls, model_path, use_cuda=False, new_config=None):
        # type: (...) -> 'RenamingModel'
        device = torch.device("cuda:0" if use_cuda else "cpu")
        params = torch.load(model_path, map_location=lambda store, _: store)
        if new_config is not None:
            config = util.update(params['config'], new_config)
        kwargs = dict()
        if params['kwargs'] is not None:
            kwargs = params['kwargs']
        model = cls.build(config, **kwargs)
        model.load_state_dict(params['state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        return model
