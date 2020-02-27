from collections import OrderedDict, namedtuple

from model.model import *
from utils.vocab import END_OF_VARIABLE_TOKEN


class EnsembleModel(nn.Module):
    def __init__(self, model_paths, use_cuda, config):
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleList()
        for model_path in model_paths:
            model = RenamingModel.load(model_path, use_cuda=use_cuda)
            self.models.append(model)

        self.max_prediction_time_step = min(model.decoder.config['max_prediction_time_step'] for model in self.models)
        self.device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

        self.Hypothesis = namedtuple('Hypothesis', ['variable_list', 'variable_ptr', 'score'])
        self.config = config
        print('Ensemble Model Config', file=sys.stderr)
        print(self.config, file=sys.stderr)

    @classmethod
    def default_params(cls):
        return {
            'encoder': {
                        "type": "EnsembleModel"
                    },
            'decoder': {
                'beam_size': 5,
                'remove_duplicates_in_prediction': False
            },
            'train': {
                'eval_batch_size': 100,
                'num_batchers': 5,
                'num_readers': 5,
                'buffer_size': 100
            },
        }

    @classmethod
    def load(cls, model_paths, use_cuda=False, new_config=None):
        params = util.update(cls.default_params(), new_config)
        model = cls(model_paths, use_cuda, params)

        return model

    def predict(self, examples):
        context_encodings = []
        attention_memories = []
        variable_encodings = []
        remove_duplicate = self.config['decoder']['remove_duplicates_in_prediction']

        h_tm1 = []
        variable_name_embed_tm1 = []

        for model in self.models:
            tensor_dict = model.batcher.to_tensor_dict(examples)
            nn_util.to(tensor_dict, self.device)
            context_encoding = model.encoder(tensor_dict)
            # prepare tensors for attention
            attention_memory = model.decoder.get_attention_memory(context_encoding)
            h_0 = model.decoder.get_init_state(context_encoding)

            context_encodings.append(context_encoding)
            variable_encodings.append(context_encoding['variable_encoding'])
            attention_memories.append(attention_memory)
            h_tm1.append(h_0)

            att_tm1 = torch.zeros(len(examples), model.decoder.lstm_cell.hidden_size, device=self.device)
            variable_name_embed_tm1.append(att_tm1)

        context_encoding_t = attention_memories

        beam_size = self.config['decoder']['beam_size']
        batch_size = len(examples)
        vocab = self.models[0].decoder.vocab
        same_variable_id = vocab.target[SAME_VARIABLE_TOKEN]
        end_of_variable_id = vocab.target[END_OF_VARIABLE_TOKEN]
        variable_nums = [len(e.ast.variables) for e in examples]

        beams = OrderedDict((ast_id, [self.Hypothesis([[]], 0, 0.)]) for ast_id in range(batch_size))
        hyp_scores_tm1 = torch.zeros(len(beams), device=self.device)
        completed_hyps = [[] for _ in range(batch_size)]
        tgt_vocab_size = len(self.models[0].decoder.vocab.target)

        for t in range(0, self.max_prediction_time_step):
            candidate_cont_hyp_scores: List[torch.Tensor] = []
            model_h_t = []
            model_q_t = []
            for model_id, model in enumerate(self.models):
                # (total_live_hyp_num, encoding_size)
                if t > 0:
                    variable_encoding_t = variable_encodings[model_id][hyp_ast_ids_t, hyp_variable_ptrs_t]
                else:
                    variable_encoding_t = variable_encodings[model_id][:, 0]

                if model.decoder.config['input_feed']:
                    x = torch.cat([variable_encoding_t, variable_name_embed_tm1[model_id], att_tm1], dim=-1)
                else:
                    x = torch.cat([variable_encoding_t, variable_name_embed_tm1[model_id]], dim=-1)

                h_t, q_t, alpha_t = model.decoder.rnn_step(x, h_tm1[model_id], context_encoding_t[model_id])
                model_h_t.append(h_t)
                model_q_t.append(q_t)

                # (total_live_hyp_num, vocab_size)
                hyp_var_name_scores_t = torch.log_softmax(model.decoder.state2names(q_t), dim=-1)
                cont_cand_hyp_scores = hyp_scores_tm1.unsqueeze(-1) + hyp_var_name_scores_t
                candidate_cont_hyp_scores.append(cont_cand_hyp_scores)

            # (total_live_hyp_num, vocab_size)
            candidate_cont_hyp_scores = torch.logsumexp(torch.stack(candidate_cont_hyp_scores, dim=-1), dim=-1)

            new_beams = OrderedDict()
            live_beam_ids = []
            new_hyp_scores = []
            live_prev_hyp_ids = []
            new_hyp_var_name_ids = []
            new_hyp_ast_ids = []
            new_hyp_variable_ptrs = []
            is_same_variable_mask = []
            beam_start_hyp_pos = 0

            for beam_id, (ast_id, beam) in enumerate(beams.items()):
                beam_end_hyp_pos = beam_start_hyp_pos + len(beam)
                # (live_beam_size, vocab_size)
                beam_cont_cand_hyp_scores = candidate_cont_hyp_scores[beam_start_hyp_pos: beam_end_hyp_pos]

                cont_beam_size = beam_size - len(completed_hyps[ast_id])
                beam_new_hyp_scores, beam_new_hyp_positions = torch.topk(beam_cont_cand_hyp_scores.view(-1),
                                                                         k=cont_beam_size,
                                                                         dim=-1)

                # (cont_beam_size)
                beam_prev_hyp_ids = beam_new_hyp_positions / tgt_vocab_size
                beam_hyp_var_name_ids = beam_new_hyp_positions % tgt_vocab_size

                _prev_hyp_ids = beam_prev_hyp_ids.cpu()
                _hyp_var_name_ids = beam_hyp_var_name_ids.cpu()
                _new_hyp_scores = beam_new_hyp_scores.cpu()

                for i in range(cont_beam_size):
                    prev_hyp_id = _prev_hyp_ids[i].item()
                    prev_hyp = beam[prev_hyp_id]
                    hyp_var_name_id = _hyp_var_name_ids[i].item()
                    new_hyp_score = _new_hyp_scores[i].item()

                    variable_ptr = prev_hyp.variable_ptr
                    new_variable_list = list(prev_hyp.variable_list)
                    new_variable_list[-1] = list(new_variable_list[-1] + [hyp_var_name_id])

                    if hyp_var_name_id == end_of_variable_id:
                        # remove empty cases
                        if new_variable_list[-1] == [end_of_variable_id]:
                            continue

                        if remove_duplicate:
                            last_pred = new_variable_list[-1]
                            if any(x == last_pred for x in new_variable_list[:-1] if x != [same_variable_id, end_of_variable_id]):
                                # print('found a duplicate!', ', '.join([str(x) for x in last_pred]))
                                continue

                        variable_ptr += 1
                        new_variable_list.append([])

                    new_hyp = self.Hypothesis(variable_list=new_variable_list,
                                              variable_ptr=variable_ptr,
                                              score=new_hyp_score)

                    if variable_ptr == variable_nums[ast_id]:
                        completed_hyps[ast_id].append(new_hyp)
                    else:
                        new_beams.setdefault(ast_id, []).append(new_hyp)
                        live_beam_ids.append(beam_id)
                        new_hyp_scores.append(new_hyp_score)
                        live_prev_hyp_ids.append(beam_start_hyp_pos + prev_hyp_id)
                        new_hyp_var_name_ids.append(hyp_var_name_id)
                        new_hyp_ast_ids.append(ast_id)
                        new_hyp_variable_ptrs.append(variable_ptr)
                        is_same_variable_mask.append(1. if prev_hyp.variable_ptr == variable_ptr else 0.)

                beam_start_hyp_pos = beam_end_hyp_pos

            if live_beam_ids:
                hyp_scores_tm1 = torch.tensor(new_hyp_scores, device=self.device)
                h_tm1 = [(model_h_t[model_id][0][live_prev_hyp_ids], model_h_t[model_id][1][live_prev_hyp_ids])
                         for model_id in range(len(self.models))]
                att_tm1 = [model_q_t[model_id][live_prev_hyp_ids]
                           for model_id in range(len(self.models))]

                variable_name_embed_tm1 = [model.decoder.state2names.weight[new_hyp_var_name_ids]
                                           for model in self.models]
                hyp_ast_ids_t = new_hyp_ast_ids
                hyp_variable_ptrs_t = new_hyp_variable_ptrs

                beams = new_beams

                # (total_hyp_num, max_tree_size, node_encoding_size)
                context_encoding_t = [dict(attention_key=attention_memories[model_id]['attention_key'][hyp_ast_ids_t],
                                           attention_value=attention_memories[model_id]['attention_value'][hyp_ast_ids_t],
                                           attention_value_mask=attention_memories[model_id]['attention_value_mask'][hyp_ast_ids_t])
                                      for model_id in range(len(self.models))]

                # if self.independent_prediction_for_each_variable:
                #     is_same_variable_mask = torch.tensor(is_same_variable_mask, device=self.device, dtype=torch.float).unsqueeze(-1)
                #     h_tm1 = [(h[0] * is_same_variable_mask, h[1] * is_same_variable_mask) for h in h_tm1]
                #     att_tm1 = [x * is_same_variable_mask for x in att_tm1]
                #     variable_name_embed_tm1 = [x * is_same_variable_mask for x in variable_name_embed_tm1]
            else:
                break

        variable_rename_results = []
        for i, hyps in enumerate(completed_hyps):
            variable_rename_result = dict()
            ast = examples[i].ast
            hyps = sorted(hyps, key=lambda hyp: -hyp.score)

            if not hyps:
                # return identity renamings
                print(f'Failed to found a hypothesis for function {ast.compilation_unit}', file=sys.stderr)
                for old_name in ast.variables:
                    variable_rename_result[old_name] = {'new_name': old_name,
                                                        'prob': 0.}
            else:
                top_hyp = hyps[0]
                # sub_token_ptr = 0
                # for old_name in ast.variables:
                #     sub_token_begin = sub_token_ptr
                #     while top_hyp.variable_list[sub_token_ptr] != end_of_variable_id:
                #         sub_token_ptr += 1
                #     sub_token_ptr += 1  # point to first sub-token of next variable
                #     sub_token_end = sub_token_ptr
                #
                #     var_name_token_ids = top_hyp.variable_list[sub_token_begin: sub_token_end]  # include ending </s>
                #     if var_name_token_ids == [same_variable_id, end_of_variable_id]:
                #         new_var_name = old_name
                #     else:
                #         new_var_name = vocab.target.subtoken_model.decode_ids(var_name_token_ids)
                #
                #     variable_rename_result[old_name] = {'new_name': new_var_name,
                #                                         'prob': top_hyp.score}

                for var_id, old_name in enumerate(ast.variables):
                    var_name_token_ids = top_hyp.variable_list[var_id]
                    if var_name_token_ids == [same_variable_id, end_of_variable_id]:
                        new_var_name = old_name
                    else:
                        new_var_name = vocab.target.subtoken_model.decode_ids(var_name_token_ids)

                    variable_rename_result[old_name] = {'new_name': new_var_name, 'prob': top_hyp.score}

            variable_rename_results.append(variable_rename_result)

        return variable_rename_results
