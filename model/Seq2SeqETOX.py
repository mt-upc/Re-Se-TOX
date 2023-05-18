import numpy as np
from torch import nn
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, NllbTokenizer
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch
import sys
import time
from .ETOX import *
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
    
def etox_single_modified(df_Eval, toxicity_list, token_level, lang_code=None, tokenizer=None):
    if not (len(df_Eval) > 0):
        print("ERROR, empty input table")
        return None

    # clean up the strings before toxicity check:
    # lowercases everying, removing punctuation to spaces,
    clean_colname = "string_clean"
    df_Eval = txt_format(df_Eval, col_name_in="string_raw", col_name=clean_colname)
    df_Eval.loc[:, ["token_level"]] = token_level
    toxicity_list = [x.lower() for x in toxicity_list]

    ## Do the actual checks for toxic words in the translation strings
    #print(
    #    f"checking for matches in {lang_code} strings.  May take a minute for large datasets"
    #)

    # uses a different tokenizer depending on input parameter
    if token_level == "space":
        df_Eval.loc[:, ["matched_toxicity_list"]] = df_Eval[clean_colname].apply(
            token_checker, toxic_word_list=toxicity_list
        )
    elif token_level == "character":
        df_Eval.loc[:, ["matched_toxicity_list"]] = df_Eval[clean_colname].apply(
            substring_checker, toxic_word_list=toxicity_list
        )

    elif token_level == "SPM":
        spm_toxicity_list = [sp.encode_as_pieces(x.lower()) for x in toxicity_list]
        df_Eval.loc[:, ["matched_toxicity_list"]] = df_Eval[clean_colname].apply(
            SPM_token_checker, spm_tokenized_toxic_word_list=spm_toxicity_list
        )

    elif token_level == "custom":
        df_Eval.loc[:, ["matched_toxicity_list"]] = df_Eval[clean_colname].apply(
            tokenizer, toxic_word_list=toxicity_list
        )

    else:
        raise Exception(
            "NO TOKENIZATION SPECIFIED, must be 'space', 'character', 'SPM', or 'custom'"
        )

    df_Eval.loc[:, ["matched_toxicity_string"]] = df_Eval[
        "matched_toxicity_list"
    ].apply(lambda x: "|".join(x))
    df_Eval.loc[:, ["toxic_phrase_count"]] = df_Eval["matched_toxicity_list"].apply(len)

    # subset of data with toxic words
    df_Eval_matched_only = df_Eval[df_Eval["toxic_phrase_count"] > 0]

    # toxic word frequency dict
    matched_phrases_list_of_lists = df_Eval_matched_only[
        "matched_toxicity_list"
    ].to_list()
    matched_phrases = []
    for tox_list in matched_phrases_list_of_lists:
        matched_phrases += tox_list
    matched_phrase_count = Counter(matched_phrases)
    matched_phrase_count_dict = dict(matched_phrase_count)

    # toxic phrase count
    matched_phrase_count = df_Eval.toxic_phrase_count.sum()

    # toxic line count
    matched_line_count = df_Eval_matched_only.shape[0]

    # toxic word percentage
    # % of lines with any toxicity
    matched_percentage = df_Eval_matched_only.shape[0] / df_Eval.shape[0]

    return (
        df_Eval,
        df_Eval_matched_only,
        matched_phrase_count_dict,
        matched_phrase_count,
        matched_line_count,
        matched_percentage,
    )

def add_context(x, y):
    return (x[0] + y[0], x[1] + y[1])

def add_context_self_cross(x, y):
    return (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3])

class TextGeneratorSeq2Seq:
    def __init__(self,
                toxicity_filename,
                seed=0,
                seq2seq_model='m2m1B',
                target_seq_length=15,
                reset_context_delta=True,
                num_iterations=5,
                quality_scale=1.,
                stepsize=0.3,
                grad_norm_factor=0.9,
                repetition_penalty=1.,
                end_factor=1.01,
                top_size = 50,
                attention_change = 'self_attention_decoder', # 'self_attention_decoder', 'cross_attention', 'self_cross_attention'
                src_lang = 'en',
                tgt_lang = 'fr',
                unmodified = False,
                update_when_toxic = False,
                toxicity_method = 'ETOX',
                **kwargs):

        # define source and target language
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        # set Random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seq2seq_model_name = seq2seq_model
        if seq2seq_model == "m2m1B":
            self.seq2seq_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
            self.seq2seq_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B", output_hidden_states=True)
            self.seq2seq_tokenizer.src_lang  = self.src_lang

        elif seq2seq_model == "m2m418M":
            self.seq2seq_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
            self.seq2seq_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", output_hidden_states=True)
            self.seq2seq_tokenizer.src_lang  = self.src_lang

        elif seq2seq_model == "nllb600M":
            self.seq2seq_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=self.src_lang, tgt_lang=self.tgt_lang)
            self.seq2seq_model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M", output_hidden_states=True)

        self.seq2seq_model.to(self.device)
        self.seq2seq_model.eval()

        # Freeze weights
        for param in self.seq2seq_model.parameters():
            param.requires_grad = False

        # Init arguments
        self.target_seq_length = target_seq_length
        self.reset_context_delta = reset_context_delta
        self.num_iterations = num_iterations
        self.quality_scale = quality_scale
        self.stepsize = stepsize
        self.grad_norm_factor = grad_norm_factor
        self.repetition_penalty = repetition_penalty
        #self.end_token = self.seq2seq_tokenizer.encode(end_token)[0]
        self.end_token = 2
        self.end_factor = end_factor
        self.ef_idx = 1
        self.top_size = top_size
        self.attention_change = attention_change
        self.unmodified = unmodified
        self.update_when_toxic = update_when_toxic
        #toxicity list filename
        self.toxicity_list = import_toxicity_list_file(toxicity_filename, verbose=False)
        self.toxicity_method = toxicity_method

        if self.toxicity_method == 'detoxify':
            # Initialize detoxify
            DOWNLOAD_URL = "https://github.com/unitaryai/detoxify/releases/download/"
            multilingual = DOWNLOAD_URL + "v0.4-alpha/multilingual_debiased-0b549669.ckpt"
            class_names = ['toxicity', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
            loaded = torch.hub.load_state_dict_from_url(multilingual, map_location=self.device)
            self.detoxify_model = XLMRobertaForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path=None,
                    config='xlm-roberta-base',
                    num_labels=16,
                    state_dict=loaded["state_dict"],
                    local_files_only=False
            )
            self.detoxify_model.to(device=self.device)
            self.detoxify_tokenizer = XLMRobertaTokenizer.from_pretrained(
                                                                        'xlm-roberta-base',
                                                                        local_files_only=False
                                                                        )
            self.threshold_detoxify = 0.5

    def get_decoder_input_ids(self, context_tokens, i):

        if self.seq2seq_model_name in ['m2m1B', 'm2m418M']:
            bos = ( torch.ones((context_tokens.shape[0], 1))*self.seq2seq_tokenizer.get_lang_id(self.tgt_lang) ).to(torch.int).to(self.device)
        elif self.seq2seq_model_name in ['nllb600M']:
            bos = (torch.ones((context_tokens.shape[0], 1))*self.seq2seq_tokenizer.lang_code_to_id[self.tgt_lang]).to(torch.int).to(self.device)

        eos = ( torch.ones((context_tokens.shape[0], 1))*self.end_token ).to(torch.int).to(self.device)
        if i == 0:
            bos = torch.hstack(( eos, bos ))
        else:
            bos = torch.hstack(( eos, bos, context_tokens[:, -i:] ))
        
        return bos

    def get_quality_scores(self, text):
        if self.toxicity_method == 'ETOX':
            text_df = pd.DataFrame(text, columns = ['string_raw'])
            text_df.index.name = 'Dataset_ID'
            init_time = time.time()
            etox_output = etox_single_modified(text_df, self.toxicity_list, token_level = "space") #etox_single(text_df, token_level = "space")
            df_Eval, _, _, _, _, _ = etox_output
            scores_raw = torch.from_numpy(df_Eval['toxic_phrase_count'].values).to(torch.float)
            toxicity_bool = 1 in scores_raw
            scores = 1 - torch.softmax(scores_raw, dim=-1)

        elif self.toxicity_method == 'detoxify':
            inputs = self.detoxify_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            out = self.detoxify_model(**inputs)[0]
            scores = 1 - torch.sigmoid(out[:,0])
            toxicity_bool = any([ i >= self.threshold_detoxify for i in out[:,0] ])

        return scores, toxicity_bool

    def run(self, cond_text, beam_size):

        init_time = time.time()
        self.cond_text = cond_text
        context_tokens = self.seq2seq_tokenizer(cond_text, return_tensors = 'pt').input_ids[0]
        output_tokens, output_text = self.generate_text(context_tokens, beam_size)
        
        print("Time: ", time.time() - init_time)
        return output_text[0].replace('</s>', '')

    def generate_text(self, context_tokens, beam_size):
        context_tokens = torch.tensor(context_tokens, device=self.device, dtype=torch.long).unsqueeze(0)

        gen_tokens = None
        scores = None
        seq_lengths = torch.ones(beam_size, device=self.device)
        is_stopped = torch.zeros(beam_size, device=self.device, dtype=torch.bool)

        for i in range(self.target_seq_length):
            probs = self.get_next_probs(i, context_tokens)
            logits = probs.log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                context_tokens = context_tokens.expand(beam_size, *context_tokens.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)

                if gen_tokens is None:
                    gen_tokens = next_tokens
                else:
                    gen_tokens = gen_tokens.expand(beam_size, *gen_tokens.shape[1:])
                    gen_tokens = torch.cat((gen_tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                gen_tokens = gen_tokens[next_tokens_source]
                gen_tokens = torch.cat((gen_tokens, next_tokens), dim=-1)
                context_tokens = context_tokens[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            context_tokens = torch.cat((context_tokens, next_tokens), dim=1)
            is_stopped = is_stopped + next_tokens.eq(self.end_token).squeeze()
            
            ####
            tmp_scores = scores / seq_lengths
            tmp_output_list = gen_tokens.cpu().numpy()
            tmp_output_texts = [
                self.seq2seq_tokenizer.decode(tmp_output)
                for tmp_output, tmp_length in zip(tmp_output_list, seq_lengths)
            ]
            tmp_order = tmp_scores.argsort(descending=True)
            tmp_output_texts = [tmp_output_texts[i] + ' %% ' + str(tmp_scores[i].cpu().numpy()) for i in tmp_order]
            ##log_info(tmp_output_texts, verbose=True)
            ####

            if is_stopped.all():
                break

        scores = scores / seq_lengths
        output_list = gen_tokens.cpu().numpy()
        output_texts = [
            self.seq2seq_tokenizer.decode(output[: int(length)])
            for output, length in zip(output_list, seq_lengths)
        ]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]

        return context_tokens, output_texts

    def get_next_probs(self, i, context_tokens):

        bos = self.get_decoder_input_ids(context_tokens, i)
        last_token = context_tokens[:, -1:]
        decoder_input_ids = context_tokens
        
        input_tokens = lambda i,x: x if i == 0 else x[:,:-i] 
        
        if self.reset_context_delta and context_tokens.size(1) > 1:
            context = self.seq2seq_model(input_tokens(i, context_tokens), decoder_input_ids = bos)["past_key_values"]
            old_context = context
            if self.attention_change == 'self_attention_decoder':
                context = [ p[:2] for p in context] # get only self attention k v
            elif self.attention_change == 'cross_attention':
                context = [ p[2:] for p in context]
            # by default self_cross_attention
        
        # Logits of Seq2Seq with unshifted context
        logits_before_shift = self.seq2seq_model(input_tokens(i, context_tokens), decoder_input_ids = bos )["logits"]
        logits_before_shift = logits_before_shift[:, -1, :]
        probs_before_shift = nn.functional.softmax(logits_before_shift, dim=-1)
        if self.unmodified:
            return self.update_special_tokens_logits(context_tokens, i, probs_before_shift)
        
        if context:
            context, toxicity_bool = self.shift_context(i, context, last_token, context_tokens, probs_before_shift, old_context)

        if toxicity_bool is False and self.update_when_toxic is True:
            return self.update_special_tokens_logits(context_tokens, i, probs_before_shift)

        lm_output = self.seq2seq_model(last_token, past_key_values = context, decoder_input_ids = bos)
        logits, past = (
            lm_output["logits"],
            lm_output["past_key_values"],
        )
        logits = logits[:, -1, :]

        #logits = self.update_special_tokens_logits(context_tokens, i, logits)

        probs = nn.functional.softmax(logits, dim=-1)
        probs = probs / probs.sum()

        return probs

    def shift_context(self, i, context, last_token, context_tokens, probs_before_shift, old_context):
        context_delta = [tuple([np.zeros(x.shape).astype("float32") for x in p]) for p in context]
        bos = self.get_decoder_input_ids(context_tokens, i)
        window_mask = torch.ones_like(context[0][0]).to(self.device)
        if self.attention_change == 'self_cross_attention':
            window_mask_cross_attention = torch.ones_like(context[0][3]).to(self.device)
        
        iter_beam = i
        for i in range(self.num_iterations):
            curr_shift = [tuple([torch.from_numpy(x).requires_grad_(True).to(device=self.device) for x in p_]) for p_ in
                          context_delta]
            
            if self.attention_change == 'self_cross_attention':
                for p0, p1, p2, p3 in curr_shift:
                    p0.retain_grad()
                    p1.retain_grad()
                    p2.retain_grad()
                    p3.retain_grad()
                shifted_context = list(map(add_context_self_cross, context, curr_shift))
            else:
                for p0, p1 in curr_shift:
                    p0.retain_grad()
                    p1.retain_grad()
                shifted_context = list(map(add_context, context, curr_shift))

            if self.attention_change == 'self_attention_decoder':
                # add unchanged cross attention
                past_key_values_modified = [tuple([p[0], p[1], p2[2], p2[3]]) for p, p2 in zip(shifted_context, old_context)]
            elif self.attention_change == 'cross_attention':
                # add unchanged self attention
                past_key_values_modified = [tuple([p2[0], p2[1], p[0], p[1]]) for p, p2 in zip(shifted_context, old_context)]
            elif self.attention_change == 'self_cross_attention':
                past_key_values_modified = shifted_context

            shifted_outputs = self.seq2seq_model(last_token, past_key_values=past_key_values_modified, decoder_input_ids = bos )
            logits = shifted_outputs["logits"][:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)

            loss = 0.0

            # quality LOSS
            quality_loss, quality_losses, toxicity_bool = self.quality_loss(probs, context_tokens, iter_beam)
            if toxicity_bool is False and self.update_when_toxic is True: # if no toxicity is detected and update_when_toxic = True; we do not update
                return context, toxicity_bool

            loss += self.quality_scale * quality_loss

            # CE/Fluency loss
            ce_loss = (1 - self.quality_scale ) * ((probs * probs.log()) - (probs * probs_before_shift.log())).sum(-1)
            loss += ce_loss.sum()

            loss.backward()

            # ---------- Weights ----------
            combined_scores_k = -(ce_loss)
            combined_scores_c = -(self.quality_scale * torch.stack(quality_losses))

            # minmax
            if combined_scores_k.shape[0] == 1:
                tmp_weights_c = tmp_weights_k = torch.ones(*combined_scores_k.shape).to(self.device)
            else:
                tmp_weights_k = ((combined_scores_k - combined_scores_k.min())) / (
                        combined_scores_k.max() - combined_scores_k.min())
                tmp_weights_c = ((combined_scores_c - combined_scores_c.min())) / (
                        combined_scores_c.max() - combined_scores_c.min())

            tmp_weights = 0.5 * tmp_weights_k + 0.5 * tmp_weights_c
            tmp_weights = tmp_weights.view(tmp_weights.shape[0], 1, 1, 1)

            factor = 1

            # --------- Specific Gen ---------
            sep_grads = None

            for b in range(context_tokens.shape[0]):
                
                tmp_sep_norms = []
                for p_ in curr_shift:
                    tmp_sep_norms_aux = []
                    for j, x in enumerate(p_):
                        if j < 2:
                            norm = torch.norm(x.grad[b:(b + 1)] * window_mask[b:(b + 1)]) + 1e-15
                        else:
                            norm = torch.norm(x.grad[b:(b + 1)] * window_mask_cross_attention[b:(b + 1)]) + 1e-15
                        tmp_sep_norms_aux.append( norm )
                    tmp_sep_norms.append(tmp_sep_norms_aux)
                        
                #tmp_sep_norms = [[(torch.norm(x.grad[b:(b + 1)] * window_mask[b:(b + 1)]) + 1e-15) for x in p_] for p_ in curr_shift]

                # normalize gradients
                #tmp_grad = [tuple([-self.stepsize * factor * (
                #        x.grad[b:(b + 1)] * window_mask[b:(b + 1)] / tmp_sep_norms[i][
                #    j] ** self.grad_norm_factor).data.cpu().numpy()
                #                   for j, x in enumerate(p_)])
                #            for i, p_ in enumerate(curr_shift)]


                tmp_grad = []
                for i, p_ in enumerate(curr_shift):
                    tmp_grad_aux = []
                    for j, x in enumerate(p_):
                        if j < 2:
                            norm_grad = -self.stepsize * factor * (x.grad[b:(b + 1)] * window_mask[b:(b + 1)] / tmp_sep_norms[i][j] ** self.grad_norm_factor).data.cpu().numpy()
                        else:
                            norm_grad = -self.stepsize * factor * (x.grad[b:(b + 1)] * window_mask_cross_attention[b:(b + 1)] / tmp_sep_norms[i][j] ** self.grad_norm_factor).data.cpu().numpy()
                        tmp_grad_aux.append( norm_grad )
                    tmp_grad_aux = tuple(tmp_grad_aux)
                    tmp_grad.append(tmp_grad_aux)

                if sep_grads is None:
                    sep_grads = tmp_grad
                else:
                    for l_index in range(len(sep_grads)):
                        sep_grads[l_index] = list(sep_grads[l_index])
                        for k_index in range(len(sep_grads[0])):
                            sep_grads[l_index][k_index] = np.concatenate(
                                (sep_grads[l_index][k_index], tmp_grad[l_index][k_index]), axis=0)
                        sep_grads[l_index] = tuple(sep_grads[l_index])
            final_grads = sep_grads

            # --------- update context ---------

            if self.attention_change == 'self_cross_attention':
                context_delta = list(map(add_context_self_cross, final_grads, context_delta))
            else:
                context_delta = list(map(add_context, final_grads, context_delta))

            if self.attention_change == 'self_cross_attention':
                for p0, p1, p2, p3 in curr_shift:
                    p0.grad.data.zero_()
                    p1.grad.data.zero_()
                    p2.grad.data.zero_()
                    p3.grad.data.zero_()

            else:
                for p0, p1 in curr_shift:
                    p0.grad.data.zero_()
                    p1.grad.data.zero_()

            new_context = []

            if self.attention_change == 'self_cross_attention':
                for p0, p1, p2, p3 in context:
                    new_context.append((p0.detach(), p1.detach(), p2.detach(), p3.detach()))
            else:
                for p0, p1 in context:
                    new_context.append((p0.detach(), p1.detach()))

            context = new_context

        context_delta = [tuple([torch.from_numpy(x).requires_grad_(True).to(device=self.device) for x in p_])
                         for p_ in context_delta]

        if self.attention_change == 'self_cross_attention':
            context = list(map(add_context_self_cross, context, context_delta))
        else:
            context = list(map(add_context, context, context_delta))

        new_context = []

        if self.attention_change == 'self_cross_attention':
            for p0, p1, p2, p3 in context:
                new_context.append((p0.detach(), p1.detach(), p2.detach(), p3.detach() ))
        else:
            for p0, p1 in context:
                new_context.append((p0.detach(), p1.detach()))
        context = new_context

        if self.attention_change == 'self_attention_decoder':
            # add unchanged cross attention
            past_key_values_modified = [tuple([p[0], p[1], p2[2], p2[3]]) for p, p2 in zip(context, old_context)]
        elif self.attention_change == 'cross_attention':
            # add unchanged self attention
            past_key_values_modified = [tuple([p2[0], p2[1], p[0], p[1]]) for p, p2 in zip(context, old_context)]
        elif self.attention_change == 'self_cross_attention':
            past_key_values_modified = context

        return past_key_values_modified, toxicity_bool

    def update_special_tokens_logits(self, context_tokens, i, logits):
        for beam_id in range(context_tokens.shape[0]):
            for token_idx in set(context_tokens[beam_id][-4:].tolist()):
                factor = self.repetition_penalty if logits[beam_id, token_idx] > 0 else (1 / self.repetition_penalty)
                logits[beam_id, token_idx] /= factor

            if i >= self.ef_idx:
                factor = self.end_factor if logits[beam_id, self.end_token] > 0 else (1 / self.end_factor)
                logits[beam_id, self.end_token] *= factor
            if i == 0:
                start_factor = 1.6
                factor = start_factor if logits[beam_id, self.end_token] > 0 else (1 / start_factor)
                logits[beam_id, self.end_token] /= factor

            #for token_idx in list(self.forbidden_tokens):
            #    factor = self.forbidden_factor if logits[beam_id, token_idx] > 0 else (1 / self.forbidden_factor)
            #    logits[beam_id, token_idx] /= factor

        return logits

    def quality_loss(self, probs, context_tokens, i):

        top_size = self.top_size ## numero de tokens de los que calcular el valor de toxicidad
        _, top_indices = probs.topk(top_size, -1)

        prefix_texts = [ self.seq2seq_tokenizer.decode(x) for x in context_tokens[:, -(i+1):-1] ]

        batch_texts = []
        for idx_p in range(probs.shape[0]):
            top_texts = []
            prefix_text = prefix_texts[idx_p]
            for x in top_indices[idx_p]:
                batch_texts.append( prefix_text.replace('eng_Latn', '') + ' ' + self.seq2seq_tokenizer.decode(x))
        quality_scores, toxicity_bool  = self.get_quality_scores(batch_texts)

        quality_loss = 0
        losses = []
        for idx_p in range(probs.shape[0]):
            top_size = top_indices[idx_p].shape[0]
            with torch.no_grad():
                target_probs = nn.functional.softmax(quality_scores[top_size*idx_p:(top_size*idx_p)+top_size], dim=-1).detach()
                target_probs = target_probs.type(torch.float32)

            target = torch.zeros_like(probs[idx_p])
            target[top_indices[idx_p]] = target_probs[0]
            target = target.unsqueeze(0)
            cur_quality_loss = torch.sum(-(target * torch.log(probs[idx_p:(idx_p + 1)])))

            quality_loss += cur_quality_loss
            losses.append(cur_quality_loss)

        return quality_loss, losses, toxicity_bool