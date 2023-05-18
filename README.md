# ReSeTOX: Re-learning attention weights for toxicity mitigation in machine translation

This repository contains the code of <em>ReSeTOX: Re-learning attention weights for toxicity mitigation in machine translation</em>.

## Abstract

Our proposed method, ReSeTOX (REdo SEarch if TOXic), addresses the issue of Neural Machine Translation (NMT) generating translation outputs that contain toxic words not present in the input. The objective is to mitigate the introduction of toxic language without the need for re-training. In the case of identified added toxicity during the inference process, ReSeTOX dynamically adjusts the key-value self-attention weights and re-evaluates the beam search hypotheses. Experimental results demonstrate that ReSeTOX achieves a remarkable 57% reduction in added toxicity while maintaining an average translation quality of 99.5% across 164 languages.

<br>

![](images_readme/beam_search.png)

## Results

## Usage

To run ReSeTOX given a text you wish to translate:

```bash
python run.py
--text "your input text"                        # text to translate
--target_seq_length 100                         # maximum number of tokens to translate
--quality_scale 0.7                             # alpha hyper-parameter
--stepsize 0.7                                  # step size of the gradient descent update
--top_size 10                                   # number of tokens used to compute the mitigation loss
--attention_change "self_attention_decoder"
--src_lang "eng_Latn"
--tgt_lang "spa_Latn"
--unmodified False
--update_when_toxic True
--toxicity_method "ETOX"
--beam_size 4
```

## Citation

If you want to cite this repository in your work, please consider citing:

```

```
