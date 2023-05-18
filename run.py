from model.Seq2SeqETOX import TextGeneratorSeq2Seq
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str)
    parser.add_argument('--target_seq_length', type=int, default=100)
    parser.add_argument('--quality_scale', type=float, default=0.7)
    parser.add_argument('--stepsize', type=float, default=0.7)
    parser.add_argument('--top_size', type=int, default=10)
    parser.add_argument('--attention_change', type=str, default='self_attention_decoder')
    parser.add_argument('--src_lang', type=str, default='eng_Latn')
    parser.add_argument('--tgt_lang', type=str, default='fra_Latn')
    parser.add_argument("--unmodified", default=False, type=lambda x: (str(x).lower() == 'true') )
    parser.add_argument("--update_when_toxic", default=False, type=lambda x: (str(x).lower() == 'true') )
    parser.add_argument('--toxicity_method', type=str, default='ETOX')
    parser.add_argument('--beam_size', type=int, default=4)
    args = parser.parse_args()
    return args

def main(grid_args):
    toxicity_filename = './NLLB-200_TWL/{}_twl.txt'.format(grid_args['tgt_lang'])

    seq2seq_model = TextGeneratorSeq2Seq(
                                        toxicity_filename,
                                        seed=0,
                                        seq2seq_model='nllb600M',
                                        target_seq_length=grid_args['target_seq_length'],
                                        num_iterations=1,
                                        quality_scale=grid_args['quality_scale'],
                                        stepsize=grid_args['stepsize'],
                                        grad_norm_factor=0.9,
                                        repetition_penalty=1.,
                                        end_factor=1.01,
                                        top_size = grid_args['top_size'],
                                        attention_change = grid_args['attention_change'],
                                        src_lang = grid_args['src_lang'],
                                        tgt_lang = grid_args['tgt_lang'],
                                        unmodified = grid_args['unmodified'],
                                        update_when_toxic = grid_args['update_when_toxic'],
                                        toxicity_method = grid_args['toxicity_method']
                                        )
    
    trans = seq2seq_model.run( grid_args['text'], grid_args['beam_size'])
    return trans

if __name__ == '__main__':
    args = get_args()

    grid_args = {
                'text':args.text,
                'target_seq_length': args.target_seq_length,
                'quality_scale':args.quality_scale,
                'stepsize':args.stepsize,
                'top_size':args.top_size,
                'attention_change':args.attention_change,
                'src_lang':args.src_lang,
                'tgt_lang':args.tgt_lang,
                'unmodified':args.unmodified,
                'update_when_toxic':args.update_when_toxic,
                'toxicity_method':args.toxicity_method,
                'beam_size':args.beam_size
                }
    
    translation = main(grid_args)
    print('Translated sentence: {}'.format(translation))