import sys, os
import torch
from collections import defaultdict as DefDict

DEFAULT_CONFIG_FILES = ['summary_inference2.yml']

class NaiveSummary:
    def __init__(self, onmt_path):
        # managing package import for OpenNMT
        sys.path.insert(0, os.path.abspath(onmt_path)) # ../myOpenNMT
        # https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath

        print('Configuring summary model...')
        import onmt
        from onmt.utils.parse import ArgumentParser
        import onmt.opts as opts
        from onmt.translate.translator import build_translator
        import onmt.inputters as inputters

        # use configs for summary in opennmt
        # using main_summary.py -config <config_file> will override default config
        parser = ArgumentParser(default_config_files=DEFAULT_CONFIG_FILES)
        opts.translate_opts(parser)
        self.opt = parser.parse_args()
        ArgumentParser.validate_translate_opts(self.opt)

        self.translator = build_translator(self.opt, report_score=True)
        print('Finished configuration.\n')

        self.translator.beam_size = 1
        self.seg_type = 'word'

        # Get vocab
        # tgt_field object is a torchtext.data.field.Field object，
        # it stores mappings between index and strings and indices of special tags
        self.tgt_field = dict(self.translator.fields)["tgt"].base_field
        # self.tgt_field.vocab.itos has length 50004, does not contain extended vocab

        self.eos_token = self.tgt_field.eos_token
        self.base_dict = self.tgt_field.vocab.itos
        self.tgt_dict = self.base_dict
        self.curr_seq = ''

    def forward(self, sequence, source_sentence, debug=False):
        #### DEPRECATED: need to treat sequence as a list of subwords
        import onmt.inputters as inputters

        src = [source_sentence]
        seq = [sequence]
        batch_size = 1 # for now
        T = self.translator

        # try to include sequence as some kind of target built into the batch
        data = inputters.Dataset(
            T.fields,
            readers=([T.src_reader, T.tgt_reader]),
            data=[("src", src), ("tgt", seq)],
            dirs=[None, None],
            sort_key=inputters.str2sortkey[T.data_type],
            filter_pred=T._filter_pred
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=T._dev,
            batch_size=batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        # these might be useful if the input sequence was a batch
        # all_probs = []
        # all_attn = []
        # all_support = []

        # imitate translate() method of Translator object in OpenNMT
        for batch in data_iter:
            log_probs, attn = self._translate_batch(batch, data.src_vocabs)

            # The following assuming only one sentence:
            # log_probs = log_probs[-1, 0, :]
            log_probs = log_probs.squeeze()
            if len(log_probs.shape) == 2:
                log_probs = log_probs[-1]
            log_probs_shape = log_probs.shape[0]

            # setup extended dictionary
            ext_dict = data.src_vocabs[0].itos
            # check if tgt_dict is extended by comparing lengths
            if len(self.tgt_dict) != len(self.base_dict) + len(ext_dict):
                self.tgt_dict = self.base_dict + ext_dict

            assert len(self.tgt_dict) == log_probs_shape, \
                'len(tgt_dict) (={}) != len(log_probs) (={})'.format( \
                    len(self.tgt_dict), log_probs_shape)

            return log_probs, [self.tgt_dict[x] for x \
                in range(log_probs_shape)] #, dec_out, dec_attn

            # all_probs.append(log_probs)
            # all_attn.append(attn)
        # return torch.cat(all_probs), None

    def _translate_batch(self, batch, src_vocabs):
        # potential optimization: consider situation where forward() is called
        # multiple times on the same sequence

        with torch.no_grad():

            src, enc_states, memory_bank, src_lengths = \
                self.translator._run_encoder(batch)
            # print('src.shape:', src.shape)
            # print('enc_states.shape:', enc_states.shape)
            # print('memory_bank.shape:', memory_bank.shape)
            # print('src_lengths:', src_lengths)

            self.translator.model.decoder.init_state(src, memory_bank, enc_states)
            # all working up to this point

            decoder_input = batch.tgt[:-1]

            return self.translator._decode_and_generate(
                decoder_input, #ok
                memory_bank, #ok
                batch,      #ok
                src_vocabs, #ok
                memory_lengths=src_lengths, #ok
                src_map=batch.src_map, #ok
                step=None, # not sure, but working
                batch_offset=None, # not sure, but working?
                verbose=True
            )

    # def optimized_translate_batch(self, batch, src_vocabs, in):

class OptimizedSummary:

    def __init__(self, onmt_path):
        # managing package import for OpenNMT
        sys.path.insert(0, os.path.abspath(onmt_path)) # ../myOpenNMT
        # https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath

        print('Configuring summary model...')
        import onmt
        from onmt.utils.parse import ArgumentParser
        import onmt.opts as opts
        from onmt.translate.translator import build_translator
        import onmt.inputters as inputters

        # use configs for summary in opennmt
        # using main_summary.py -config <config_file> will override default config
        parser = ArgumentParser(default_config_files=DEFAULT_CONFIG_FILES)
        opts.translate_opts(parser)
        self.opt = parser.parse_args()
        ArgumentParser.validate_translate_opts(self.opt)

        self.translator = build_translator(self.opt, report_score=True)
        print('Finished configuration.\n')

        self.translator.beam_size = 1
        self.batch_size = 1

        # Get vocab
        # tgt_field object is a torchtext.data.field.Field object，
        # it stores mappings between index and strings and indices of special tags
        self.tgt_field = dict(self.translator.fields)["tgt"].base_field
        # self.tgt_field.vocab.itos has length 50004, does not contain extended vocab
        self.eos_token = self.tgt_field.eos_token
        self.base_itos = self.tgt_field.vocab.itos
        self.base_stoi = self.tgt_field.vocab.stoi
        self.curr_seq = None

        # the following are required by the interface:
        self.bpe_code_path = None
        self.seg_type = 'word'
        self.cache = {}

    def memoize_forward(f):
        def helper(self,sequence,source_sentence,debug=False):
            # world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
            hashable_args = (tuple(sequence),tuple(source_sentence))
            if hashable_args not in self.cache:
                # print("MEM\n\n")
                self.cache[hashable_args] = f(self, sequence, source_sentence, debug)
            return self.cache[hashable_args]
        return helper

    @memoize_forward
    def forward(self, sequence, source_sentence, debug=False):
        # TODO: memoize sequences that are already read

        # for each growing sequence, the following are invariant:
        # self.data, self.data_iter, self.batch; encoder output: self.src,
        # self.enc_states, self.memory_bank, self.src_lengths;
        # decoder state (possibly try saving entire translator object

        # treat sequence as substring
        import onmt.inputters as inputters
        T = self.translator

        if isinstance(sequence, str):
            sequence = sequence.split()

        if sequence != [] and self.curr_seq is not None and \
            _is_continuation(self.curr_seq, sequence):
            curr_in = sequence[-1]
            self.curr_seq = sequence
        else:
            print('Starting new sequence')
            self.step = 0
            self.curr_seq = sequence

            self._start_from_new_seq = (self.curr_seq != [])

            # setup for new sequence:
            raw_src = [source_sentence]

            # setting up data inputters
            self.data = inputters.Dataset(
                T.fields,
                readers=([T.src_reader]),
                data=[("src", raw_src)],
                dirs=[None],
                sort_key=inputters.str2sortkey[T.data_type],
                filter_pred=T._filter_pred
            )

            self.data_iter = inputters.OrderedIterator(
                dataset=self.data,
                device=T._dev,
                batch_size=self.batch_size, # 1 by default
                train=False,
                sort=False,
                sort_within_batch=True,
                shuffle=False
            )
            # NOTE: assume only one sentence in batch
            ext_itos = self.data.src_vocabs[0].itos

            self.full_itos = self.base_itos + ext_itos
            self.full_stoi = DefDict(int, self.base_stoi)
            for i, wd in enumerate(ext_itos):
                fullidx = len(self.base_itos) + i
                if wd not in self.full_stoi:
                    self.full_stoi[wd] = fullidx
            # ensure that words in self.base_stoi have the same idx in full_stoi
            # if word only appears in ext vocab, use extended id in full_stoi

            curr_in = self.tgt_field.init_token
            if self._start_from_new_seq:
                curr_in = [curr_in] + self.curr_seq
            ## End of configuring a new input

        with torch.no_grad():
            if self.step == 0:
                for batch in self.data_iter:
                    self.batch = batch
                    self.src, self.enc_states, self.memory_bank, self.src_lengths =\
                        self.translator._run_encoder(batch)
                    self.translator.model.decoder.init_state(
                        self.src, self.memory_bank, self.enc_states)

            if not self._start_from_new_seq:
                decoder_input = torch.tensor(self.full_stoi[curr_in], \
                    dtype=torch.long).view(1,1,1)
                log_probs, attn = self.translator._decode_and_generate(
                    decoder_input, #ok
                    self.memory_bank, #ok
                    self.batch,      #ok
                    self.data.src_vocabs, #ok
                    memory_lengths=self.src_lengths, #ok
                    src_map=self.batch.src_map, #ok
                    step=self.step, # need to see
                    batch_offset=None, # not sure, but working
                    verbose=True
                )
                self.step += 1
            else:
                seq = [self.full_stoi[x] for x in curr_in]
                for wd in seq:
                    decoder_input = torch.tensor(wd).view(1,1,1)
                    log_probs, attn = self.translator._decode_and_generate(
                        decoder_input, #ok
                        self.memory_bank, #ok
                        self.batch,      #ok
                        self.data.src_vocabs, #ok
                        memory_lengths=self.src_lengths, #ok
                        src_map=self.batch.src_map, #ok
                        step=self.step, # need to see
                        batch_offset=None, # not sure, but working
                        verbose=True
                    )
                    self.step += 1
                self._start_from_new_seq = False

        # The following assumes only one sentence in batch:
        log_probs = log_probs.squeeze()
        if len(log_probs.shape) == 2:
            log_probs = log_probs[-1]
        log_probs_shape = log_probs.shape[0]

        assert len(self.full_itos) == log_probs_shape, \
            'len(tgt_dict) (={}) != len(log_probs) (={})'.format( \
                len(self.full_itos), log_probs_shape)

        return log_probs, self.full_itos

    def likelihood(self,sequence,source_sentence,target,debug=False):
        probs,support = self.forward(sequence=sequence,source_sentence=source_sentence)
        # print(target,"target word")
        try: out = probs[support.index(target)]
        except:
            print("target word failed:",target)
            raise Exception
        return out

def _is_continuation(l1, l2):
    len1, len2 = len(l1), len(l2)
    if len2 - len1 != 1:
        return False
    minl = min(len(l1), len(l2))
    return all(i == j for i, j in zip(l1[:minl], l2[:minl]))


def greedy_summary(model, sent):
    seq, pred = '', ''

    while pred != model.eos_token:
        log_probs, support = model.forward(seq, sent)
        predid = torch.argmax(log_probs)
        pred = support[predid]
        if pred != model.eos_token:
            if seq == '':
                seq = pred
            else:
                seq = seq + ' ' + pred
    return seq
