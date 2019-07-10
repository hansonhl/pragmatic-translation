import sys, os
import torch
from collections import defaultdict as DefDict

DEFAULT_CONFIG_FILES = ['summary_inference2.yml']

class Summary:

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
        self.curr_seq = ''

    def forward(self, sequence, source_sentence, debug=False):
        import onmt.inputters as inputters
        T = self.translator

        if sequence != '' and sequence.startswith(self.curr_seq):
            curr_in = sequence[len(self.curr_seq):].strip()
            self.curr_seq = sequence
        else:
            # NOTE: assume a new sequence must start with ''
            self.step = 0
            self.curr_seq = ''

            # setup for new sequence:
            self.raw_src = [source_sentence]
            self.data = inputters.Dataset(
                T.fields,
                readers=([T.src_reader]),
                data=[("src", self.raw_src)],
                dirs=[None],
                sort_key=inputters.str2sortkey[T.data_type],
                filter_pred=T._filter_pred
            )
            self.data_iter = inputters.OrderedIterator(
                dataset=self.data,
                device=T._dev,
                batch_size=self.batch_size,
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
            print('Successfully initialized')

        log_probs, attn = self.translate_step(curr_in)

        # The following assumes only one sentence in batch:
        log_probs = log_probs.squeeze()
        log_probs_shape = log_probs.shape[0]

        assert len(self.full_itos) == log_probs_shape, \
            'len(tgt_dict) (={}) != len(log_probs) (={})'.format( \
                len(self.full_itos), log_probs_shape)

        return log_probs, self.full_itos

    def translate_step(self, curr_in):
        with torch.no_grad():
            if self.step == 0:
                for batch in self.data_iter:
                    self.batch = batch
                    self.src, self.enc_states, self.memory_bank, self.src_lengths =\
                        self.translator._run_encoder(batch)
                    self.translator.model.decoder.init_state(
                        self.src, self.memory_bank, self.enc_states)

            decoder_input= torch.tensor(self.full_stoi[curr_in], \
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
            return log_probs, attn

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
