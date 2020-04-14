import os
import tensorflow as tf

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import text_encoder

from tensor2tensor.utils import registry
from tensor2tensor.utils import mlperf_log


def txt_line_iterator(txt_path):
  """Iterate through lines of file."""
  with open(txt_path) as f:
    for line in f:
        yield line.strip()


CONFIG = {}


@registry.register_problem
class TranslateManyToMany(translate.TranslateProblem):

    @property
    def use_small_dataset(self):
        return False  #'wtf?'

    @property
    def name(self):
        return "translate_many_to_many"

    @property
    def vocab_filename(self):
        return "vocab.txt"

    def vocab_data_files(self):
        """Files to be passed to get_or_generate_vocab. Skips langpair files."""
        return [[x1, x2[:-1]] for x1, x2 in self.source_data_files(problem.DatasetSplit.TRAIN)]

    @property
    def approx_vocab_size(self):
        return 65536  # 2**16

    def dataset_filename(self):
        return "translate"

    @property
    def additional_reserved_tokens(self):
        return self.prefixes

    @property
    def prefixes(self):
        return ["2<%s>" % lang for lang in CONFIG['languages']]

    @property
    def inputs_prefix(self):
        raise NotImplementedError()  #return "translate English German "

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 50,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]
    
    def source_data_files(self, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            return CONFIG['training_files']
        else:
            return CONFIG['testing_files']
    
    def generate_samples(self, data_dir, tmp_dir, dataset_split, custom_iterator='unused'):
        datasets = self.source_data_files(dataset_split)
        if dataset_split == problem.DatasetSplit.TRAIN:
            tag = "train"
            datatypes_to_clean = self.datatypes_to_clean
        else:
            tag = "dev"
            datatypes_to_clean = None
        
        data_paths = [d[1] for d in datasets]  #self.check_data_files(datasets)
        return self._meta_iterator(data_paths)
    
    # def check_data_files(self, data_dir, datasets):
    #     fnames = []
    #     for dataset in datasets:
    #         unused_url = dataset[0]
    #         lang1_filename, lang2_filename, idx_filename = dataset[1]
    #         lang1_filepath = os.path.join(data_dir, lang1_filename)
    #         lang2_filepath = os.path.join(data_dir, lang2_filename)
    #         if idx_filename is not None:
    #             idx_filepath = os.path.join(data_dir, idx_filename)
    #         else:
    #             idx_filepath = None
    #         fnames.append((lang1_filepath, lang2_filepath, idx_filepath))
    #     return fnames

    def _meta_iterator(self, data_files):
        for source, target, index in data_files:
            print("Processing file", source)
            for example_dict in self.text2text_txt_iterator(source, target, index):
                yield example_dict
    
    def _determine_language_from_suffix(self, filename):
        if filename.endswith(".txt"):
            filename = filename[:-4]
        language = filename.split(".")[-1]
        if language not in CONFIG['languages']:
            raise ValueError(f"The text file {filename} has an unexpected suffix: {language}. Expecting one of the language codes: {CONFIG['languages']}")
        return language

    def text2text_txt_iterator(self, source_txt_path, target_txt_path, index_path=None, bidirectional=True):
        """Yield dicts for Text2TextProblem.generate_samples from lines of files."""       
        if index_path is not None:
            # We're dealing with pre-merged files with multiple languages ...
            #  Each line of index_path contains the languages of the corresponding
            #  line in source_txt_path and target_txt_path.
            for inputs, targets, language_pair in zip(
                txt_line_iterator(source_txt_path), 
                txt_line_iterator(target_txt_path),
                txt_line_iterator(index_path)
            ):
                src_lang, tgt_lang = language_pair.split(" ")
                yield {"inputs": inputs, "targets": targets, "src_lang": src_lang, "tgt_lang": tgt_lang}
                if bidirectional:
                    yield {"inputs": targets, "targets": inputs, "src_lang": tgt_lang, "tgt_lang": src_lang}
        else:
            # We're dealing with a file containing only one language pair ...
            #  We only need to determine the language from the files' suffixes.
            src_lang = self._determine_language_from_suffix(source_txt_path)
            tgt_lang = self._determine_language_from_suffix(target_txt_path)
            for inputs, targets in zip(txt_line_iterator(source_txt_path), txt_line_iterator(target_txt_path)):
                yield {"inputs": inputs, "targets": targets, "src_lang": src_lang, "tgt_lang": tgt_lang}
                if bidirectional:
                    yield {"inputs": targets, "targets": inputs, "src_lang": tgt_lang, "tgt_lang": src_lang}

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
        elif dataset_split == problem.DatasetSplit.EVAL:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        return self.text2text_generate_encoded(generator, encoder,
                                      has_inputs=self.has_inputs,
                                      inputs_prefix=None,
                                      targets_prefix=None)

    def text2text_generate_encoded(self, sample_generator,
                               vocab,
                               targets_vocab=None,
                               has_inputs=True,
                               inputs_prefix=None,
                               targets_prefix=None):
        """Encode Text2Text samples from the generator with the vocab."""
        targets_vocab = targets_vocab or vocab
        for sample in sample_generator:
            if has_inputs:
                sample["inputs"] = vocab.encode(sample["inputs"])
                sample["inputs"].append(text_encoder.EOS_ID)
                sample["inputs"].insert(0, CONFIG['languages'].index(sample["tgt_lang"]) + 2)
            sample["targets"] = targets_vocab.encode(sample["targets"])
            sample["targets"].append(text_encoder.EOS_ID)
            yield sample
