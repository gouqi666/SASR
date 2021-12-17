import os
import time

import numpy as np
import tensorflow as tf
from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer

from AMmodel.conformer import ConformerCTC
# from AMmodel.deepspeech2 import DeepSpeech2CTC, DeepSpeech2LAS, DeepSpeech2Transducer
# from AMmodel.MultiConformer import ConformerMultiTaskCTC


class AM():
    def __init__(self, config):
        self.config = config
        self.update_model_type()
        self.speech_config = self.config['speech_config']
        if self.model_type != 'MultiTask':
            self.text_config = self.config['decoder_config']
        else:
            self.text_config = self.config['decoder3_config']
        self.model_config = self.config['model_config']
        self.text_feature = TextFeaturizer(self.text_config, True)
        self.speech_feature = SpeechFeaturizer(self.speech_config)

        self.init_steps = None

    def update_model_type(self):
        if 'Streaming' in self.config['model_config']['name']:
            assert self.config['speech_config']['streaming'] is True
            assert 'Conformer' in self.config['model_config']['name']
        else:
            assert self.config['speech_config']['streaming'] is False
        if 'CTC' in self.config['model_config']['name'] and 'Multi' not in self.config['model_config']['name']:
            self.config['decoder_config'].update({'model_type': 'CTC'})
            self.model_type = 'CTC'
        elif 'Multi' in self.config['model_config']['name']:
            self.config['decoder1_config'].update({'model_type': 'CTC'})
            self.config['decoder2_config'].update({'model_type': 'CTC'})
            self.config['decoder3_config'].update({'model_type': 'CTC'})
            self.config['decoder_config'].update({'model_type': 'CTC'})
            self.model_type = 'MultiTask'
        elif 'LAS' in self.config['model_config']['name']:
            self.config['decoder_config'].update({'model_type': 'LAS'})
            self.model_type = 'LAS'
        else:
            self.config['decoder_config'].update({'model_type': 'Transducer'})
            self.model_type = 'Transducer'

    def conformer_model(self, training):

        self.model_config.update({'vocabulary_size': self.text_feature.num_classes})

        assert self.model_config['name'] == 'ConformerCTC'
        self.model_config.update({'speech_config': self.speech_config})
        self.model = ConformerCTC(**self.model_config)


    # def ds2_model(self, training):
    #
    #     self.model_config['Transducer_decoder']['vocabulary_size'] = self.text_feature.num_classes
    #     f, c = self.speech_feature.compute_feature_dim()
    #     input_shape = [None, f, c]
    #     self.model_config.update({'input_shape': input_shape})
    #     self.model_config.update({'dmodel': self.model_config['rnn_conf']['rnn_units']})
    #
    #     if self.model_config['name'] == 'DeepSpeech2Transducer':
    #         self.model_config.pop('LAS_decoder')
    #         self.model_config.pop('enable_tflite_convertible')
    #         self.model = DeepSpeech2Transducer(input_shape, self.model_config, speech_config=self.speech_config)
    #     elif self.model_config['name'] == 'DeepSpeech2CTC':
    #         self.model = DeepSpeech2CTC(input_shape, self.model_config, self.text_feature.num_classes,
    #                                     speech_config=self.speech_config)
    #     elif self.model_config['name'] == 'DeepSpeech2LAS':
    #         self.model_config['LAS_decoder'].update({'n_classes': self.text_feature.num_classes})
    #         self.model_config['LAS_decoder'].update({'startid': self.text_feature.start})
    #         self.model = DeepSpeech2LAS(self.model_config, input_shape, training=training,
    #                                     enable_tflite_convertible=self.model_config[
    #                                         'enable_tflite_convertible'], speech_config=self.speech_config)
    #     else:
    #         raise ('not in supported model list')

    def multi_task_model(self, training):
        token1_feature = TextFeaturizer(self.config['decoder1_config'])
        token2_feature = TextFeaturizer(self.config['decoder2_config'])
        token3_feature = TextFeaturizer(self.config['decoder3_config'])

        self.model_config.update({
            'classes1': token1_feature.num_classes,
            'classes2': token2_feature.num_classes,
            'classes3': token3_feature.num_classes,
        })

        # self.model = ConformerMultiTaskCTC(self.model_config, training=training,
        #                                    speech_config=self.speech_config)

    def load_model(self, training=True):

        # if 'Multi' in self.model_config['name']:
        #     self.multi_task_model(training)


        if 'Conformer' in self.model_config['name']:
            self.conformer_model(training)
        else:
            # self.ds2_model(training)
            pass
        self.model.add_featurizers(self.text_feature)
        f, c = self.speech_feature.compute_feature_dim()

        if not training:
            if self.text_config['model_type'] != 'LAS':
                if self.model.mel_layer is not None:
                    self.model._build(
                        [3, 16000 if self.speech_config['streaming'] is False else self.model.chunk_size * 2, 1])
                    self.model.return_pb_function([None, None, 1])
                else:
                    self.model._build([3, 80, f, c])
                    self.model.return_pb_function([None, None, f, c])

            self.load_checkpoint(self.config)

    def convert_to_pb(self, export_path):
        concrete_func = self.model.recognize_pb.get_concrete_function()
        tf.saved_model.save(self.model, export_path, signatures=concrete_func)

    def decode_result(self, word):
        de = []
        for i in word:
            if i != self.text_feature.stop:
                de.append(self.text_feature.index_to_token[int(i)])
            else:
                break
        return de

    def  predict(self, fp):
        if '.pcm' in fp:
            data = np.fromfile(fp, 'int16')
            data = np.array(data, 'float32')
            data /= 32768
        else:
            data = self.speech_feature.load_wav(fp)
        if self.model.mel_layer is None:
            mel = self.speech_feature.extract(data)
            mel = np.expand_dims(mel, 0)

            input_length = np.array([[mel.shape[1] // self.model.time_reduction_factor]], 'int32')
        else:
            mel = data.reshape([1, -1, 1])
            input_length = np.array(
                [[mel.shape[1] // self.model.time_reduction_factor // (self.speech_config['sample_rate'] *
                                                                       self.speech_config['stride_ms'] / 1000)]],
                'int32')
        result = self.model.recognize_pb(mel, input_length)[0]

        return result

    def load_checkpoint(self, config):
        """Load checkpoint."""

        self.checkpoint_dir = os.path.join(config['learning_config']['running_config']["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        self.init_steps = int(files[-1].split('_')[-1].replace('.h5', ''))
