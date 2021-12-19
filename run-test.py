from AM.AMmodel.model import AM
from AM.utils.user_config import UserConfig
# from LMmodel.trm_lm import LM
import pypinyin


class ASR:
    def __init__(self, am_config, lm_config, punc_config=None):

        self.am = AM(am_config)
        self.am.load_model(False)

        if punc_config is not None:
            self.punc_recover = True
        else:
            self.punc_recover = False

    def decode_am_result(self, result):
        return self.am.decode_result(result)

    def stt(self, wav_path):

        am_result = self.am.predict(wav_path)

        am_result = self.decode_am_result(am_result[0])

        return am_result

    def am_test(self, wav_path):
        am_result = self.am.predict(wav_path)
        if self.am.model_type == 'Transducer':
            am_result = self.decode_am_result(am_result[1:-1])
        else:
            am_result = self.decode_am_result(am_result[0])
        return am_result


if __name__ == '__main__':
    import time
    am_config = UserConfig(r'AM/conformerCTC(M)/am_data.yml', r'AM/conformerCTC(M)/conformerM.yml')
    lm_config, punc_config = None, None
    asr = ASR(am_config, lm_config, punc_config)
    s = time.time()
    a = asr.stt(r'BAC009S0764W0121.wav')
    e = time.time()
    print(a)
    print('asr.stt first infenrence cost time:', e - s)

