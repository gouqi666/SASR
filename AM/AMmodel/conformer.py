
from AMmodel.ctc_wrap_cfm import CtcModel
from AMmodel.conformer_blocks import ConformerEncoder

class ConformerCTC(CtcModel):
    def __init__(self,
                 dmodel: int,
                 reduction_factor: int,
                 vocabulary_size: int,
                 num_blocks: int = 16,
                 head_size: int = 36,
                 num_heads: int = 4,
                 kernel_size: int = 32,
                 fc_factor: float = 0.5,
                 dropout: float = 0,
                 name='conformerCTC',
                 speech_config=dict,
                 **kwargs):
        super(ConformerCTC, self).__init__(
            encoder=ConformerEncoder(
                dmodel=dmodel,
                reduction_factor=reduction_factor,
                num_blocks=num_blocks,
                head_size=head_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                fc_factor=fc_factor,

                dropout=dropout,
                add_wav_info=speech_config['add_wav_info'],
                hop_size=int(speech_config['stride_ms'] * speech_config['sample_rate']//1000)*reduction_factor,
            ),num_classes=vocabulary_size,name=name,speech_config=speech_config)
        self.time_reduction_factor = reduction_factor

