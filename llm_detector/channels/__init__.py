"""Evidence fusion channels."""


class ChannelResult:
    """Result from a single detection channel."""
    __slots__ = ('channel', 'score', 'severity', 'explanation',
                 'mode_eligibility', 'sub_signals')

    SEVERITIES = ('GREEN', 'YELLOW', 'AMBER', 'RED')
    SEV_ORDER = {'GREEN': 0, 'YELLOW': 1, 'AMBER': 2, 'RED': 3}

    def __init__(self, channel, score=0.0, severity='GREEN', explanation='',
                 mode_eligibility=None, sub_signals=None):
        self.channel = channel
        self.score = score
        self.severity = severity
        self.explanation = explanation
        self.mode_eligibility = mode_eligibility or ['task_prompt', 'generic_aigt']
        self.sub_signals = sub_signals or {}

    @property
    def sev_level(self):
        return self.SEV_ORDER.get(self.severity, 0)

    def __repr__(self):
        return f"CH:{self.channel}={self.severity}({self.score:.2f})"


from llm_detector.channels.prompt_structure import score_prompt_structure
from llm_detector.channels.stylometric import score_stylometric
from llm_detector.channels.continuation import score_continuation
from llm_detector.channels.windowed import score_windowed
