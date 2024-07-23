"""
MS_DSP 498
Author: pritamchatterjee2023@u.northwestern.edu
"""
from moderator.text_ai.base import BaseModerator


class SingleModerator(BaseModerator):
    DUMMY = {
        "category": {
            "hate": True,
            "harassment": False,
            "self_harm": False,
            "violence": False,
            "normal": False
        },
        "scores": {
            "hate": 0.96,
            "harassment": 0.24,
            "self_harm": 0.04,
            "violence": 0.02,
            "normal": 0.14
        }
    }

    def __init__(self):
        self.model = 'single_moderator'  # should be model object

    def moderate(self, text):
        return self.DUMMY
