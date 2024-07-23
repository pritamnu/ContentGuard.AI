"""
MS_DSP 498
Author: pritamchatterjee2023@u.northwestern.edu
"""


class AbstractModerator:
    def moderate(self, text):
        raise NotImplementedError


class AbstractModel:
    def predict(self, text):
        raise NotImplementedError