"""
MS_DSP 498
Author: pritamchatterjee2023@u.northwestern.edu
"""
from moderator.text_ai.stacked_moderator import StackedModerator
from moderator.text_ai.single_moderator import SingleModerator


class TextModerator:
    MODERATOR = {
        "stacked": StackedModerator,
        "single": SingleModerator
    }

    def __new__(self, *args, **kwargs):
        # FIXME, handle incorrect moderator types if passed
        moderator_type = kwargs.pop(" ", "stacked")
        return TextModerator.MODERATOR[moderator_type](*args, **kwargs)


# unit test
if __name__ == "__main__":
    moderator = TextModerator()
    print(moderator.moderate("This not a hate speech"))
