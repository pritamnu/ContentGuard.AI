from moderator.text_ai.base import BaseModerator
from moderator.text_ai.base import BaseModel

# hate model imports
# import numpy as np
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import mean_squared_error as MSE
# from sklearn.metrics import accuracy_score
# import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline


##############################################
# Constants
##############################################
SCORE_THRESHOLD = 0.5

##############################################
# Individual Models
##############################################


class HateModel(BaseModel):
    def __init__(self):
        super().__init__()
        # load the actual model here
        self.model = self._load_model()

    def predict(self, text):
        """
        Predict if the text is Hate speech or not
        :param text:
        :return:
        """
        result = self.model(text)[0]
        final = {'label': 'hate'}
        if result['label'] == 'LABEL_0':
            final['score'] = 1-result['score']
        else:
            final['score'] = result['score']
        return final

    def _load_model(self):
        """
        Load the model for Hate speech detection
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/english-abusive-MuRIL")
        model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/english-abusive-MuRIL")

        # FIXME: Uncomment on a device with Nvidia GPU
        # device = torch.device('cuda')
        # print(device)
        # Load model directly
        # model.to(device)
        # pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)  # return_all_scores=True
        return pipe


class HarassmentModel(BaseModel):
    def __init__(self):
        super().__init__()
        # load the actual model here
        self.model = None

    def predict(self, text):
        """
        Load the model for Harassment speech detection
        :param text:
        :return:
        """
        return {
            "label": "harassment",
            "score": 0.23
        }


class SuicideModel(BaseModel):
    def __init__(self):
        super().__init__()
        # load the actual model here
        self.model = None

    def predict(self, text):
        """
        Load the model for Suicide speech detection
        :param text:
        :return:
        """
        return {
            "label": "self_harm",
            "score": 0.12
        }


#####################################
# Stacked Model: Integrate individual models
#####################################
class StackedModerator(BaseModerator):
    SCHEMA = {
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
    MODELS = {
        "hate_offense": HateModel,
        "harassment": HarassmentModel,
        "suicide": SuicideModel
    }

    def __init__(self):
        self.models = []
        for model_name, model_cls in self.MODELS.items():
            print(f"Loading Model for {model_name}")
            self.models.append(model_cls())
        print(f"Loaded Models: {self.models}")

    def moderate(self, text):
        """
        Passed the text to each of the models and get individual category scores.
        :param text:
        :return:
        """
        self._pre_process(text)
        results = []
        for model in self.models:
            results.append(model.predict(text))
        print(f"Results: {results}")
        return self._post_process(results)

    def _pre_process(self, text):
        """
        Add cleaning and preprocessing steps on the text
        :return:
        """
        # FIXME: Nothing is done as of now. Need to add data clean-up
        #
        return text

    def _post_process(self, results):
        """
        Calculate the final category based on score and return
        full json response in format SCHEMA
        :param results:
        :return:
        """
        # sort based on scores.
        temp = sorted(results, key=lambda x: x['score'], reverse=True)
        category = {}
        scores = {}
        response = {
            "category": category,
            "scores": scores
        }
        for result in temp:
            if temp[0]['score'] > SCORE_THRESHOLD:
                category.update({result['label']: True})
            else:
                category.update({result['label']: False})
            scores.update({result['label']: result['score']})
        # add the normal
        if temp[0]['score'] > SCORE_THRESHOLD:
            category.update({'normal': False})
        else:
            category.update({'normal': True})
        scores.update({'normal': 1-temp[0]['score']})
        return response
