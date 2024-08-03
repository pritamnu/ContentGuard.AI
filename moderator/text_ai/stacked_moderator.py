from moderator.text_ai.base import BaseModerator
from moderator.text_ai.base import BaseModel

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import BertTokenizer

# helpers
from moderator.text_ai.model_helpers import BERTClassifier


##############################################
# Constants
##############################################
SCORE_THRESHOLD = 0.5

##############################################
# Individual Models
##############################################


class HateModel(BaseModel):
    def __init__(self):
        super(HateModel).__init__()
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
        if result['label'] == 'nothate':
            final['score'] = 1-result['score']
        else:
            final['score'] = result['score']
        return final

    def _load_model(self):
        """
        Load the model for Hate speech detection
        :return:
        """
        # tokenizer = AutoTokenizer.from_pretrained(
        #     "./models/models--facebook--roberta-hate-speech-dynabench-r4-target")
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     "./models/models--facebook--roberta-hate-speech-dynabench-r4-target")
        tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
        model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")

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
        # save the best model path here.
        self.model_path = "models/harassment.pth"
        # load the actual model here
        self.tokenizer = None
        self.model = self._load_model()

    def predict(self, text):
        """
        Load the model for Harassment speech detection
        :param text:
        :return:
        """
        device = "cpu"
        self.model.eval()
        encoding = self.tokenizer(text, return_tensors='pt',
                                  max_length=128,
                                  padding='max_length',
                                  truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_probabilities = probabilities.cpu().numpy().flatten()

        result = {
            "label": "harassment" if predicted_class == 1 else "no_harassment",
            "score": predicted_probabilities[1]
        }
        return result

    def _load_model(self):
        """
        Load the model for Harassment speech detection
        :return:
        """
        device = "cpu"
        bert_model_name = 'bert-base-uncased'
        num_classes = 2
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        model = BERTClassifier(bert_model_name, num_classes).to(device)
        model.load_state_dict(
            torch.load(self.model_path, map_location=device))
        return model
        # return TextClassificationPipeline(model=model, tokenizer=self.tokenizer)


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


class ViolenceModel(BaseModel):
    def __init__(self):
        super(ViolenceModel).__init__()
        # load the actual model here
        self.model = self._load_model()

    def predict(self, text):
        """
        Predict if the text is Hate speech or not
        :param text:
        :return:
        """
        result = self.model(text)[0]
        final = {'label': 'violence'}
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
        # Load model directly
        # Load model directly

        tokenizer = AutoTokenizer.from_pretrained("catalpa-cl/violence-hate-bert-de", cache_dir="models")
        model = AutoModelForSequenceClassification.from_pretrained("catalpa-cl/violence-hate-bert-de")

        # FIXME: Uncomment on a device with Nvidia GPU
        # device = torch.device('cuda')
        # print(device)
        # Load model directly
        # model.to(device)
        # pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)  # return_all_scores=True
        return pipe


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
        "suicide": SuicideModel,
        "violence": ViolenceModel
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
        category = {x['label']: False for x in results}
        scores = {}
        response = {
            "category": category,
            "scores": scores
        }
        # initialize the schema
        for result in temp:
            if temp[0]['score'] > SCORE_THRESHOLD:
                category.update({result['label']: True})
                break
        for result in temp:
            scores.update({result['label']: result['score']})
        # add the normal
        if temp[0]['score'] > SCORE_THRESHOLD:
            category.update({'normal': False})
        else:
            category.update({'normal': True})
        scores.update({'normal': 1-temp[0]['score']})
        return response
