from flask import Flask, request, jsonify
import json

# local libs
from moderator.text_moderator import TextModerator

app = Flask(__name__)


@app.route('/api/v0/moderate', methods=['POST'])
def moderate_text():
    data = request.get_json()
    print(type(data))
    text = data.pop('text')
    moderator = TextModerator(**data)
    response = moderator.moderate(text)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
