from flask import Flask, request, jsonify

# local libs
from moderator.text_moderator import TextModerator

app = Flask(__name__)


@app.route('/api/v0/moderate', methods=['POST'])
def moderate_text():
    data = request.get_json()
    text = data.pop('text')
    moderator = TextModerator(**data)
    response = moderator.moderate(text)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
