Steps for deploying the REST API server
========================================

1. git clone this repo: https://github.com/<username>/ContentGuard.AI.git
2. Create the virtual environment
3. Activate the virtual environment
4. Execute: pip install -r requirements.txt [One time only]
5. Execute: python app.py
6. Use postman to run: POST http://127.0.0.1:5000/api/v0/moderate

Body:
---------------------------
{
    "text": "This is not a hate speech"
}

Response:
---------------------------
{
    "category": {
        "harassment": true,
        "hate": true,
        "normal": false,
        "self_harm": true
    },
    "scores": {
        "harassment": 0.23,
        "hate": 0.9511423110961914,
        "normal": 0.048857688903808594,
        "self_harm": 0.12
    }
}
