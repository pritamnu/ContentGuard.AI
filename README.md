Steps for deploying the REST API server
========================================

1. git clone this repo
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
