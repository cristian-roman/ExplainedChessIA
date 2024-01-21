from flask import Flask, request, jsonify, app
from flask_cors import CORS
from AI.Explainer import Explainer


class Server:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.explainer = Explainer("./AI/Data/chat.json")

        @self.app.route('/test', methods=['POST', 'OPTIONS'])
        def your_endpoint():
            if request.method == 'OPTIONS':
                # Respond to the OPTIONS request with the appropriate headers
                response = jsonify({'message': 'CORS preflight request successful'})
                response.headers.add('Access-Control-Allow-Origin', '*')
                response.headers.add('Access-Control-Allow-Methods', 'POST')
                response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
                return response

            try:
                # Access request data
                data = request.get_json()

                # Process the data (replace this with your own logic)
                result = self.process_data(data['user_input'])

                # Return a response
                return jsonify({'result': result}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def process_data(self, question):

        # check if it starts with 'Avand urmatoarea partida'

        if question.startswith("Avand urmatoarea partida"):
            return self.explainer.get_next_set_of_moves_from_moves(question)
        elif question.startswith("Avand urmatoarea pozitie"):
            return self.explainer.get_next_set_of_moves_from_position(question)

        return self.explainer.explain(question)

    def run(self):
        self.app.run(debug=True)