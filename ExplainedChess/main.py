from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/test', methods=['POST'])
def your_endpoint():
    try:
        # Access request data
        data = request.get_json()

        # Process the data (replace this with your own logic)
        result = process_data(data['user_input'])

        # Return a response
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_data(data):
    # Replace this function with your own data processing logic
    # For example, you can print the data and return a simple message
    print("Received data:", data)
    return data


if __name__ == '__main__':
    app.run(debug=True)
