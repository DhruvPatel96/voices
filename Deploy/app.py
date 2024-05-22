from flask import Flask, request, jsonify, send_file, render_template
import os
import sys
import logging

# Add the inference directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'inference'))

from inference import end_to_end_infer, output_directory, sanitize_filename

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error serving index.html: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Text input is required"}), 400

    try:
        app.logger.debug(f"Received text: {text}")
        end_to_end_infer(text, False, False)
        sanitized_text = sanitize_filename(text)
        output_path = os.path.join(output_directory, f"{sanitized_text}.wav")

        app.logger.debug(f"Output path: {output_path}")

        if os.path.exists(output_path):
            return send_file(output_path, as_attachment=True)
        else:
            error_message = "Failed to generate audio, output file not found."
            app.logger.error(error_message)
            return jsonify({"error": error_message}), 500
    except Exception as e:
        error_message = f"Error processing text: {str(e)}"
        app.logger.error(error_message)
        return jsonify({"error": error_message}), 500


if __name__ == '__main__':
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    app.run(host='0.0.0.0', port=5050, debug=True)
