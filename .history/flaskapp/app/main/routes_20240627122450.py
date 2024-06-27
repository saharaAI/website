from flask import Blueprint, request, jsonify, current_app
from marshmallow import ValidationError

from app.main.schemas import PromptRequestSchema, GenerateRequestSchema
from app.models.credit_risk_analyzer import CreditRiskAnalyzer
from app.main.utils import allowed_file

main_bp = Blueprint('main', __name__)
analyzer = CreditRiskAnalyzer() # Initialize the analyzer

@main_bp.route('/', methods=['GET'])
def read_root():
    """Root endpoint."""
    return jsonify({"message": "Welcome to Sahara Analytics API!"}), 200

@main_bp.route('/prompts', methods=['GET'])
def get_prompts():
    """Returns a list of available prompts."""
    return jsonify({"prompts": analyzer.prompts}), 200

@main_bp.route('/analyze', methods=['POST'])
def analyze_risk():
    """Endpoint to perform credit risk analysis."""
    try:
        data = request.get_json()
        generate_request = GenerateRequestSchema().load(data)

        prompt = analyzer.get_prompt(generate_request['prompt'])
        response = analyzer.analyze(generate_request['message'], prompt)
        return jsonify({"analysis": response}), 200

    except ValidationError as err:
        return jsonify({"error": err.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Error in analyze_risk: {str(e)}")
        return jsonify({"error": "An error occurred during analysis"}), 500

@main_bp.route('/upload_file/', methods=['POST'])
def upload_and_analyze():
    """Endpoint to upload a file and analyze its content."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # Save the uploaded file
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Process the uploaded file
            file_content = analyzer.process_uploaded_file(file_path)

            # Get prompt number from request arguments or default to 0
            prompt_number = request.args.get('prompt_number', 0, type=int)

            generate_request = {
                "message": "Analyze this information.",
                "prompt": {"number": prompt_number, "context": file_content}
            }
            analysis_result = analyze_risk(generate_request)  # Re-use analyze_risk

            return analysis_result 

        else:
            return jsonify({"error": "File type not allowed"}), 400
    
    except Exception as e:
        current_app.logger.error(f"Error in upload_and_analyze: {str(e)}")
        return jsonify({"error": "An error occurred during file processing"}), 500