# routes/predict.py

from flask import Blueprint, request, jsonify
from backend.models import prediction

predict_bp = Blueprint('predict_bp', __name__)

@predict_bp.route('/current_traffic', methods=['GET'])
def current_traffic():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if lat and lon:
        try:
            lat = float(lat)
            lon = float(lon)
        except ValueError:
            return jsonify({"error": "Invalid latitude or longitude"}), 400

        traffic_info = prediction.get_realtime_traffic_info(lat, lon)
        return jsonify(traffic_info)
    else:
        return jsonify({"error": "Latitude and longitude are required"}), 400

@predict_bp.route('/all_current_traffic', methods=['GET'])
def all_current_traffic():
    try:
        traffic_info = prediction.get_all_realtime_traffic()
        return jsonify(traffic_info)
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@predict_bp.route('/congestion_level', methods=['GET'])
def congestion_level():
    location = request.args.get('location')
    if location:
        try:
            lat, lon = map(float, location.split(','))
        except ValueError:
            return jsonify({"error": "Invalid location format"}), 400
        
        congestion_info = prediction.check_congestion_level((lat, lon))
        return jsonify(congestion_info)
    else:
        return jsonify({"error": "Location is required"}), 400
    
@predict_bp.route('/traffic_prediction', methods=['GET'])
def traffic_prediction():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if lat and lon:
        try:
            lat = float(lat)
            lon = float(lon)
        except ValueError:
            return jsonify({"error": "Invalid latitude or longitude format"}), 400

        prediction_info = prediction.get_prediction_traffic_info(lat, lon)
        return jsonify(prediction_info)
    else:
        return jsonify({"error": "Latitude and longitude are required"}), 400

@predict_bp.route('/alternative_routes', methods=['GET'])
def alternative_routes():
    start = request.args.get('start')
    end = request.args.get('end')
    if start and end:
        try:
            start_lat, start_lon = map(float, start.split(','))
            end_lat, end_lon = map(float, end.split(','))
        except ValueError:
            return jsonify({"error": "Invalid start or end location format"}), 400

        routes = prediction.suggest_alternative_routes(start_lat, start_lon, end_lat, end_lon)
        return jsonify(routes)
    else:
        return jsonify({"error": "Start and end locations are required"}), 400

@predict_bp.route('/directions', methods=['GET'])
def directions():
    start = request.args.get('start')
    end = request.args.get('end')
    if start and end:
        try:
            start_lat, start_lon = map(float, start.split(','))
            end_lat, end_lon = map(float, end.split(','))
        except ValueError:
            return jsonify({"error": "Invalid start or end location format"}), 400

        directions = prediction.get_directions_with_traffic(start_lat, start_lon, end_lat, end_lon)
        return jsonify(directions)
    else:
        return jsonify({"error": "Start and end locations are required"}), 400