import os
from datetime import timedelta

from dotenv import load_dotenv
from flask import Flask
from flask_jwt_extended import JWTManager

from routes.auth_routes import auth_bp
from routes.farm_routes import farm_bp
from routes.parcel_routes import parcel_bp
from routes.parcel_season_routes import parcel_season_bp
from routes.parcel_soil_routes import parcel_soil_bp
from routes.yield_simulation_routes import yield_simulation_bp
from routes.sustainability_routes import sustainability_bp
from flask_cors import CORS


load_dotenv()

app = Flask(__name__)
CORS(
    app,
    origins=[
        "http://localhost:4200",
    ],
)

app.config["JWT_SECRET_KEY"] = os.getenv(
    "JWT_SECRET_KEY"
)

app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(
    hours=1
)

jwt = JWTManager(app)

app.register_blueprint(auth_bp)
app.register_blueprint(farm_bp)
app.register_blueprint(parcel_bp)
app.register_blueprint(parcel_season_bp)
app.register_blueprint(parcel_soil_bp)
app.register_blueprint(yield_simulation_bp)
app.register_blueprint(sustainability_bp)


@app.get("/api/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True)