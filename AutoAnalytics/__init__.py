from flask import Flask
from AutoAnalytics.config import Config

def create_app(config_class=Config):

    app = Flask(__name__)
    app.config.from_object(Config)
    
    from AutoAnalytics.routes import main
    from AutoAnalytics.errors.handlers import errors
    app.register_blueprint(main)
    app.register_blueprint(errors)    

    return app
