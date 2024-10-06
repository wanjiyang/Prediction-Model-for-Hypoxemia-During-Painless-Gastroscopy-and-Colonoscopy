from flask import Flask

def create_app():
    app = Flask(__name__)

    # 导入并注册路由
    from .routes import init_routes
    init_routes(app)

    return app
