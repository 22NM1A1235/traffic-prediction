#!/usr/bin/env python3
"""
Production WSGI Application Entry Point
Used with gunicorn: gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
"""
import os
from app import app

if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
