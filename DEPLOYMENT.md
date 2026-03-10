# Traffic Flow Prediction System - Deployment Guide

## Overview

This guide covers deploying the Traffic Flow Prediction System to production using various methods.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Linux/Ubuntu Server](#linuxubuntu-server)
4. [AWS EC2 Deployment](#aws-ec2-deployment)
5. [Heroku Deployment](#heroku-deployment)
6. [Performance Optimization](#performance-optimization)
7. [Security Checklist](#security-checklist)

---

## Local Development

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone or extract project
cd Code

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py
```

**Access:** http://localhost:5000

---

## Docker Deployment

### Prerequisites
- Docker
- Docker Compose

### Quick Start

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access:** http://localhost:5000

### Manual Docker Build

```bash
# Build image
docker build -t traffic-prediction:latest .

# Run container
docker run -d \
  --name traffic-app \
  -p 5000:5000 \
  -v $(pwd)/saved_models:/app/saved_models \
  -v $(pwd)/users.db:/app/users.db \
  traffic-prediction:latest

# View logs
docker logs -f traffic-app
```

### Docker Troubleshooting

```bash
# Rebuild without cache
docker-compose build --no-cache

# Remove old containers/images
docker system prune -a

# Check container health
docker ps --all

# Enter container shell
docker exec -it traffic-app bash
```

---

## Linux/Ubuntu Server

### Prerequisites
- Python 3.8+
- pip
- systemd
- Nginx (optional, for reverse proxy)

### Installation

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y postgresql postgresql-contrib  # Optional

# Clone/extract project
cd /opt
sudo git clone <repository-url> traffic-prediction
cd traffic-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn
```

### Create Systemd Service

Create `/etc/systemd/system/traffic-prediction.service`:

```ini
[Unit]
Description=Traffic Flow Prediction Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/traffic-prediction
Environment="PATH=/opt/traffic-prediction/venv/bin"
EnvironmentFile=/opt/traffic-prediction/.env
ExecStart=/opt/traffic-prediction/venv/bin/gunicorn \
    -w 4 \
    -b 127.0.0.1:5000 \
    --timeout 120 \
    --access-logfile /var/log/traffic-prediction/access.log \
    --error-logfile /var/log/traffic-prediction/error.log \
    wsgi:app

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Enable Service

```bash
# Create log directory
sudo mkdir -p /var/log/traffic-prediction
sudo chown www-data:www-data /var/log/traffic-prediction

# Enable service
sudo systemctl daemon-reload
sudo systemctl enable traffic-prediction
sudo systemctl start traffic-prediction

# Check status
sudo systemctl status traffic-prediction

# View logs
sudo journalctl -u traffic-prediction -f
```

### Nginx Reverse Proxy Configuration

Create `/etc/nginx/sites-available/traffic-prediction`:

```nginx
upstream traffic_app {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 16M;

    location / {
        proxy_pass http://traffic_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        proxy_buffering off;
        proxy_request_buffering off;
    }

    location /static {
        alias /opt/traffic-prediction/static;
    }

    # SSL configuration (if using HTTPS)
    # listen 443 ssl http2;
    # ssl_certificate /path/to/cert.pem;
    # ssl_certificate_key /path/to/key.pem;
}
```

Enable Nginx config:

```bash
sudo ln -s /etc/nginx/sites-available/traffic-prediction \
    /etc/nginx/sites-enabled/

sudo nginx -t
sudo systemctl restart nginx
```

---

## AWS EC2 Deployment

### Step 1: Launch EC2 Instance

```bash
# Instance specifications
- AMI: Ubuntu 22.04 LTS
- Instance Type: t3.medium (minimum)
- Storage: 50GB (for models and data)
- Security Group: Allow SSH (22), HTTP (80), HTTPS (443)
```

### Step 2: Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<public-ip>

# Follow Linux/Ubuntu setup above
```

### Step 3: SSL Certificate (Let's Encrypt)

```bash
sudo apt install -y certbot python3-certbot-nginx

sudo certbot certonly --nginx -d your-domain.com

# Update Nginx config with SSL certificates
```

### Step 4: AWS RDS Setup (Optional)

```bash
# If using RDS for database instead of SQLite
# Update DATABASE_URL in .env:
DATABASE_URL=postgresql://user:password@rds-endpoint:5432/traffic_db
```

### Step 5: Monitoring

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure and start
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -s
```

---

## Heroku Deployment

### Prerequisites
- Heroku CLI
- Git

### Setup

```bash
# Login to Heroku
heroku login

# Create app
heroku create traffic-prediction-app

# Add buildpacks
heroku buildpacks:add heroku/python

# Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=<generate-random-key>

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

### Procfile

Create `Procfile` in project root:

```
web: gunicorn -w 4 -b 0.0.0.0:$PORT wsgi:app
clock: python scheduler.py
```

---

## Performance Optimization

### 1. Caching

```python
# Add to app.py
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/prediction', methods=['GET'])
@cache.cached(timeout=300)  # Cache for 5 minutes
def prediction():
    # ...
```

### 2. Database Optimization

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

SQLALCHEMY_ENGINE_OPTIONS = {
    "poolclass": QueuePool,
    "pool_size": 10,
    "pool_recycle": 3600,
    "pool_pre_ping": True,
}
```

### 3. Model Optimization

```python
# Load model once at startup
model = None

@app.before_first_request
def load_model():
    global model
    model = STMLP(...)
    model.eval()
```

### 4. Async Processing

```bash
# Install Celery for task queue
pip install celery redis

# Process predictions asynchronously
@celery.task
def predict_traffic(lat, lng):
    # Long-running prediction
    return result
```

---

## Security Checklist

- [ ] Change SECRET_KEY in production
- [ ] Set FLASK_ENV=production
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Add rate limiting
- [ ] Implement API authentication (JWT tokens)
- [ ] Sanitize user inputs
- [ ] Use environment variables for secrets
- [ ] Enable CSRF protection
- [ ] Set secure session cookies
- [ ] Regular security updates
- [ ] Log and monitor access
- [ ] Use strong database passwords
- [ ] Backup database regularly
- [ ] Implement DDoS protection (CloudFlare, AWS Shield)

### Security Headers

```python
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response
```

---

## Monitoring & Logging

### Application Logs

```bash
# Using Python logging
import logging

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
```

### Health Check

```bash
# Endpoint for monitoring
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    }
```

### Metrics to Monitor

- Request latency
- Error rate
- CPU usage
- Memory usage
- Database connections
- Model prediction accuracy

---

## Troubleshooting

### Port Already in Use
```bash
# Find and kill process
lsof -i :5000
kill -9 <PID>
```

### Module Not Found
```bash
# Ensure all dependencies installed
pip install -r requirements.txt --upgrade
```

### Model Loading Issues
```bash
# Check model path
ls -la saved_models/

# Verify torch installation
python -c "import torch; print(torch.__version__)"
```

### Database Locked
```bash
# SQLite issue, restart service
sudo systemctl restart traffic-prediction
```

---

## Disaster Recovery

### Backup Strategy

```bash
# Daily backup
0 2 * * * backup_script.sh

# Backup script
#!/bin/bash
tar -czf backup_$(date +%Y%m%d).tar.gz \
    users.db \
    saved_models/
```

### Restore from Backup

```bash
tar -xzf backup_20260305.tar.gz
```

---

## Support & Further Reading

- Flask Deployment: https://flask.palletsprojects.com/deployment/
- Gunicorn: https://gunicorn.org/
- Docker: https://docs.docker.com/
- Nginx: https://nginx.org/en/docs/

---

**Last Updated:** March 5, 2026
