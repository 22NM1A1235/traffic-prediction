#!/bin/bash

# Traffic Prediction System - Deployment Script
# Usage: ./deploy.sh [docker|linux|aws|heroku]

set -e

DEPLOYMENT_METHOD=${1:-docker}
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "═══════════════════════════════════════════════════════════════"
echo "Traffic Flow Prediction System - Deployment Script"
echo "═══════════════════════════════════════════════════════════════"
echo ""

case $DEPLOYMENT_METHOD in
    docker)
        echo "🐳 Deploying with Docker & Docker Compose..."
        
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker is not installed. Please install Docker first."
            exit 1
        fi
        
        echo "✓ Building Docker image..."
        docker-compose build
        
        echo "✓ Starting services..."
        docker-compose up -d
        
        echo "✓ Waiting for service to be ready..."
        sleep 5
        
        if docker-compose ps | grep -q "traffic_prediction_app"; then
            echo "✅ Deployment successful!"
            echo "   Application URL: http://localhost:5000"
            echo "   View logs: docker-compose logs -f"
            echo "   Stop: docker-compose down"
        else
            echo "❌ Deployment failed. Check logs with: docker-compose logs"
            exit 1
        fi
        ;;
        
    linux)
        echo "🐧 Preparing Linux/Ubuntu deployment..."
        
        if ! command -v python3 &> /dev/null; then
            echo "❌ Python 3 is not installed."
            exit 1
        fi
        
        echo "✓ Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        
        echo "✓ Installing dependencies..."
        pip install -r requirements.txt
        pip install gunicorn
        
        echo "✓ Copying service file..."
        echo "⚠️  You need to manually:"
        echo "   1. Copy service file to /etc/systemd/system/"
        echo "   2. Run: sudo systemctl daemon-reload"
        echo "   3. Run: sudo systemctl enable traffic-prediction"
        echo "   4. Run: sudo systemctl start traffic-prediction"
        echo ""
        echo "For more details, see DEPLOYMENT.md"
        ;;
        
    aws)
        echo "☁️  Preparing AWS EC2 deployment..."
        
        echo "⚠️  Manual AWS deployment steps:"
        echo "   1. Launch EC2 instance (Ubuntu 22.04+)"
        echo "   2. SSH into instance"
        echo "   3. Run: curl -fsSL https://raw.githubusercontent.com/.../deploy-linux.sh | bash"
        echo "   4. Follow Linux deployment instructions"
        echo ""
        echo "For detailed steps, see DEPLOYMENT.md"
        ;;
        
    heroku)
        echo "☁️  Preparing Heroku deployment..."
        
        if ! command -v heroku &> /dev/null; then
            echo "❌ Heroku CLI is not installed."
            echo "   Install from: https://devcenter.heroku.com/articles/heroku-cli"
            exit 1
        fi
        
        echo "✓ Logging into Heroku..."
        heroku login
        
        echo "✓ Creating app..."
        heroku create traffic-prediction-$(date +%s)
        
        echo "✓ Setting environment variables..."
        heroku config:set FLASK_ENV=production
        heroku config:set SECRET_KEY=$(openssl rand -hex 16)
        
        echo "✓ Deploying..."
        git push heroku main
        
        echo "✅ Deployment successful!"
        echo "   View logs: heroku logs --tail"
        echo "   Open app: heroku open"
        ;;
        
    *)
        echo "Usage: ./deploy.sh [docker|linux|aws|heroku]"
        echo ""
        echo "Options:"
        echo "  docker   - Deploy using Docker & Docker Compose"
        echo "  linux    - Deploy on Linux/Ubuntu server"
        echo "  aws      - Deploy on AWS EC2"
        echo "  heroku   - Deploy on Heroku"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
