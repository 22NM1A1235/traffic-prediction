#!/usr/bin/env python3
"""
Quick Deployment Test & Verification
"""
import subprocess
import sys
import os

print("\n" + "="*100)
print("🚀 DEPLOYMENT READINESS CHECK")
print("="*100 + "\n")

checks = {
    "Core Files": [
        ("app.py", "Flask application"),
        ("model.py", "Neural network"),
        ("wsgi.py", "WSGI entry point"),
        ("Dockerfile", "Docker config"),
        ("docker-compose.yml", "Docker Compose config"),
        ("requirements.txt", "Dependencies"),
        ("Procfile", "Heroku config"),
    ],
    "Model Files": [
        ("saved_models/st_mlp.pth", "Trained model"),
        ("saved_models/scaler.pkl", "Flow scaler"),
        ("saved_models/static_scaler.pkl", "Coordinate scaler"),
        ("saved_models/sensor_ids.pkl", "Sensor list"),
    ],
    "Data Files": [
        ("sensors.csv", "Sensor locations"),
        ("traffic_time_series.csv", "Traffic data"),
    ],
    "Documentation": [
        ("README.md", "README"),
        ("DEPLOYMENT.md", "Deployment guide"),
        ("requirements.txt", "Requirements"),
    ]
}

total_files = 0
found_files = 0

for category, files in checks.items():
    print(f"\n📋 {category}:")
    for filename, description in files:
        filepath = f"c:/Users/91888/Downloads/wetransfer_code-zip_2026-02-14_1046/Code/Code/{filename}"
        exists = os.path.exists(filepath)
        total_files += 1
        if exists:
            found_files += 1
            print(f"  ✓ {filename:<40} ({description})")
        else:
            print(f"  ✗ {filename:<40} (⚠️  {description})")

print("\n" + "-"*100)
print(f"\nFile Check: {found_files}/{total_files} files ready")

# Check Python packages
print("\n📦 Dependencies Check:")
required_packages = [
    "flask",
    "torch",
    "numpy",
    "pandas",
    "sklearn",
]

packages_ok = 0
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
        packages_ok += 1
    except ImportError:
        print(f"  ✗ {package} (not installed)")

print("\n" + "="*100)
print("DEPLOYMENT OPTIONS AVAILABLE:")
print("="*100)

deployment_options = [
    {
        "name": "Docker (Recommended)",
        "command": "docker-compose up -d",
        "requires": "Docker & Docker Compose",
        "time": "~30 seconds",
        "cost": "Variable"
    },
    {
        "name": "Local/Development",
        "command": "python app.py",
        "requires": "Python 3.8+",
        "time": "~3 seconds",
        "cost": "Free"
    },
    {
        "name": "Linux/Ubuntu Server",
        "command": "See DEPLOYMENT.md",
        "requires": "Linux server + systemd",
        "time": "~10 minutes",
        "cost": "~$5-20/month"
    },
    {
        "name": "AWS EC2",
        "command": "See DEPLOYMENT.md",
        "requires": "AWS account",
        "time": "~20 minutes",
        "cost": "~$10-50/month"
    },
    {
        "name": "Heroku",
        "command": "git push heroku main",
        "requires": "Heroku account + CLI",
        "time": "~1 minute",
        "cost": "Free-$7/month"
    },
    {
        "name": "Docker Hub + Kubernetes",
        "command": "kubectl apply -f k8s/",
        "requires": "Kubernetes cluster",
        "time": "~5 minutes",
        "cost": "$20-100+/month"
    },
]

for i, option in enumerate(deployment_options, 1):
    print(f"\n{i}. {option['name'].upper()}")
    print(f"   Command:  {option['command']}")
    print(f"   Requires: {option['requires']}")
    print(f"   Time:     {option['time']}")
    print(f"   Cost:     {option['cost']}")

print("\n" + "="*100)
print("RECOMMENDED NEXT STEPS:")
print("="*100)

print("""
1️⃣  For Development/Testing:
    → Run: python app.py
    → Access: http://localhost:5000

2️⃣  For Production with Docker:
    → Install Docker Desktop
    → Run: docker-compose up -d
    → Access: http://localhost:5000
    → View logs: docker-compose logs -f

3️⃣  For Cloud Deployment:
    → Read: DEPLOYMENT.md
    → Choose platform (Heroku, AWS, Azure, GCP)
    → Follow platform-specific instructions

4️⃣  For Enterprise/Kubernetes:
    → Create Kubernetes manifests
    → Set up CI/CD pipeline
    → Deploy to K8s cluster

📚 Full Documentation: See DEPLOYMENT.md
🔒 Security Checklist: See DEPLOYMENT.md (Security Checklist section)
""")

print("="*100)
print(f"\n✅ DEPLOYMENT STATUS: {'READY ✓' if found_files == total_files and packages_ok >= 4 else 'PARTIALLY READY ⚠️'}")
print("="*100 + "\n")
