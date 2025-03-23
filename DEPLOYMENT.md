# Beat Detection Web Application - Deployment Guide

This comprehensive guide covers the deployment process for the Beat Detection Web Application in a production environment. We provide instructions for different hosting options with a focus on security, performance, and maintainability.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Preparing Your Application](#preparing-your-application)
4. [Deployment Options](#deployment-options)
   - [Self-Hosted VPS Deployment](#self-hosted-vps-deployment)
   - [Gandi Web Hosting Deployment](#gandi-web-hosting-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Database Setup](#database-setup)
7. [User Management](#user-management)
8. [Securing Your Application](#securing-your-application)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)
10. [Troubleshooting](#troubleshooting)

## Deployment Overview

The Beat Detection Web Application is built with FastAPI and relies on several components:

- **Web Server**: Serves the application through ASGI (Uvicorn/Gunicorn)
- **Background Worker**: Processes audio files and generates videos (Celery)
- **Message Broker**: Coordinates between web server and workers (Redis)
- **File Storage**: Stores uploaded audio files and generated videos
- **Video Processing**: Requires ffmpeg for video generation

Each component must be properly configured for a reliable production deployment.

## Prerequisites

Ensure you have:

- Basic familiarity with Linux server administration
- SSH access to your server (for VPS deployments)
- Server IP address
- SSL certificate (optional for development)

### System Requirements

1. **Install ffmpeg** (required for video generation):

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install -y ffmpeg

# Verify installation
ffmpeg -version
```

2. **Install uv** (fast Python package installer):

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Install Python 3.10+** (required for librosa):

```bash
# Install Python 3.10 using uv
uv venv --python 3.10.13

# Activate the virtual environment
source .venv/bin/activate

# Verify installation
python --version
```

## Preparing Your Application

1. **Clone your repository or prepare your deployment package**:

```bash
git clone https://your-repo-url.git
# OR upload your application files
```

2. **Install dependencies**:

```bash
# Install dependencies from pyproject.toml
uv pip install .
```

3. **Create essential directories**:

```bash
mkdir -p web_app/uploads
```

## Deployment Options

### Self-Hosted VPS Deployment

This option gives you full control over the environment and is recommended for applications with specific requirements.

#### 1. Server Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y nginx supervisor redis-server

# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sudo sh
```

#### 2. Configure Nginx

Create `/etc/nginx/sites-available/beats-app.conf`:

```nginx
server {
    listen 80;
    server_name your_server_ip;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /static {
        alias /path/to/your/app/web_app/static;
        expires 30d;
    }
    
    # Increase upload size limit for audio files
    client_max_body_size 50M;
}
```

Enable the configuration:

```bash
sudo ln -s /etc/nginx/sites-available/beats-app.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 3. Configure Supervisor

Create `/etc/supervisor/conf.d/beats-app.conf`:

```ini
[program:beats-web]
command=/path/to/your/app/.venv/bin/uvicorn web_app.asgi:app --host 127.0.0.1 --port 8000 --workers 4
directory=/path/to/your/app
user=www-data
autostart=true
autorestart=true
stderr_logfile=/path/to/your/app/logs/web.err.log
stdout_logfile=/path/to/your/app/logs/web.out.log

[program:beats-worker]
command=/path/to/your/app/.venv/bin/celery -A web_app.celery_app worker --loglevel=info
directory=/path/to/your/app
user=www-data
autostart=true
autorestart=true
stderr_logfile=/path/to/your/app/logs/worker.err.log
stdout_logfile=/path/to/your/app/logs/worker.out.log

[group:beats-app]
programs=beats-web,beats-worker
```

Create required directories and set permissions:

```bash
# Create logs directory
mkdir -p /path/to/your/app/logs

# Set appropriate permissions
sudo chown -R www-data:www-data /path/to/your/app/logs
sudo chmod -R 755 /path/to/your/app/logs
```

Apply the configuration:

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status
```

#### 4. Set Up SSL with Let's Encrypt (Optional)

If you want to use HTTPS, you'll need a domain name. For development, you can use HTTP or set up a self-signed certificate:

```bash
# Generate self-signed certificate (for development only)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
-keyout /etc/ssl/private/nginx-selfsigned.key \
-out /etc/ssl/certs/nginx-selfsigned.crt
```

Then update your Nginx configuration to use HTTPS:

```nginx
server {
    listen 443 ssl;
    server_name your_server_ip;
    
    ssl_certificate /etc/ssl/certs/nginx-selfsigned.crt;
    ssl_certificate_key /etc/ssl/private/nginx-selfsigned.key;
    
    # ... rest of your configuration ...
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your_server_ip;
    return 301 https://$server_name$request_uri;
}
```

### Gandi Web Hosting Deployment

If you're using Gandi's hosting services, follow these steps.

#### Gandi Simple Hosting (Shared Hosting)

1. **Log in to Gandi dashboard** and select your Simple Hosting instance

2. **Create a deployment package**:

```bash
# Create a deployment zip
zip -r deployment.zip . -x ".venv/*" -x ".git/*" -x "*.pyc" -x "__pycache__/*"
```

3. **Upload the package** via Gandi's web interface or SFTP

4. **Configure the application** through Gandi's control panel:
   - Set Python version to 3.10+
   - Configure environment variables (see Environment Configuration below)
   - Set the entry point to `web_app.asgi:app`

5. **Set up a vhost** to map your domain to the application

#### Gandi Cloud VPS

Follow the [Self-Hosted VPS Deployment](#self-hosted-vps-deployment) instructions after setting up your Gandi VPS.

## Environment Configuration

Create a `.env` file in your application root (never commit this to version control):

```bash
# Application settings
DEBUG=False
JWT_SECRET_KEY=your_secure_random_key
ALLOWED_HOSTS=your_server_ip

# Redis configuration
REDIS_URL=redis://localhost:6379/0

# File storage paths
UPLOAD_DIR=./web_app/uploads

# Media limits
MAX_UPLOAD_SIZE=50
```

The application also uses configuration files stored in the `web_app/config` directory:

- `config.json` - General application settings
- `users.json` - User credentials and information

These files are created automatically when the application starts if they don't exist. You can modify them manually or through the application's admin interface.

In production, you should use a secure random string for the JWT secret:

```bash
# Generate a secure random key
python -c "import secrets; print(secrets.token_hex(32))"
```

## Database Setup

If you're using Redis for Celery tasks:

```bash
# Install Redis (if not installed)
sudo apt install -y redis-server

# Start and enable Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify Redis is working
redis-cli ping
```

For enhanced security, configure Redis to require a password and limit connections.

## User Management

The application includes a command-line tool to manage user accounts. This is particularly useful during initial setup and for ongoing user administration:

```bash
# List all users
python tools/manage_users.py list

# Add a new regular user (password will be generated if not provided)
python tools/manage_users.py add username [--password PASSWORD]

# Add an admin user
python tools/manage_users.py add username [--password PASSWORD] --admin

# Delete a user
python tools/manage_users.py delete username

# Change a user's password
python tools/manage_users.py password username new_password
```

Initial setup should include creating at least one admin user:

```bash
python tools/manage_users.py add admin --password your_secure_password --admin
```

## Securing Your Application

### 1. File Permissions

```bash
# Set appropriate permissions
sudo chown -R www-data:www-data /path/to/your/app
sudo chmod -R 755 /path/to/your/app
sudo chmod -R 770 /path/to/your/app/web_app/uploads
sudo chmod -R 750 /path/to/your/app/web_app/config
```

### 2. Firewall Configuration

```bash
# Configure firewall (UFW)
sudo apt install -y ufw
sudo ufw allow 'Nginx Full'
sudo ufw allow 22/tcp  # SSH
sudo ufw enable
```

### 3. Install Fail2ban to Prevent Brute Force Attacks

```bash
sudo apt install -y fail2ban
```

Create `/etc/fail2ban/jail.local`:

```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[nginx-http-auth]
enabled = true
```

Start and enable fail2ban:

```bash
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 4. Security Headers Configuration

Add security headers in Nginx:

```nginx
# In your server {} block
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;" always;
```

## Monitoring and Maintenance

### 1. Set Up Basic System Monitoring

```bash
# Install monitoring tools
sudo apt install -y htop logwatch

# Install a more comprehensive monitoring tool like Netdata
bash <(curl -Ss https://my-netdata.io/kickstart.sh)
```

### 2. Log Rotation

Create `/etc/logrotate.d/beats-app`:

```
/path/to/your/app/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        supervisorctl restart beats-app:*
    endscript
}
```

### 3. Regular Updates

Set up a maintenance schedule:

```bash
# Create update script
cat > /usr/local/bin/update-beats-app.sh << 'EOF'
#!/bin/bash
cd /path/to/your/app
git pull
source .venv/bin/activate
uv pip install .
supervisorctl restart beats-app:*
EOF

chmod +x /usr/local/bin/update-beats-app.sh
```

## Troubleshooting

### Common Issues and Solutions

#### Application Won't Start

Check supervisor logs:
```bash
sudo supervisorctl status
sudo tail -f /path/to/your/app/logs/web.err.log
```

#### Celery Workers Not Processing Tasks

Check Redis connection:
```bash
redis-cli ping
```

Check worker logs:
```bash
sudo tail -f /path/to/your/app/logs/worker.err.log
```

#### Permission Errors

Check and fix permissions:
```bash
sudo chown -R www-data:www-data /path/to/your/app
sudo chmod -R 755 /path/to/your/app
```

#### Nginx Errors

Check Nginx configuration:
```bash
sudo nginx -t
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log
```

## Conclusion

This deployment document provides a comprehensive guide for setting up your Beat Detection Web Application in a production environment. Follow these guidelines to ensure a secure, reliable, and maintainable deployment.

For specific issues or advanced configurations, consult the documentation for individual components (FastAPI, Celery, Redis, Nginx, etc.) or reach out to your hosting provider's support team.

---

**Note**: Always test your deployment in a staging environment before applying changes to production. Regularly review and update your deployment procedures as your application evolves. 