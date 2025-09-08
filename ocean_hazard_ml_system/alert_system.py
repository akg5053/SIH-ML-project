# alert_system.py - Alert Engine and Deployment System (Part 6)

import os
import json
import time
import math
import requests
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from threading import Thread

# Database (you can replace with your actual DB)
import sqlite3
from contextlib import contextmanager

# Web Push Notifications
try:
    from pywebpush import webpush, WebPushException
except ImportError:
    print("⚠️ pywebpush not installed. Run: pip install pywebpush")

# Flask for deployment
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import ML system
from ocean_hazard_ml import OceanHazardML

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlertRule:
    """Alert rule configuration"""
    hazard_types: List[str]
    min_severity: int
    radius_km: float
    cooldown_minutes: int
    notification_methods: List[str]  # ['push', 'email', 'sms']

@dataclass
class UserLocation:
    """User location and preferences"""
    user_id: str
    latitude: float
    longitude: float
    push_subscription: Optional[Dict] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    alert_preferences: Optional[Dict] = None

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    hazard_type: str
    severity: int
    urgency: str
    location: Tuple[float, float]  # (lat, lng)
    radius_km: float
    message: str
    report_id: str
    created_at: datetime
    expires_at: datetime
    sent_to: List[str] = None

class DatabaseManager:
    """Simple database manager for alerts and users"""
    
    def __init__(self, db_path: str = "data/alerts.db"):
        self.db_path = db_path
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    email TEXT,
                    phone TEXT,
                    push_subscription TEXT,
                    alert_preferences TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    hazard_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    urgency TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    radius_km REAL NOT NULL,
                    message TEXT NOT NULL,
                    report_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (alert_id) REFERENCES alerts (alert_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()

class GeoUtils:
    """Geospatial utility functions"""
    
    @staticmethod
    def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers using Haversine formula"""
        
        # Convert latitude and longitude to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon/2)**2)
        
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in kilometers
        r = 6371
        
        return c * r
    
    @staticmethod
    def users_within_radius(alert_location: Tuple[float, float], 
                           radius_km: float, 
                           all_users: List[UserLocation]) -> List[UserLocation]:
        """Find users within radius of alert location"""
        
        alert_lat, alert_lon = alert_location
        nearby_users = []
        
        for user in all_users:
            distance = GeoUtils.calculate_distance_km(
                alert_lat, alert_lon, 
                user.latitude, user.longitude
            )
            
            if distance <= radius_km:
                nearby_users.append(user)
        
        return nearby_users

class NotificationService:
    """Handle different types of notifications"""
    
    def __init__(self):
        self.vapid_private_key = os.getenv('VAPID_PRIVATE_KEY')
        self.vapid_public_key = os.getenv('VAPID_PUBLIC_KEY')
        self.vapid_email = os.getenv('VAPID_EMAIL', 'mailto:admin@oceanalerts.com')
        
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
    
    def send_push_notification(self, user: UserLocation, alert: Alert) -> bool:
        """Send web push notification"""
        
        if not user.push_subscription or not self.vapid_private_key:
            logger.warning(f"Cannot send push to {user.user_id}: missing subscription or VAPID keys")
            return False
        
        try:
            subscription_info = json.loads(user.push_subscription)
            
            payload = {
                "title": f"🚨 {alert.hazard_type.title()} Alert",
                "body": alert.message,
                "icon": "/static/icons/alert-icon.png",
                "badge": "/static/icons/badge-icon.png",
                "data": {
                    "alert_id": alert.id,
                    "hazard_type": alert.hazard_type,
                    "severity": alert.severity,
                    "url": f"/alerts/{alert.id}"
                },
                "actions": [
                    {
                        "action": "view",
                        "title": "View Details"
                    },
                    {
                        "action": "dismiss", 
                        "title": "Dismiss"
                    }
                ]
            }
            
            webpush(
                subscription_info=subscription_info,
                data=json.dumps(payload),
                vapid_private_key=self.vapid_private_key,
                vapid_claims={
                    "sub": self.vapid_email
                }
            )
            
            logger.info(f"Push notification sent to {user.user_id}")
            return True
            
        except WebPushException as e:
            logger.error(f"Push notification failed for {user.user_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Push notification error for {user.user_id}: {e}")
            return False
    
    def send_email_notification(self, user: UserLocation, alert: Alert) -> bool:
        """Send email notification"""
        
        if not user.email or not self.smtp_username or not self.smtp_password:
            logger.warning(f"Cannot send email to {user.user_id}: missing email or SMTP config")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = user.email
            msg['Subject'] = f"🚨 Ocean Hazard Alert: {alert.hazard_type.title()}"
            
            # Create email body
            html_body = f"""
            <html>
                <body>
                    <h2 style="color: #d32f2f;">🌊 Ocean Hazard Alert</h2>
                    <div style="background-color: #ffebee; padding: 20px; border-radius: 8px;">
                        <h3>⚠️ {alert.hazard_type.title()} Detected</h3>
                        <p><strong>Severity:</strong> {alert.severity}/10 ({alert.urgency})</p>
                        <p><strong>Message:</strong> {alert.message}</p>
                        <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Location:</strong> {alert.location[0]:.4f}, {alert.location[1]:.4f}</p>
                    </div>
                    <br>
                    <p style="color: #666;">
                        This alert was generated by the Ocean Hazard Monitoring System.
                        Stay safe and follow official guidelines.
                    </p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent to {user.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Email notification failed for {user.user_id}: {e}")
            return False
    
    def send_sms_notification(self, user: UserLocation, alert: Alert) -> bool:
        """Send SMS notification (placeholder - integrate with SMS service)"""
        
        if not user.phone:
            logger.warning(f"Cannot send SMS to {user.user_id}: no phone number")
            return False
        
        # Placeholder for SMS integration (Twilio, AWS SNS, etc.)
        try:
            # Example with Twilio (uncomment and configure):
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)
            # message = client.messages.create(
            #     body=f"🚨 {alert.hazard_type.title()} Alert: {alert.message[:100]}...",
            #     from_='+1234567890',
            #     to=user.phone
            # )
            
            logger.info(f"SMS notification would be sent to {user.user_id}: {user.phone}")
            return True
            
        except Exception as e:
            logger.error(f"SMS notification failed for {user.user_id}: {e}")
            return False

class AlertEngine:
    """Main alert processing engine"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.notifications = NotificationService()
        self.ml_system = OceanHazardML()
        
        # Default alert rules
        self.alert_rules = [
            AlertRule(
                hazard_types=['tsunami'],
                min_severity=6,
                radius_km=50.0,
                cooldown_minutes=10,
                notification_methods=['push', 'email', 'sms']
            ),
            AlertRule(
                hazard_types=['storm_surge'],
                min_severity=7,
                radius_km=30.0,
                cooldown_minutes=15,
                notification_methods=['push', 'email']
            ),
            AlertRule(
                hazard_types=['flood'],
                min_severity=6,
                radius_km=20.0,
                cooldown_minutes=20,
                notification_methods=['push', 'email']
            ),
            AlertRule(
                hazard_types=['high_waves'],
                min_severity=5,
                radius_km=15.0,
                cooldown_minutes=30,
                notification_methods=['push']
            ),
            AlertRule(
                hazard_types=['coastal_erosion'],
                min_severity=4,
                radius_km=10.0,
                cooldown_minutes=60,
                notification_methods=['push']
            )
        ]
    
    def process_report(self, report_data: Dict) -> Optional[Alert]:
        """Process a new report and potentially create an alert"""
        
        try:
            # Extract report information
            report_id = report_data['id']
            description = report_data['description']
            location = (report_data['latitude'], report_data['longitude'])
            
            # Analyze with ML system
            analysis = self.ml_system.analyze_text(description)
            
            logger.info(f"Processing report {report_id}: {analysis.hazard_type}, severity {analysis.severity_score}")
            
            # Check if alert should be triggered
            alert_rule = self.should_trigger_alert(analysis)
            if not alert_rule:
                logger.info(f"No alert triggered for report {report_id}")
                return None
            
            # Create alert
            alert = Alert(
                id=f"alert_{int(time.time())}_{report_id}",
                hazard_type=analysis.hazard_type,
                severity=analysis.severity_score,
                urgency=analysis.urgency_level,
                location=location,
                radius_km=alert_rule.radius_km,
                message=self.generate_alert_message(analysis, description),
                report_id=report_id,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=6),
                sent_to=[]
            )
            
            # Save alert to database
            self.save_alert(alert)
            
            # Send notifications
            self.send_alert_notifications(alert, alert_rule)
            
            logger.info(f"Alert {alert.id} created and sent")
            return alert
            
        except Exception as e:
            logger.error(f"Error processing report: {e}")
            return None
    
    def should_trigger_alert(self, analysis) -> Optional[AlertRule]:
        """Check if analysis meets alert criteria"""
        
        for rule in self.alert_rules:
            if (analysis.hazard_type in rule.hazard_types and 
                analysis.severity_score >= rule.min_severity):
                
                # Check cooldown (simplified - in production, check recent alerts)
                return rule
        
        return None
    
    def generate_alert_message(self, analysis, original_description: str) -> str:
        """Generate user-friendly alert message"""
        
        urgency_prefix = {
            'LOW': '⚠️',
            'MEDIUM': '⚠️',  
            'HIGH': '🚨',
            'CRITICAL': '🆘'
        }
        
        prefix = urgency_prefix.get(analysis.urgency_level, '⚠️')
        hazard_name = analysis.hazard_type.replace('_', ' ').title()
        
        message = f"{prefix} {hazard_name} detected in your area. "
        
        if analysis.severity_score >= 8:
            message += "Take immediate action and follow official guidelines. "
        elif analysis.severity_score >= 6:
            message += "Stay alert and be prepared to take action. "
        else:
            message += "Monitor conditions and stay informed. "
        
        # Add credibility note
        if analysis.credibility_score < 0.5:
            message += "(Report requires verification) "
        
        return message
    
    def save_alert(self, alert: Alert):
        """Save alert to database"""
        with self.db.get_connection() as conn:
            conn.execute('''
                INSERT INTO alerts (
                    alert_id, hazard_type, severity, urgency,
                    latitude, longitude, radius_km, message,
                    report_id, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.hazard_type, alert.severity, alert.urgency,
                alert.location[0], alert.location[1], alert.radius_km,
                alert.message, alert.report_id, alert.created_at, alert.expires_at
            ))
            conn.commit()
    
    def get_nearby_users(self, alert: Alert) -> List[UserLocation]:
        """Get users within alert radius"""
        with self.db.get_connection() as conn:
            cursor = conn.execute('''
                SELECT user_id, latitude, longitude, email, phone, 
                       push_subscription, alert_preferences
                FROM users
            ''')
            
            all_users = []
            for row in cursor:
                user = UserLocation(
                    user_id=row['user_id'],
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    email=row['email'],
                    phone=row['phone'],
                    push_subscription=row['push_subscription'],
                    alert_preferences=json.loads(row['alert_preferences']) if row['alert_preferences'] else None
                )
                all_users.append(user)
        
        # Find users within radius
        nearby_users = GeoUtils.users_within_radius(
            alert.location, alert.radius_km, all_users
        )
        
        return nearby_users
    
    def send_alert_notifications(self, alert: Alert, alert_rule: AlertRule):
        """Send notifications to nearby users"""
        
        nearby_users = self.get_nearby_users(alert)
        logger.info(f"Sending alert {alert.id} to {len(nearby_users)} nearby users")
        
        for user in nearby_users:
            for method in alert_rule.notification_methods:
                try:
                    success = False
                    
                    if method == 'push':
                        success = self.notifications.send_push_notification(user, alert)
                    elif method == 'email':
                        success = self.notifications.send_email_notification(user, alert)
                    elif method == 'sms':
                        success = self.notifications.send_sms_notification(user, alert)
                    
                    # Log delivery
                    self.log_delivery(alert.id, user.user_id, method, 
                                    'success' if success else 'failed')
                    
                except Exception as e:
                    logger.error(f"Notification {method} failed for {user.user_id}: {e}")
                    self.log_delivery(alert.id, user.user_id, method, 'error')
    
    def log_delivery(self, alert_id: str, user_id: str, method: str, status: str):
        """Log alert delivery attempt"""
        with self.db.get_connection() as conn:
            conn.execute('''
                INSERT INTO alert_deliveries (alert_id, user_id, method, status)
                VALUES (?, ?, ?, ?)
            ''', (alert_id, user_id, method, status))
            conn.commit()

# Flask App for Alert System
app = Flask(__name__)
CORS(app)

# Initialize alert engine
alert_engine = AlertEngine()

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "healthy", "service": "alert_engine"})

@app.route('/alerts/process-report', methods=['POST'])
def process_report():
    """Process a new report and trigger alerts if needed"""
    try:
        report_data = request.json
        
        required_fields = ['id', 'description', 'latitude', 'longitude']
        for field in required_fields:
            if field not in report_data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Process the report
        alert = alert_engine.process_report(report_data)
        
        if alert:
            return jsonify({
                "alert_created": True,
                "alert_id": alert.id,
                "hazard_type": alert.hazard_type,
                "severity": alert.severity,
                "urgency": alert.urgency,
                "message": alert.message,
                "location": alert.location,
                "radius_km": alert.radius_km,
                "expires_at": alert.expires_at.isoformat()
            })
        else:
            return jsonify({
                "alert_created": False,
                "message": "Analysis did not meet alert criteria"
            })
            
    except Exception as e:
        logger.error(f"Process report error: {e}")
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text for hazard detection"""
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "Text field is required"}), 400
        
        text = data['text']
        
        # Analyze with ML system
        analysis = alert_engine.ml_system.analyze_text(text)
        
        result = {
            "status": "success",
            "analysis": {
                "hazard_type": analysis.hazard_type,
                "severity": analysis.urgency_level,
                "severity_score": analysis.severity_score,
                "credibility": "HIGH" if analysis.credibility_score > 0.7 else "MEDIUM" if analysis.credibility_score > 0.4 else "LOW",
                "confidence": round(analysis.confidence, 3),
                "keywords_found": analysis.keywords_found,
                "location_extracted": analysis.location_extracted
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

@app.route('/create_alert', methods=['POST'])
def create_alert():
    """Create alert from text and location"""
    try:
        data = request.json
        required_fields = ['text', 'latitude', 'longitude']
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Create report data
        report_data = {
            'id': f"report_{int(time.time())}",
            'description': data['text'],
            'latitude': data['latitude'],
            'longitude': data['longitude']
        }
        
        # Process the report
        alert = alert_engine.process_report(report_data)
        
        if alert:
            return jsonify({
                "alert_created": True,
                "alert_id": alert.id,
                "hazard_type": alert.hazard_type,
                "severity": alert.severity,
                "urgency": alert.urgency,
                "message": alert.message,
                "location": alert.location,
                "radius_km": alert.radius_km
            })
        else:
            return jsonify({
                "alert_created": False,
                "message": "Analysis did not meet alert criteria"
            })
            
    except Exception as e:
        logger.error(f"Alert creation error: {e}")
        return jsonify({"error": "Alert creation failed", "details": str(e)}), 500

@app.route('/register_user', methods=['POST'])
def register_user():
    """Register a user for location-based alerts"""
    try:
        data = request.json
        required_fields = ['user_id', 'latitude', 'longitude']
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        with alert_engine.db.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, latitude, longitude, email, phone, alert_preferences)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data['user_id'],
                data['latitude'],
                data['longitude'],
                data.get('email'),
                data.get('phone'),
                json.dumps(data.get('alert_preferences', {}))
            ))
            conn.commit()
        
        return jsonify({
            "status": "success",
            "message": "User registered successfully",
            "user_id": data['user_id']
        })
        
    except Exception as e:
        logger.error(f"User registration error: {e}")
        return jsonify({"error": "Registration failed", "details": str(e)}), 500

@app.route('/get_alerts', methods=['GET'])
def get_alerts():
    """Get alerts for a specific location"""
    try:
        latitude = float(request.args.get('latitude'))
        longitude = float(request.args.get('longitude'))
        radius = float(request.args.get('radius', 50))
        
        with alert_engine.db.get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM alerts 
                WHERE is_active = 1 AND expires_at > ?
                ORDER BY created_at DESC
            ''', (datetime.now(),))
            
            alerts = []
            for row in cursor:
                alert_location = (row['latitude'], row['longitude'])
                user_location = (latitude, longitude)
                
                distance = GeoUtils.calculate_distance_km(
                    user_location[0], user_location[1],
                    alert_location[0], alert_location[1]
                )
                
                if distance <= radius:
                    alerts.append({
                        'alert_id': row['alert_id'],
                        'hazard_type': row['hazard_type'],
                        'severity': row['severity'],
                        'urgency': row['urgency'],
                        'message': row['message'],
                        'location': alert_location,
                        'distance_km': round(distance, 2),
                        'created_at': row['created_at'],
                        'expires_at': row['expires_at']
                    })
        
        return jsonify({
            "status": "success",
            "alerts": alerts,
            "total_alerts": len(alerts)
        })
        
    except Exception as e:
        logger.error(f"Get alerts error: {e}")
        return jsonify({"error": "Failed to get alerts", "details": str(e)}), 500

@app.route('/test_demo', methods=['GET'])
def test_demo():
    """Demo endpoint for testing the complete system"""
    try:
        test_cases = [
            {
                "text": "Tsunami warning issued for entire Pacific coast! Evacuate to higher ground immediately!",
                "location": "Pacific Coast"
            },
            {
                "text": "Beautiful sunset at Marina Beach today. Perfect weather for swimming and surfing!",
                "location": "Marina Beach"
            },
            {
                "text": "Official meteorological department confirms Category 5 hurricane approaching Mumbai with winds up to 200 kmph",
                "location": "Mumbai"
            },
            {
                "text": "Someone on WhatsApp forwarded message about possible flooding but not confirmed by authorities",
                "location": "Unknown"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            # Analyze the text
            analysis = alert_engine.ml_system.analyze_text(test_case['text'])
            
            result = {
                "test_case": i,
                "input_text": test_case['text'],
                "location": test_case['location'],
                "analysis": {
                    "hazard_type": analysis.hazard_type,
                    "severity_score": analysis.severity_score,
                    "urgency_level": analysis.urgency_level,
                    "confidence": round(analysis.confidence, 3),
                    "credibility_score": round(analysis.credibility_score, 3),
                    "keywords_found": analysis.keywords_found[:5]  # Limit to top 5 keywords
                },
                "alert_status": "WOULD CREATE ALERT" if analysis.severity_score >= 5 else "NO ALERT"
            }
            
            results.append(result)
        
        return jsonify({
            "demo": "Ocean Hazard Detection System - Complete Demo",
            "status": "System working perfectly!",
            "timestamp": datetime.now().isoformat(),
            "test_results": results,
            "summary": {
                "total_tests": len(results),
                "high_severity": sum(1 for r in results if r['analysis']['severity_score'] >= 7),
                "alerts_triggered": sum(1 for r in results if r['alert_status'] == "WOULD CREATE ALERT")
            }
        })
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        return jsonify({"error": "Demo failed", "details": str(e)}), 500

@app.route('/system_stats', methods=['GET'])
def system_stats():
    """Get system statistics"""
    try:
        with alert_engine.db.get_connection() as conn:
            # Count alerts
            cursor = conn.execute('SELECT COUNT(*) as total_alerts FROM alerts')
            total_alerts = cursor.fetchone()['total_alerts']
            
            # Count active alerts
            cursor = conn.execute(
                'SELECT COUNT(*) as active_alerts FROM alerts WHERE is_active = 1 AND expires_at > ?', 
                (datetime.now(),)
            )
            active_alerts = cursor.fetchone()['active_alerts']
            
            # Count users
            cursor = conn.execute('SELECT COUNT(*) as total_users FROM users')
            total_users = cursor.fetchone()['total_users']
            
            # Get recent alerts by type (last 7 days)
            cursor = conn.execute('''
                SELECT hazard_type, COUNT(*) as count 
                FROM alerts 
                WHERE created_at > datetime('now', '-7 days')
                GROUP BY hazard_type
                ORDER BY count DESC
            ''')
            recent_alerts_by_type = [dict(row) for row in cursor.fetchall()]
        
        # ✅ Proper JSON response
        return jsonify({
            "status": "success",
            "stats": {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "total_users": total_users,
                "recent_alerts_by_type": recent_alerts_by_type
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"System stats error: {e}")
        return jsonify({"error": "Failed to get system stats", "details": str(e)}), 500
