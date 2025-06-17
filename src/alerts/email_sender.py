import os
import logging
from typing import List, Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

from ..core.signal_generator import TradingSignal

logger = logging.getLogger(__name__)

class EmailSender:
    def __init__(self, api_key: str = None, from_email: str = None, 
                 use_sendgrid: bool = True, smtp_config: dict = None):
        
        self.api_key = api_key
        self.from_email = from_email
        self.use_sendgrid = use_sendgrid and SENDGRID_AVAILABLE
        self.smtp_config = smtp_config or {}
        
        if self.use_sendgrid and not api_key:
            logger.warning("SendGrid API key not provided, falling back to SMTP")
            self.use_sendgrid = False
        
        if self.use_sendgrid:
            self.sg_client = SendGridAPIClient(api_key=self.api_key)
        else:
            # Default SMTP config for Gmail
            self.smtp_config = {
                "server": self.smtp_config.get("server", "smtp.gmail.com"),
                "port": self.smtp_config.get("port", 587),
                "username": self.smtp_config.get("username", from_email),
                "password": self.smtp_config.get("password"),
                "use_tls": self.smtp_config.get("use_tls", True)
            }
        
        logger.info(f"EmailSender initialized - Using {'SendGrid' if self.use_sendgrid else 'SMTP'}")

    def send_signal_alert(self, signal: TradingSignal, recipients: List[str], 
                         message: str) -> bool:
        """Send trading signal alert via email"""
        
        if not recipients:
            logger.warning("No email recipients configured")
            return False
        
        subject = f"🚨 Maritime Trading Signal - {signal.symbol} {signal.direction.value}"
        
        # Create HTML version of the email
        html_content = self._create_signal_html(signal, message)
        
        try:
            if self.use_sendgrid:
                return self._send_via_sendgrid(recipients, subject, message, html_content)
            else:
                return self._send_via_smtp(recipients, subject, message, html_content)
        
        except Exception as e:
            logger.error(f"Failed to send signal alert email: {e}")
            return False

    def send_daily_summary(self, recipients: List[str], message: str, 
                          signals: List[TradingSignal]) -> bool:
        """Send daily summary email"""
        
        if not recipients:
            return False
        
        subject = f"📊 Daily Trading Summary - {datetime.now().strftime('%Y-%m-%d')}"
        html_content = self._create_summary_html(message, signals)
        
        try:
            if self.use_sendgrid:
                return self._send_via_sendgrid(recipients, subject, message, html_content)
            else:
                return self._send_via_smtp(recipients, subject, message, html_content)
        
        except Exception as e:
            logger.error(f"Failed to send daily summary email: {e}")
            return False

    def send_system_alert(self, recipients: List[str], subject: str, message: str) -> bool:
        """Send system alert email"""
        
        if not recipients:
            return False
        
        try:
            if self.use_sendgrid:
                return self._send_via_sendgrid(recipients, subject, message)
            else:
                return self._send_via_smtp(recipients, subject, message)
        
        except Exception as e:
            logger.error(f"Failed to send system alert email: {e}")
            return False

    def send_test_email(self, recipients: List[str], message: str) -> bool:
        """Send test email"""
        
        subject = "🔧 Maritime Trading System - Test Email"
        
        try:
            if self.use_sendgrid:
                return self._send_via_sendgrid(recipients, subject, message)
            else:
                return self._send_via_smtp(recipients, subject, message)
        
        except Exception as e:
            logger.error(f"Failed to send test email: {e}")
            return False

    def _send_via_sendgrid(self, recipients: List[str], subject: str, 
                          text_content: str, html_content: str = None) -> bool:
        """Send email via SendGrid API"""
        
        try:
            mail = Mail(
                from_email=self.from_email,
                to_emails=recipients,
                subject=subject,
                plain_text_content=text_content,
                html_content=html_content or text_content
            )
            
            response = self.sg_client.send(mail)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Email sent successfully via SendGrid to {len(recipients)} recipients")
                return True
            else:
                logger.error(f"SendGrid API returned status {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"SendGrid send failed: {e}")
            return False

    def _send_via_smtp(self, recipients: List[str], subject: str, 
                      text_content: str, html_content: str = None) -> bool:
        """Send email via SMTP"""
        
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = ", ".join(recipients)
            
            # Add text version
            text_part = MIMEText(text_content, "plain")
            msg.attach(text_part)
            
            # Add HTML version if provided
            if html_content:
                html_part = MIMEText(html_content, "html")
                msg.attach(html_part)
            
            # Connect to server and send
            server = smtplib.SMTP(self.smtp_config["server"], self.smtp_config["port"])
            
            if self.smtp_config["use_tls"]:
                server.starttls()
            
            if self.smtp_config.get("password"):
                server.login(self.smtp_config["username"], self.smtp_config["password"])
            
            server.sendmail(self.from_email, recipients, msg.as_string())
            server.quit()
            
            logger.info(f"Email sent successfully via SMTP to {len(recipients)} recipients")
            return True
        
        except Exception as e:
            logger.error(f"SMTP send failed: {e}")
            return False

    def _create_signal_html(self, signal: TradingSignal, text_message: str) -> str:
        """Create HTML version of signal alert"""
        
        # Color coding based on direction and tier
        direction_color = "#28a745" if signal.direction.value == "LONG" else "#dc3545"
        tier_color = {"1": "#dc3545", "2": "#ffc107", "3": "#17a2b8"}[str(signal.tier.value)]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Maritime Trading Signal</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; color: {tier_color}; border-bottom: 2px solid {tier_color}; padding-bottom: 10px; margin-bottom: 20px; }}
                .signal-info {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0; }}
                .direction {{ color: {direction_color}; font-weight: bold; font-size: 1.2em; }}
                .price-levels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 15px 0; }}
                .price-box {{ background: #e9ecef; padding: 10px; border-radius: 5px; text-align: center; }}
                .reason {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin: 15px 0; }}
                .confidence {{ text-align: center; font-size: 1.1em; margin: 15px 0; }}
                .footer {{ text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 20px; border-top: 1px solid #dee2e6; padding-top: 15px; }}
                .button {{ display: inline-block; padding: 10px 20px; background: {direction_color}; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 TIER {signal.tier.value} MARITIME SIGNAL</h1>
                    <h2>{signal.symbol}</h2>
                </div>
                
                <div class="signal-info">
                    <div class="direction">Direction: {signal.direction.value} 
                        {'📈' if signal.direction.value == 'LONG' else '📉'}
                    </div>
                </div>
                
                <div class="price-levels">
                    <div class="price-box">
                        <strong>Entry Price</strong><br>
                        ${signal.entry_price:.2f}
                    </div>
                    <div class="price-box">
                        <strong>Stop Loss</strong><br>
                        ${signal.stop_loss:.2f}
                    </div>
                    <div class="price-box">
                        <strong>Target 1</strong><br>
                        ${signal.target_1:.2f}
                    </div>
                    <div class="price-box">
                        <strong>Target 2</strong><br>
                        ${signal.target_2:.2f}
                    </div>
                </div>
                
                <div class="reason">
                    <strong>Reason:</strong> {signal.reason}
                </div>
                
                <div class="confidence">
                    <strong>Confidence: {signal.confidence_score*100:.0f}%</strong>
                </div>
                
                <div style="text-align: center; margin: 20px 0;">
                    <a href="#" class="button">📊 View Chart</a>
                    <a href="#" class="button">💰 Trade Now</a>
                    <a href="#" class="button">❌ Skip Signal</a>
                </div>
                
                <div class="footer">
                    Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br>
                    Expires: {signal.expiry_time.strftime('%Y-%m-%d %H:%M:%S')}<br>
                    Maritime Trading Signals System
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

    def _create_summary_html(self, text_message: str, signals: List[TradingSignal]) -> str:
        """Create HTML version of daily summary"""
        
        # Generate signals table
        signals_html = ""
        for signal in signals[:10]:  # Show top 10 signals
            color = "#28a745" if signal.direction.value == "LONG" else "#dc3545"
            signals_html += f"""
            <tr>
                <td>{signal.symbol}</td>
                <td style="color: {color}; font-weight: bold;">{signal.direction.value}</td>
                <td>T{signal.tier.value}</td>
                <td>${signal.entry_price:.2f}</td>
                <td>{signal.confidence_score*100:.0f}%</td>
                <td>{signal.reason[:50]}...</td>
            </tr>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Daily Trading Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; color: #007bff; border-bottom: 2px solid #007bff; padding-bottom: 10px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
                th {{ background-color: #e9ecef; }}
                .footer {{ text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 20px; border-top: 1px solid #dee2e6; padding-top: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Daily Trading Summary</h1>
                    <h2>{datetime.now().strftime('%Y-%m-%d')}</h2>
                </div>
                
                <div style="white-space: pre-line; margin: 20px 0;">
                    {text_message}
                </div>
                
                <h3>Recent Signals</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Direction</th>
                            <th>Tier</th>
                            <th>Entry</th>
                            <th>Confidence</th>
                            <th>Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {signals_html}
                    </tbody>
                </table>
                
                <div class="footer">
                    Maritime Trading Signals System<br>
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

    def health_check(self) -> bool:
        """Check if email service is healthy"""
        
        try:
            if self.use_sendgrid:
                # Test SendGrid connection
                # This is a simplified check - in production you might want to send a test email
                return bool(self.api_key and self.from_email)
            else:
                # Test SMTP connection
                server = smtplib.SMTP(self.smtp_config["server"], self.smtp_config["port"])
                if self.smtp_config["use_tls"]:
                    server.starttls()
                server.quit()
                return True
        
        except Exception as e:
            logger.error(f"Email health check failed: {e}")
            return False