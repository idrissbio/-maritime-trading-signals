import logging
from typing import List, Optional
from datetime import datetime
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

from ..core.signal_generator import TradingSignal

logger = logging.getLogger(__name__)

class SMSSender:
    def __init__(self, account_sid: str = None, auth_token: str = None, 
                 from_number: str = None):
        
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        
        if not TWILIO_AVAILABLE:
            logger.error("Twilio not available. Install with: pip install twilio")
            self.client = None
            return
        
        if not all([account_sid, auth_token, from_number]):
            logger.warning("Twilio credentials not fully configured")
            self.client = None
            return
        
        try:
            self.client = Client(account_sid, auth_token)
            logger.info("SMSSender initialized with Twilio")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
            self.client = None

    def send_signal_alert(self, signal: TradingSignal, recipients: List[str], 
                         message: str) -> bool:
        """Send trading signal alert via SMS"""
        
        if not self.client or not recipients:
            logger.warning("SMS not configured or no recipients")
            return False
        
        # Limit SMS length (160 characters standard)
        if len(message) > 160:
            message = message[:157] + "..."
        
        success_count = 0
        
        for recipient in recipients:
            try:
                # Clean phone number
                phone_number = self._clean_phone_number(recipient)
                if not phone_number:
                    logger.warning(f"Invalid phone number: {recipient}")
                    continue
                
                # Send SMS
                sms = self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=phone_number
                )
                
                if sms.sid:
                    success_count += 1
                    logger.info(f"SMS sent successfully to {phone_number} (SID: {sms.sid})")
                else:
                    logger.error(f"Failed to send SMS to {phone_number}")
                
            except TwilioException as e:
                logger.error(f"Twilio error sending to {recipient}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error sending SMS to {recipient}: {e}")
        
        return success_count > 0

    def send_test_sms(self, recipients: List[str], message: str) -> bool:
        """Send test SMS"""
        
        if not self.client or not recipients:
            return False
        
        test_message = f"🔧 Maritime Trading Test SMS - {datetime.now().strftime('%H:%M:%S')}"
        
        return self.send_signal_alert(
            signal=None,  # Not used for test
            recipients=recipients,
            message=test_message
        )

    def send_urgent_alert(self, recipients: List[str], message: str) -> bool:
        """Send urgent system alert via SMS"""
        
        if not self.client or not recipients:
            return False
        
        urgent_message = f"🚨 URGENT: {message}"
        
        # Truncate if too long
        if len(urgent_message) > 160:
            urgent_message = urgent_message[:157] + "..."
        
        return self.send_signal_alert(
            signal=None,
            recipients=recipients,
            message=urgent_message
        )

    def _clean_phone_number(self, phone_number: str) -> Optional[str]:
        """Clean and validate phone number"""
        
        if not phone_number:
            return None
        
        # Remove all non-digit characters except +
        cleaned = ''.join(c for c in phone_number if c.isdigit() or c == '+')
        
        # Ensure it starts with + for international format
        if not cleaned.startswith('+'):
            # Assume US number if no country code
            if len(cleaned) == 10:
                cleaned = '+1' + cleaned
            elif len(cleaned) == 11 and cleaned.startswith('1'):
                cleaned = '+' + cleaned
            else:
                # Try to add + if it looks like an international number
                cleaned = '+' + cleaned
        
        # Basic validation
        if len(cleaned) < 10 or len(cleaned) > 15:
            return None
        
        return cleaned

    def get_account_info(self) -> dict:
        """Get Twilio account information"""
        
        if not self.client:
            return {"status": "not_configured"}
        
        try:
            account = self.client.api.accounts(self.account_sid).fetch()
            return {
                "status": "active",
                "account_sid": account.sid,
                "friendly_name": account.friendly_name,
                "type": account.type,
                "status": account.status
            }
        except Exception as e:
            logger.error(f"Failed to fetch account info: {e}")
            return {"status": "error", "error": str(e)}

    def get_message_history(self, limit: int = 20) -> List[dict]:
        """Get recent SMS message history"""
        
        if not self.client:
            return []
        
        try:
            messages = self.client.messages.list(limit=limit)
            
            history = []
            for msg in messages:
                history.append({
                    "sid": msg.sid,
                    "to": msg.to,
                    "from": msg.from_,
                    "body": msg.body,
                    "status": msg.status,
                    "date_sent": msg.date_sent.isoformat() if msg.date_sent else None,
                    "error_code": msg.error_code,
                    "error_message": msg.error_message
                })
            
            return history
        
        except Exception as e:
            logger.error(f"Failed to fetch message history: {e}")
            return []

    def check_delivery_status(self, message_sid: str) -> dict:
        """Check delivery status of a specific message"""
        
        if not self.client:
            return {"status": "client_not_available"}
        
        try:
            message = self.client.messages(message_sid).fetch()
            
            return {
                "sid": message.sid,
                "status": message.status,
                "error_code": message.error_code,
                "error_message": message.error_message,
                "date_sent": message.date_sent.isoformat() if message.date_sent else None,
                "date_updated": message.date_updated.isoformat() if message.date_updated else None
            }
        
        except Exception as e:
            logger.error(f"Failed to check delivery status for {message_sid}: {e}")
            return {"status": "error", "error": str(e)}

    def health_check(self) -> bool:
        """Check if SMS service is healthy"""
        
        if not self.client:
            return False
        
        try:
            # Test by fetching account info
            account = self.client.api.accounts(self.account_sid).fetch()
            return account.status.lower() == "active"
        
        except Exception as e:
            logger.error(f"SMS health check failed: {e}")
            return False

    def estimate_cost(self, message_length: int, recipient_count: int) -> dict:
        """Estimate SMS cost based on message length and recipient count"""
        
        # Twilio pricing (approximate, varies by region)
        base_cost_per_sms = 0.0075  # $0.0075 per SMS in US
        
        # Calculate number of SMS segments (160 chars per segment)
        segments = max(1, (message_length + 159) // 160)
        
        cost_per_recipient = base_cost_per_sms * segments
        total_cost = cost_per_recipient * recipient_count
        
        return {
            "message_length": message_length,
            "segments_per_message": segments,
            "recipient_count": recipient_count,
            "cost_per_recipient": cost_per_recipient,
            "total_estimated_cost": total_cost,
            "currency": "USD"
        }

    def format_signal_for_sms(self, signal: TradingSignal) -> str:
        """Format trading signal specifically for SMS constraints"""
        
        direction_emoji = "📈" if signal.direction.value == "LONG" else "📉"
        
        # Ultra-compact format for SMS
        message = (
            f"🚨T{signal.tier.value} {signal.symbol} {signal.direction.value} {direction_emoji}\n"
            f"Entry: ${signal.entry_price:.2f}\n"
            f"Stop: ${signal.stop_loss:.2f}\n"
            f"Target: ${signal.target_1:.2f}\n"
            f"Conf: {signal.confidence_score*100:.0f}%"
        )
        
        return message

    def send_bulk_sms(self, recipients: List[str], message: str, 
                     delay_seconds: int = 1) -> dict:
        """Send SMS to multiple recipients with rate limiting"""
        
        if not self.client or not recipients:
            return {"success": False, "sent": 0, "failed": 0}
        
        import time
        
        sent_count = 0
        failed_count = 0
        results = []
        
        for i, recipient in enumerate(recipients):
            try:
                # Rate limiting delay
                if i > 0 and delay_seconds > 0:
                    time.sleep(delay_seconds)
                
                phone_number = self._clean_phone_number(recipient)
                if not phone_number:
                    failed_count += 1
                    results.append({"recipient": recipient, "status": "invalid_number"})
                    continue
                
                sms = self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=phone_number
                )
                
                if sms.sid:
                    sent_count += 1
                    results.append({"recipient": recipient, "status": "sent", "sid": sms.sid})
                else:
                    failed_count += 1
                    results.append({"recipient": recipient, "status": "failed"})
            
            except Exception as e:
                failed_count += 1
                results.append({"recipient": recipient, "status": "error", "error": str(e)})
        
        return {
            "success": sent_count > 0,
            "sent": sent_count,
            "failed": failed_count,
            "total": len(recipients),
            "results": results
        }