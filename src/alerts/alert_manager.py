import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
from .email_sender import EmailSender
from .sms_sender import SMSSender
from .discord_webhook import DiscordWebhook
from ..core.signal_generator import TradingSignal, SignalDirection, SignalTier

logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    email_enabled: bool = True
    sms_enabled: bool = True
    discord_enabled: bool = True
    email_recipients: List[str] = None
    sms_recipients: List[str] = None
    min_tier_for_sms: int = 2  # Only Tier 1 and 2 get SMS
    rate_limit_minutes: int = 5  # Minimum time between same alerts

class AlertManager:
    def __init__(self, config: AlertConfig, 
                 email_sender: EmailSender = None,
                 sms_sender: SMSSender = None,
                 discord_webhook: DiscordWebhook = None):
        
        self.config = config
        self.email_sender = email_sender
        self.sms_sender = sms_sender 
        self.discord_webhook = discord_webhook
        
        # Alert deduplication
        self.sent_alerts = {}  # Hash -> timestamp
        self.alert_history = []
        
        logger.info("AlertManager initialized")

    def send_signal_alert(self, signal: TradingSignal) -> Dict[str, bool]:
        """Send alert for a trading signal via all configured channels"""
        
        # Check for duplicate alerts
        alert_hash = self._get_alert_hash(signal)
        if self._is_duplicate_alert(alert_hash):
            logger.info(f"Skipping duplicate alert for {signal.symbol}")
            return {"duplicate": True}
        
        # Format the alert message
        formatted_message = self._format_signal_alert(signal)
        
        # Send results
        results = {
            "email": False,
            "sms": False,
            "discord": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send email alert
        if self.config.email_enabled and self.email_sender:
            try:
                success = self.email_sender.send_signal_alert(
                    signal=signal,
                    recipients=self.config.email_recipients,
                    message=formatted_message
                )
                results["email"] = success
                if success:
                    logger.info(f"Email alert sent for {signal.symbol}")
            except Exception as e:
                logger.error(f"Email alert failed: {e}")
        
        # Send SMS alert (only for higher tier signals)
        if (self.config.sms_enabled and self.sms_sender and 
            signal.tier.value <= self.config.min_tier_for_sms):
            try:
                success = self.sms_sender.send_signal_alert(
                    signal=signal,
                    recipients=self.config.sms_recipients,
                    message=self._format_sms_alert(signal)
                )
                results["sms"] = success
                if success:
                    logger.info(f"SMS alert sent for {signal.symbol}")
            except Exception as e:
                logger.error(f"SMS alert failed: {e}")
        
        # Send Discord alert
        if self.config.discord_enabled and self.discord_webhook:
            try:
                success = self.discord_webhook.send_signal_alert(
                    signal=signal,
                    message=formatted_message
                )
                results["discord"] = success
                if success:
                    logger.info(f"Discord alert sent for {signal.symbol}")
            except Exception as e:
                logger.error(f"Discord alert failed: {e}")
        
        # Record the alert
        self.sent_alerts[alert_hash] = datetime.now()
        self.alert_history.append({
            "signal": signal.to_dict(),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
        # Clean up old alerts
        self._cleanup_old_alerts()
        
        return results

    def _format_signal_alert(self, signal: TradingSignal) -> str:
        """Format a comprehensive signal alert message"""
        
        # Emoji mapping
        tier_emojis = {1: "🚨", 2: "⚡", 3: "📈"}
        direction_emojis = {
            SignalDirection.LONG: "📈",
            SignalDirection.SHORT: "📉"
        }
        
        tier_emoji = tier_emojis.get(signal.tier.value, "📊")
        direction_emoji = direction_emojis.get(signal.direction, "📊")
        
        # Calculate percentages
        stop_pct = abs((signal.stop_loss - signal.entry_price) / signal.entry_price * 100)
        target1_pct = abs((signal.target_1 - signal.entry_price) / signal.entry_price * 100)
        target2_pct = abs((signal.target_2 - signal.entry_price) / signal.entry_price * 100)
        
        message = f"""
{tier_emoji} TIER {signal.tier.value} SIGNAL - {signal.symbol}
Direction: {signal.direction.value} {direction_emoji}
Entry: ${signal.entry_price:.2f}
Stop Loss: ${signal.stop_loss:.2f} ({stop_pct:.2f}%)
Target 1: ${signal.target_1:.2f} (+{target1_pct:.2f}%)
Target 2: ${signal.target_2:.2f} (+{target2_pct:.2f}%)

Position Size: {signal.position_size} contracts
Risk: ${signal.risk_amount:.2f} ({self._get_risk_percentage(signal):.1f}% of account)

Reason: {signal.reason}
Confidence: {signal.confidence_score*100:.0f}%

Maritime Score: {signal.maritime_score*100:.0f}%
Volume Score: {signal.volume_score*100:.0f}%
Technical Score: {signal.technical_score*100:.0f}%

Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Expires: {signal.expiry_time.strftime('%Y-%m-%d %H:%M:%S')}

[Trade Now] [View Chart] [Skip Signal]
        """.strip()
        
        return message

    def _format_sms_alert(self, signal: TradingSignal) -> str:
        """Format a concise SMS alert message"""
        
        direction_symbol = "↗️" if signal.direction == SignalDirection.LONG else "↘️"
        
        message = f"""
🚨 T{signal.tier.value} {signal.symbol} {signal.direction.value} {direction_symbol}
Entry: ${signal.entry_price:.2f}
Stop: ${signal.stop_loss:.2f}
Target: ${signal.target_1:.2f}
Confidence: {signal.confidence_score*100:.0f}%
{signal.reason[:50]}...
        """.strip()
        
        return message

    def _get_alert_hash(self, signal: TradingSignal) -> str:
        """Generate hash for alert deduplication"""
        
        # Create hash from key signal components
        hash_data = f"{signal.symbol}_{signal.direction.value}_{signal.entry_price:.2f}_{signal.reason}"
        return hashlib.md5(hash_data.encode()).hexdigest()

    def _is_duplicate_alert(self, alert_hash: str) -> bool:
        """Check if this alert was recently sent"""
        
        if alert_hash not in self.sent_alerts:
            return False
        
        last_sent = self.sent_alerts[alert_hash]
        time_diff = datetime.now() - last_sent
        
        return time_diff.total_seconds() < (self.config.rate_limit_minutes * 60)

    def _get_risk_percentage(self, signal: TradingSignal) -> float:
        """Calculate risk as percentage of account (mock implementation)"""
        # This would normally calculate from actual account balance
        # For now, return a mock percentage
        return 1.0  # 1% risk per trade

    def _cleanup_old_alerts(self):
        """Remove old alerts from memory"""
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean sent_alerts dict
        old_hashes = [
            hash_key for hash_key, timestamp in self.sent_alerts.items()
            if timestamp < cutoff_time
        ]
        
        for hash_key in old_hashes:
            del self.sent_alerts[hash_key]
        
        # Keep only last 100 alerts in history
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

    def send_test_alert(self) -> Dict[str, bool]:
        """Send a test alert to verify all channels are working"""
        
        test_message = f"""
🔧 TEST ALERT - Maritime Trading System
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
All systems operational ✅
        """.strip()
        
        results = {
            "email": False,
            "sms": False,
            "discord": False
        }
        
        # Test email
        if self.config.email_enabled and self.email_sender:
            try:
                results["email"] = self.email_sender.send_test_email(
                    recipients=self.config.email_recipients,
                    message=test_message
                )
            except Exception as e:
                logger.error(f"Test email failed: {e}")
        
        # Test SMS
        if self.config.sms_enabled and self.sms_sender:
            try:
                results["sms"] = self.sms_sender.send_test_sms(
                    recipients=self.config.sms_recipients,
                    message="🔧 Maritime Trading System - Test SMS"
                )
            except Exception as e:
                logger.error(f"Test SMS failed: {e}")
        
        # Test Discord
        if self.config.discord_enabled and self.discord_webhook:
            try:
                results["discord"] = self.discord_webhook.send_test_message(
                    message=test_message
                )
            except Exception as e:
                logger.error(f"Test Discord failed: {e}")
        
        return results

    def send_daily_summary(self, signals: List[TradingSignal], 
                          performance_data: Dict[str, Any] = None) -> Dict[str, bool]:
        """Send daily summary of signals and performance"""
        
        if not signals:
            return {"no_signals": True}
        
        # Calculate summary stats
        total_signals = len(signals)
        by_tier = {}
        by_symbol = {}
        
        for signal in signals:
            tier_key = f"tier_{signal.tier.value}"
            by_tier[tier_key] = by_tier.get(tier_key, 0) + 1
            by_symbol[signal.symbol] = by_symbol.get(signal.symbol, 0) + 1
        
        avg_confidence = sum(s.confidence_score for s in signals) / len(signals)
        
        # Format summary message
        summary_message = f"""
📊 DAILY TRADING SUMMARY - {datetime.now().strftime('%Y-%m-%d')}

🎯 SIGNALS GENERATED: {total_signals}
"""
        
        # Add tier breakdown
        for tier, count in sorted(by_tier.items()):
            summary_message += f"   {tier.upper()}: {count} signals\n"
        
        summary_message += f"\n💪 AVERAGE CONFIDENCE: {avg_confidence*100:.1f}%\n"
        
        # Add symbol breakdown
        summary_message += "\n📈 BY SYMBOL:\n"
        for symbol, count in sorted(by_symbol.items()):
            summary_message += f"   {symbol}: {count} signals\n"
        
        # Add performance data if available
        if performance_data:
            summary_message += f"""
📊 PERFORMANCE:
   Win Rate: {performance_data.get('win_rate', 0)*100:.1f}%
   Total P&L: ${performance_data.get('total_pnl', 0):.2f}
   Best Trade: ${performance_data.get('best_trade', 0):.2f}
   Worst Trade: ${performance_data.get('worst_trade', 0):.2f}
"""
        
        summary_message += f"\n⏰ Generated: {datetime.now().strftime('%H:%M:%S')}"
        
        # Send via configured channels
        results = {
            "email": False,
            "discord": False
        }
        
        # Email summary
        if self.config.email_enabled and self.email_sender:
            try:
                results["email"] = self.email_sender.send_daily_summary(
                    recipients=self.config.email_recipients,
                    message=summary_message,
                    signals=signals
                )
            except Exception as e:
                logger.error(f"Daily summary email failed: {e}")
        
        # Discord summary
        if self.config.discord_enabled and self.discord_webhook:
            try:
                results["discord"] = self.discord_webhook.send_daily_summary(
                    message=summary_message,
                    signals=signals
                )
            except Exception as e:
                logger.error(f"Daily summary Discord failed: {e}")
        
        return results

    def send_system_alert(self, alert_type: str, message: str, 
                         severity: str = "info") -> Dict[str, bool]:
        """Send system alerts (errors, warnings, etc.)"""
        
        severity_emojis = {
            "error": "🚨",
            "warning": "⚠️", 
            "info": "ℹ️",
            "success": "✅"
        }
        
        emoji = severity_emojis.get(severity, "📢")
        
        formatted_message = f"""
{emoji} SYSTEM ALERT - {alert_type.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Severity: {severity.upper()}

{message}
        """.strip()
        
        results = {
            "email": False,
            "discord": False
        }
        
        # Send system alerts via email and Discord (not SMS to avoid spam)
        if self.email_sender:
            try:
                results["email"] = self.email_sender.send_system_alert(
                    recipients=self.config.email_recipients,
                    subject=f"Maritime Trading System - {alert_type}",
                    message=formatted_message
                )
            except Exception as e:
                logger.error(f"System alert email failed: {e}")
        
        if self.discord_webhook:
            try:
                results["discord"] = self.discord_webhook.send_system_alert(
                    message=formatted_message,
                    severity=severity
                )
            except Exception as e:
                logger.error(f"System alert Discord failed: {e}")
        
        return results

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about sent alerts"""
        
        if not self.alert_history:
            return {"total_alerts": 0}
        
        total_alerts = len(self.alert_history)
        
        # Count by channel
        email_success = sum(1 for alert in self.alert_history if alert["results"].get("email", False))
        sms_success = sum(1 for alert in self.alert_history if alert["results"].get("sms", False))
        discord_success = sum(1 for alert in self.alert_history if alert["results"].get("discord", False))
        
        # Count by signal tier
        tier_counts = {}
        for alert in self.alert_history:
            signal_data = alert["signal"]
            tier = signal_data.get("tier", 0)
            tier_counts[f"tier_{tier}"] = tier_counts.get(f"tier_{tier}", 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "channel_success": {
                "email": email_success,
                "sms": sms_success,
                "discord": discord_success
            },
            "success_rates": {
                "email": email_success / total_alerts if total_alerts > 0 else 0,
                "sms": sms_success / total_alerts if total_alerts > 0 else 0,
                "discord": discord_success / total_alerts if total_alerts > 0 else 0
            },
            "by_tier": tier_counts,
            "last_24h": len([a for a in self.alert_history 
                           if datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(hours=24)])
        }

    def health_check(self) -> Dict[str, Any]:
        """Check health of all alert channels"""
        
        health_status = {
            "email": {"enabled": self.config.email_enabled, "healthy": False},
            "sms": {"enabled": self.config.sms_enabled, "healthy": False},
            "discord": {"enabled": self.config.discord_enabled, "healthy": False},
            "overall_health": "unknown"
        }
        
        # Check email health
        if self.config.email_enabled and self.email_sender:
            try:
                health_status["email"]["healthy"] = self.email_sender.health_check()
            except Exception as e:
                logger.error(f"Email health check failed: {e}")
        
        # Check SMS health
        if self.config.sms_enabled and self.sms_sender:
            try:
                health_status["sms"]["healthy"] = self.sms_sender.health_check()
            except Exception as e:
                logger.error(f"SMS health check failed: {e}")
        
        # Check Discord health
        if self.config.discord_enabled and self.discord_webhook:
            try:
                health_status["discord"]["healthy"] = self.discord_webhook.health_check()
            except Exception as e:
                logger.error(f"Discord health check failed: {e}")
        
        # Overall health
        enabled_channels = [
            health_status["email"]["healthy"] if health_status["email"]["enabled"] else True,
            health_status["sms"]["healthy"] if health_status["sms"]["enabled"] else True,
            health_status["discord"]["healthy"] if health_status["discord"]["enabled"] else True
        ]
        
        if all(enabled_channels):
            health_status["overall_health"] = "healthy"
        elif any(enabled_channels):
            health_status["overall_health"] = "degraded"
        else:
            health_status["overall_health"] = "unhealthy"
        
        return health_status