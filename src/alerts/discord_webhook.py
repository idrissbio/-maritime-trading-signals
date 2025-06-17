import json
import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
try:
    from discord_webhook import DiscordWebhook as DiscordWebhookLib, DiscordEmbed
    DISCORD_WEBHOOK_AVAILABLE = True
except ImportError:
    DISCORD_WEBHOOK_AVAILABLE = False
    # Create dummy classes when discord-webhook is not available
    class DiscordEmbed:
        def __init__(self, *args, **kwargs):
            pass
        def add_embed_field(self, *args, **kwargs):
            pass
        def set_color(self, *args, **kwargs):
            pass
        def set_timestamp(self, *args, **kwargs):
            pass

from ..core.signal_generator import TradingSignal, SignalDirection, SignalTier

logger = logging.getLogger(__name__)

class DiscordWebhook:
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url
        self.webhook_available = DISCORD_WEBHOOK_AVAILABLE and bool(webhook_url)
        
        if not DISCORD_WEBHOOK_AVAILABLE:
            logger.warning("discord-webhook not available. Install with: pip install discord-webhook")
        elif not webhook_url:
            logger.warning("Discord webhook URL not configured")
        else:
            logger.info("Discord webhook initialized")

    def send_signal_alert(self, signal: TradingSignal, message: str) -> bool:
        """Send trading signal alert to Discord"""
        
        if not self.webhook_available:
            logger.warning("Discord webhook not available")
            return False
        
        try:
            webhook = DiscordWebhookLib(url=self.webhook_url)
            
            # Create rich embed for the signal
            embed = self._create_signal_embed(signal)
            webhook.add_embed(embed)
            
            response = webhook.execute()
            
            if response.status_code in [200, 204]:
                logger.info(f"Discord signal alert sent for {signal.symbol}")
                return True
            else:
                logger.error(f"Discord webhook failed with status {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Discord signal alert: {e}")
            return False

    def send_daily_summary(self, message: str, signals: List[TradingSignal]) -> bool:
        """Send daily summary to Discord"""
        
        if not self.webhook_available:
            return False
        
        try:
            webhook = DiscordWebhookLib(url=self.webhook_url)
            
            # Create summary embed
            embed = self._create_summary_embed(message, signals)
            webhook.add_embed(embed)
            
            response = webhook.execute()
            
            if response.status_code in [200, 204]:
                logger.info("Discord daily summary sent")
                return True
            else:
                logger.error(f"Discord webhook failed with status {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Discord daily summary: {e}")
            return False

    def send_system_alert(self, message: str, severity: str = "info") -> bool:
        """Send system alert to Discord"""
        
        if not self.webhook_available:
            return False
        
        try:
            webhook = DiscordWebhookLib(url=self.webhook_url)
            
            # Create system alert embed
            embed = self._create_system_alert_embed(message, severity)
            webhook.add_embed(embed)
            
            response = webhook.execute()
            
            if response.status_code in [200, 204]:
                logger.info(f"Discord system alert sent ({severity})")
                return True
            else:
                logger.error(f"Discord webhook failed with status {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Discord system alert: {e}")
            return False

    def send_test_message(self, message: str) -> bool:
        """Send test message to Discord"""
        
        if not self.webhook_available:
            return False
        
        try:
            webhook = DiscordWebhookLib(url=self.webhook_url)
            
            embed = DiscordEmbed(
                title="🔧 Test Message",
                description=message,
                color=0x00ff00  # Green
            )
            embed.set_footer(text=f"Maritime Trading System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            webhook.add_embed(embed)
            response = webhook.execute()
            
            return response.status_code in [200, 204]
        
        except Exception as e:
            logger.error(f"Failed to send Discord test message: {e}")
            return False

    def _create_signal_embed(self, signal: TradingSignal) -> DiscordEmbed:
        """Create rich Discord embed for trading signal"""
        
        # Color coding
        tier_colors = {1: 0xff0000, 2: 0xff8800, 3: 0x0088ff}  # Red, Orange, Blue
        direction_colors = {"LONG": 0x00ff00, "SHORT": 0xff0000}  # Green, Red
        
        color = tier_colors.get(signal.tier.value, 0x888888)
        
        # Title with emojis
        tier_emojis = {1: "🚨", 2: "⚡", 3: "📈"}
        direction_emojis = {"LONG": "📈", "SHORT": "📉"}
        
        title = f"{tier_emojis.get(signal.tier.value, '📊')} TIER {signal.tier.value} SIGNAL - {signal.symbol}"
        
        embed = DiscordEmbed(
            title=title,
            color=color,
            timestamp=signal.timestamp
        )
        
        # Direction field
        direction_text = f"{signal.direction.value} {direction_emojis.get(signal.direction.value, '📊')}"
        embed.add_embed_field(name="Direction", value=direction_text, inline=True)
        
        # Price levels
        embed.add_embed_field(name="Entry Price", value=f"${signal.entry_price:.2f}", inline=True)
        embed.add_embed_field(name="Stop Loss", value=f"${signal.stop_loss:.2f}", inline=True)
        
        embed.add_embed_field(name="Target 1", value=f"${signal.target_1:.2f}", inline=True)
        embed.add_embed_field(name="Target 2", value=f"${signal.target_2:.2f}", inline=True)
        embed.add_embed_field(name="Confidence", value=f"{signal.confidence_score*100:.0f}%", inline=True)
        
        # Risk/Reward
        risk_pct = abs((signal.stop_loss - signal.entry_price) / signal.entry_price * 100)
        reward_pct = abs((signal.target_1 - signal.entry_price) / signal.entry_price * 100)
        
        embed.add_embed_field(name="Risk", value=f"{risk_pct:.2f}%", inline=True)
        embed.add_embed_field(name="Reward (T1)", value=f"{reward_pct:.2f}%", inline=True)
        embed.add_embed_field(name="R:R Ratio", value=f"1:{signal.reward_ratio:.1f}", inline=True)
        
        # Scores
        scores_text = (
            f"Maritime: {signal.maritime_score*100:.0f}%\n"
            f"Volume: {signal.volume_score*100:.0f}%\n"
            f"Technical: {signal.technical_score*100:.0f}%"
        )
        embed.add_embed_field(name="Analysis Scores", value=scores_text, inline=False)
        
        # Reason
        embed.add_embed_field(name="Reason", value=signal.reason, inline=False)
        
        # Footer
        embed.set_footer(
            text=f"Expires: {signal.expiry_time.strftime('%Y-%m-%d %H:%M:%S')} | Maritime Trading System"
        )
        
        return embed

    def _create_summary_embed(self, message: str, signals: List[TradingSignal]) -> DiscordEmbed:
        """Create Discord embed for daily summary"""
        
        embed = DiscordEmbed(
            title="📊 Daily Trading Summary",
            description=f"Summary for {datetime.now().strftime('%Y-%m-%d')}",
            color=0x0088ff,
            timestamp=datetime.now()
        )
        
        if signals:
            # Summary stats
            total_signals = len(signals)
            tier_counts = {}
            symbol_counts = {}
            
            for signal in signals:
                tier_key = f"Tier {signal.tier.value}"
                tier_counts[tier_key] = tier_counts.get(tier_key, 0) + 1
                symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1
            
            avg_confidence = sum(s.confidence_score for s in signals) / len(signals)
            
            # Add fields
            embed.add_embed_field(name="Total Signals", value=str(total_signals), inline=True)
            embed.add_embed_field(name="Avg Confidence", value=f"{avg_confidence*100:.1f}%", inline=True)
            embed.add_embed_field(name="Top Symbol", value=max(symbol_counts, key=symbol_counts.get), inline=True)
            
            # Tier breakdown
            tier_text = "\n".join([f"{tier}: {count}" for tier, count in sorted(tier_counts.items())])
            embed.add_embed_field(name="By Tier", value=tier_text, inline=True)
            
            # Symbol breakdown
            symbol_text = "\n".join([f"{symbol}: {count}" for symbol, count in sorted(symbol_counts.items())])
            embed.add_embed_field(name="By Symbol", value=symbol_text, inline=True)
            
            # Recent signals (top 5)
            recent_signals_text = ""
            for i, signal in enumerate(signals[:5]):
                direction_emoji = "📈" if signal.direction == SignalDirection.LONG else "📉"
                recent_signals_text += f"{i+1}. {signal.symbol} {signal.direction.value} {direction_emoji} (T{signal.tier.value})\n"
            
            if recent_signals_text:
                embed.add_embed_field(name="Recent Signals", value=recent_signals_text, inline=False)
        
        else:
            embed.add_embed_field(name="Status", value="No signals generated today", inline=False)
        
        embed.set_footer(text="Maritime Trading System")
        
        return embed

    def _create_system_alert_embed(self, message: str, severity: str) -> DiscordEmbed:
        """Create Discord embed for system alerts"""
        
        severity_colors = {
            "error": 0xff0000,    # Red
            "warning": 0xff8800,  # Orange
            "info": 0x0088ff,     # Blue
            "success": 0x00ff00   # Green
        }
        
        severity_emojis = {
            "error": "🚨",
            "warning": "⚠️",
            "info": "ℹ️",
            "success": "✅"
        }
        
        color = severity_colors.get(severity, 0x888888)
        emoji = severity_emojis.get(severity, "📢")
        
        embed = DiscordEmbed(
            title=f"{emoji} System Alert - {severity.upper()}",
            description=message,
            color=color,
            timestamp=datetime.now()
        )
        
        embed.add_embed_field(name="Severity", value=severity.upper(), inline=True)
        embed.add_embed_field(name="Time", value=datetime.now().strftime('%H:%M:%S'), inline=True)
        
        embed.set_footer(text="Maritime Trading System")
        
        return embed

    def send_batch_alerts(self, signals: List[TradingSignal], max_embeds: int = 10) -> bool:
        """Send multiple signal alerts in a single message"""
        
        if not self.webhook_available or not signals:
            return False
        
        try:
            webhook = DiscordWebhookLib(url=self.webhook_url)
            
            # Discord has a limit of 10 embeds per message
            signals_to_send = signals[:max_embeds]
            
            for signal in signals_to_send:
                embed = self._create_signal_embed(signal)
                webhook.add_embed(embed)
            
            response = webhook.execute()
            
            if response.status_code in [200, 204]:
                logger.info(f"Discord batch alerts sent for {len(signals_to_send)} signals")
                return True
            else:
                logger.error(f"Discord batch webhook failed with status {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Discord batch alerts: {e}")
            return False

    def send_chart_image(self, image_path: str, signal: TradingSignal) -> bool:
        """Send chart image with signal information"""
        
        if not self.webhook_available:
            return False
        
        try:
            webhook = DiscordWebhookLib(url=self.webhook_url)
            
            # Add chart image
            with open(image_path, "rb") as f:
                webhook.add_file(file=f.read(), filename=f"chart_{signal.symbol}.png")
            
            # Add signal embed
            embed = self._create_signal_embed(signal)
            webhook.add_embed(embed)
            
            response = webhook.execute()
            
            return response.status_code in [200, 204]
        
        except Exception as e:
            logger.error(f"Failed to send chart image: {e}")
            return False

    def health_check(self) -> bool:
        """Check if Discord webhook is healthy"""
        
        if not self.webhook_available:
            return False
        
        try:
            # Send a minimal test request
            webhook = DiscordWebhookLib(url=self.webhook_url)
            
            embed = DiscordEmbed(
                title="Health Check",
                description="Testing webhook connectivity",
                color=0x00ff00
            )
            
            webhook.add_embed(embed)
            response = webhook.execute()
            
            return response.status_code in [200, 204]
        
        except Exception as e:
            logger.error(f"Discord health check failed: {e}")
            return False

    def get_webhook_info(self) -> Dict[str, Any]:
        """Get information about the webhook"""
        
        if not self.webhook_url:
            return {"status": "not_configured"}
        
        try:
            # Extract webhook ID from URL for basic info
            webhook_id = self.webhook_url.split('/')[-2] if '/' in self.webhook_url else "unknown"
            
            return {
                "status": "configured",
                "webhook_id": webhook_id,
                "webhook_available": self.webhook_available,
                "discord_lib_available": DISCORD_WEBHOOK_AVAILABLE
            }
        
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def send_performance_chart(self, chart_data: Dict[str, Any]) -> bool:
        """Send performance chart as embedded message"""
        
        if not self.webhook_available:
            return False
        
        try:
            webhook = DiscordWebhookLib(url=self.webhook_url)
            
            embed = DiscordEmbed(
                title="📊 Performance Chart",
                description="Trading performance metrics",
                color=0x0088ff,
                timestamp=datetime.now()
            )
            
            # Add performance metrics
            for metric, value in chart_data.items():
                if isinstance(value, (int, float)):
                    if metric.endswith('_pct'):
                        embed.add_embed_field(name=metric.replace('_', ' ').title(), 
                                            value=f"{value:.2f}%", inline=True)
                    else:
                        embed.add_embed_field(name=metric.replace('_', ' ').title(), 
                                            value=f"{value:.2f}", inline=True)
            
            webhook.add_embed(embed)
            response = webhook.execute()
            
            return response.status_code in [200, 204]
        
        except Exception as e:
            logger.error(f"Failed to send performance chart: {e}")
            return False