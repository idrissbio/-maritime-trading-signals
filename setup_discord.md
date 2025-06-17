# Discord Webhook Setup Guide

## Step 1: Create Discord Server (if needed)
1. Open Discord app or web version
2. Click "+" to add server
3. Choose "Create My Own"
4. Name it "Maritime Trading Signals"

## Step 2: Create Webhook
1. Right-click your server name
2. Select "Server Settings"
3. Click "Integrations" in left menu
4. Click "Webhooks"
5. Click "New Webhook"
6. Name it "Trading Alerts"
7. Choose channel (e.g., #trading-signals)
8. Copy the webhook URL

## Step 3: Test Webhook
Run this command to test:
```bash
python test_discord.py
```

## Webhook URL Format:
```
https://discord.com/api/webhooks/123456789/abcdefghijk...
```

Save this URL for your .env file!