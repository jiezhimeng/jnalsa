"""Sanitize local paths in exported error logs."""
import re


def handle_sensitive_image(image):
    # No Jiangnan account-panel detector has been validated. Keep the original
    # screenshot for local diagnosis; review images before public sharing.
    return image


def handle_sensitive_text(text):
    return re.sub(r'(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\r\n]+',
                  lambda match: 'C:/Users/<user>', text)


def handle_sensitive_logs(logs):
    return [handle_sensitive_text(line) for line in logs]
