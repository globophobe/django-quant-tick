"""Management command for viewing and resolving trend alerts."""
import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from django.utils import timezone

from quant_tick.lib.trend_scanning import detect_break
from quant_tick.models import MLConfig, TrendAlert, TrendScan

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Manage trend alerts."""

    help = "View and resolve trend alerts."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--config-code-name", type=str, help="Filter by config code name")
        parser.add_argument("--list", action="store_true", help="List all active alerts")
        parser.add_argument("--resolve", type=int, help="Resolve alert by ID")
        parser.add_argument("--check-break", action="store_true", help="Check for structural breaks")
        parser.add_argument("--threshold", type=float, default=1.5, help="Break threshold")

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        cfg_code = options.get("config_code_name")
        list_alerts = options["list"]
        resolve_id = options.get("resolve")
        check_break = options["check_break"]
        threshold = options["threshold"]

        if list_alerts:
            self._list_alerts(cfg_code)
        elif resolve_id:
            self._resolve_alert(resolve_id)
        elif check_break:
            self._check_break(cfg_code, threshold)
        else:
            logger.info("No action specified. Use --list, --resolve, or --check-break")

    def _list_alerts(self, cfg_code: str | None = None) -> None:
        """List all active alerts."""
        alerts = TrendAlert.objects.filter(status="active")

        if cfg_code:
            alerts = alerts.filter(ml_config__code_name=cfg_code)

        alerts = alerts.order_by("-timestamp")

        if not alerts.exists():
            logger.info("No active alerts")
            return

        logger.info(f"Found {alerts.count()} active alerts:")
        for alert in alerts:
            logger.info(
                f"  [ID={alert.id}] {alert.ml_config.code_name} @ {alert.timestamp} - "
                f"score={alert.current_top_score}, threshold={alert.threshold}, "
                f"action={alert.action}, message={alert.message}"
            )

    def _resolve_alert(self, alert_id: int) -> None:
        """Resolve an alert by ID."""
        try:
            alert = TrendAlert.objects.get(id=alert_id)
        except TrendAlert.DoesNotExist:
            logger.error(f"Alert {alert_id} not found")
            return

        alert.status = "resolved"
        alert.resolved_at = timezone.now()
        alert.save(update_fields=["status", "resolved_at"])

        logger.info(f"Resolved alert {alert_id} for {alert.ml_config.code_name}")

        if alert.action == "pause_trading":
            cfg = alert.ml_config
            if not cfg.is_active:
                cfg.is_active = True
                cfg.save(update_fields=["is_active"])
                logger.info(f"Resumed trading for {cfg.code_name}")

    def _check_break(self, cfg_code: str | None, threshold: float) -> None:
        """Check for structural breaks in recent trend scans."""
        if not cfg_code:
            logger.error("--config-code-name required for --check-break")
            return

        try:
            cfg = MLConfig.objects.get(code_name=cfg_code)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig {cfg_code} not found")
            return

        recent_scans = TrendScan.objects.filter(
            ml_config=cfg,
            returns_type="realized_pnl"
        ).order_by("-timestamp")[:2]

        if recent_scans.count() < 2:
            logger.info(f"{cfg}: insufficient trend scans for break detection ({recent_scans.count()}/2)")
            return

        current_scan_records = TrendScan.objects.filter(
            ml_config=cfg,
            timestamp=recent_scans[0].timestamp
        ).order_by("-score")

        previous_scan_records = TrendScan.objects.filter(
            ml_config=cfg,
            timestamp=recent_scans[1].timestamp
        ).order_by("-score")

        current_scan = [
            {
                'window': {'size': s.window_size},
                'statistic': {'score': float(s.score)},
                'n_events': s.n_events
            }
            for s in current_scan_records
        ]
        previous_scan = [
            {
                'window': {'size': s.window_size},
                'statistic': {'score': float(s.score)},
                'n_events': s.n_events
            }
            for s in previous_scan_records
        ]

        break_info = detect_break(current_scan, previous_scan, threshold, top_k=5)

        logger.info(f"{cfg}: break detection result:")
        logger.info(f"  Break detected: {break_info['break_detected']}")
        logger.info(f"  Current top score: {break_info['current_top_score']:.3f}")
        logger.info(f"  Previous top score: {break_info.get('previous_top_score', 'N/A')}")
        logger.info(f"  Deterioration: {break_info.get('deterioration', 'N/A')}")
        logger.info(f"  Message: {break_info['message']}")

        if break_info['break_detected']:
            logger.warning(f"{cfg}: STRUCTURAL BREAK DETECTED!")
