"""Offline controller tests; no platform requests or real waiting."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import elastic_pool as pool


class ScaleUpConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.json"
        self.args = SimpleNamespace(apply=True, cooldown=120, interval=60)
        self.api = Mock()
        self.deployment = {"Replicas": 2, "Status": "Running",
                           "UpdateTimestamp": "unchanged"}
        self.api.deployment.return_value = self.deployment
        self.controller = pool.Controller(self.args, self.api, self.path)

    def tick(self, seconds, *, replicas=2, idle=8, jobs=0, gpus=0,
             stable=True, releasing=0, ready=None, error=None):
        sample = pool.Snapshot(replicas, idle, jobs, gpus,
                               max(0, replicas - 1) * 32,
                               replicas * 4 if ready is None else ready,
                               replicas * 4, releasing, stable)
        with patch.object(pool, "snapshot", return_value=(sample, self.deployment),
                          side_effect=error), \
                patch.object(pool.time, "monotonic", return_value=seconds), \
                patch.object(pool.time, "time", return_value=1000 + seconds):
            self.controller.tick()

    def test_waits_full_ten_minutes_and_keeps_original_target(self):
        self.tick(0)
        for seconds in (*range(60, 600, 60), 599):
            self.tick(seconds, idle=16)
            self.api.scale.assert_not_called()
        self.tick(600, idle=16)
        self.api.scale.assert_called_once_with(self.args, 4)
        self.assertEqual(json.loads(self.path.read_text())["pending"]["target"], 4)
        self.assertIsNone(self.controller.scale_up_candidate)
        self.tick(660, idle=16)
        self.api.scale.assert_called_once()

    def test_insufficient_nodes_restart_with_smaller_plan(self):
        self.tick(0)
        self.tick(240, idle=4)
        self.assertEqual(self.controller.scale_up_candidate["target"], 3)
        self.tick(600, idle=8)
        self.api.scale.assert_not_called()
        self.tick(840, idle=8)
        self.api.scale.assert_called_once_with(self.args, 3)

    def test_fewer_nodes_can_still_support_fixed_plan(self):
        self.tick(0, idle=7)
        for seconds in range(60, 601, 60):
            self.tick(seconds, idle=4)
        self.api.scale.assert_called_once_with(self.args, 3)

    def test_too_few_nodes_return_to_monitoring_without_new_plan(self):
        self.tick(0)
        self.tick(240, idle=3)
        self.assertIsNone(self.controller.scale_up_candidate)
        self.tick(600)
        self.tick(840)
        self.api.scale.assert_not_called()
        self.tick(1200)
        self.api.scale.assert_called_once_with(self.args, 4)

    def test_queue_even_when_covered_cancels_confirmation(self):
        self.tick(0)
        self.tick(240, jobs=1, gpus=8)
        self.assertIsNone(self.controller.scale_up_candidate)
        self.tick(600)
        self.tick(840)
        self.api.scale.assert_not_called()
        self.tick(1200)
        self.api.scale.assert_called_once_with(self.args, 4)

    def test_new_queue_shortfall_scales_down_immediately(self):
        self.tick(0)
        self.tick(60, idle=0, jobs=1, gpus=32)
        self.api.scale.assert_called_once_with(self.args, 1)
        self.assertIsNone(self.controller.scale_up_candidate)

    def test_unstable_deployment_cancels_confirmation(self):
        self.tick(0)
        self.tick(240, stable=False, ready=4)
        self.assertIsNone(self.controller.scale_up_candidate)
        self.tick(600)
        self.tick(840)
        self.api.scale.assert_not_called()

    def test_releases_do_not_interrupt_existing_confirmation(self):
        self.tick(0)
        for seconds in range(60, 600, 60):
            self.tick(seconds, stable=False, releasing=32)
            self.api.scale.assert_not_called()
        self.tick(600, stable=False, releasing=32)
        self.api.scale.assert_called_once_with(self.args, 4)

    def test_releases_prevent_starting_confirmation(self):
        self.tick(0, stable=False, releasing=32)
        self.tick(600, stable=False, releasing=32)
        self.assertIsNone(self.controller.scale_up_candidate)
        self.api.scale.assert_not_called()
        self.tick(660)
        self.tick(1260)
        self.api.scale.assert_called_once_with(self.args, 4)

    def test_capacity_loss_during_releases_cancels_without_new_plan(self):
        self.tick(0)
        self.tick(240, idle=4, stable=False, releasing=32)
        self.assertIsNone(self.controller.scale_up_candidate)
        self.tick(600, stable=False, releasing=32)
        self.api.scale.assert_not_called()

    def test_nonrunning_deployment_during_releases_cancels_confirmation(self):
        self.tick(0)
        self.deployment["Status"] = "Updating"
        self.tick(240, stable=False, releasing=32)
        self.assertIsNone(self.controller.scale_up_candidate)
        self.api.scale.assert_not_called()

    def test_sampling_error_cancels_confirmation(self):
        self.tick(0)
        with self.assertRaisesRegex(ValueError, "bad sample"):
            self.tick(240, error=ValueError("bad sample"))
        self.assertIsNone(self.controller.scale_up_candidate)
        self.tick(600)
        self.tick(840)
        self.api.scale.assert_not_called()

    def test_external_replica_change_starts_fresh_window(self):
        self.tick(0)
        self.tick(240, replicas=3)
        self.assertEqual(self.controller.scale_up_candidate["previous"], 3)
        self.assertEqual(self.controller.scale_up_candidate["started_at"], 240)
        self.tick(600, replicas=3)
        self.api.scale.assert_not_called()

    def test_final_deployment_recheck_failure_cancels_confirmation(self):
        self.tick(0)
        self.api.deployment.return_value = {"Replicas": 3}
        with self.assertRaisesRegex(ValueError, "changed before scaling"):
            self.tick(600)
        self.api.scale.assert_not_called()
        self.assertIsNone(self.controller.scale_up_candidate)

    def test_submission_error_preserves_pending_and_does_not_retry(self):
        self.tick(0)
        self.api.scale.side_effect = TimeoutError("uncertain response")
        with self.assertRaises(TimeoutError):
            self.tick(600)
        self.assertEqual(json.loads(self.path.read_text())["pending"]["target"], 4)
        self.tick(660)
        self.api.scale.assert_called_once()

    def test_restart_requires_new_confirmation_window(self):
        self.tick(0)
        self.controller = pool.Controller(self.args, self.api, self.path)
        self.tick(600)
        self.api.scale.assert_not_called()
        self.tick(1200)
        self.api.scale.assert_called_once_with(self.args, 4)

    def test_reset_pending_also_resets_confirmation(self):
        self.tick(0)
        self.controller.reset_pending()
        self.assertIsNone(self.controller.scale_up_candidate)
        self.tick(600)
        self.api.scale.assert_not_called()
        self.tick(1200)
        self.api.scale.assert_called_once_with(self.args, 4)

    def test_cooldown_precedes_confirmation(self):
        self.controller.state = {"cooldown_until": 1120}
        self.tick(0)
        self.assertIsNone(self.controller.scale_up_candidate)
        self.tick(120)
        self.tick(600)
        self.api.scale.assert_not_called()
        self.tick(720)
        self.api.scale.assert_called_once_with(self.args, 4)

    def test_dry_run_waits_without_writing_or_scaling(self):
        self.args.apply = False
        self.tick(0)
        self.tick(600)
        self.api.scale.assert_not_called()
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
