from __future__ import annotations

from datetime import datetime

from config.settings import CONFIG
from utils.cancellation import CancelledException
from utils.logger import get_logger
from utils.recoverable import JSON_RECOVERABLE_EXCEPTIONS

log = get_logger(__name__)
_AUTO_LEARNER_RECOVERABLE_EXCEPTIONS = JSON_RECOVERABLE_EXCEPTIONS

def _main_loop(
    self, mode, max_stocks, epochs, min_market_cap, include_all,
    continuous, interval, horizon, lookback, cycle_seconds, incremental,
    priority_stock_codes,
) -> None:
    cycle = 0
    current_epochs = epochs
    base_incremental = bool(incremental)
    current_incremental = bool(incremental)
    forced_full_retrain_cycles = 0

    try:
        while not self._should_stop():
            cycle += 1
            self._update(
                stage="cycle_start",
                message=(
                    f"=== Cycle {cycle} | Learned: {len(self._replay)} | "
                    f"Best: {self.progress.best_accuracy_ever:.1%} | "
                    f"Trend: {self._metrics.trend} ==="
                ),
                progress=0.0,
                stocks_processed=0,
                training_epoch=0,
                training_total_epochs=max(1, int(current_epochs)),
                validation_accuracy=0.0,
            )

            if self._wait_if_paused():
                break

            plateau = self._metrics.get_plateau_response()
            if plateau['action'] != 'none':
                current_epochs, current_incremental = self._handle_plateau(
                    plateau, current_epochs, current_incremental
                )

            cycle_incremental = bool(current_incremental)
            force_reason = ""
            if forced_full_retrain_cycles > 0:
                cycle_incremental = False
                forced_full_retrain_cycles = max(
                    0, int(forced_full_retrain_cycles) - 1
                )
                force_reason = "safety_forced_full_retrain"
            elif (
                base_incremental
                and int(self._FULL_RETRAIN_EVERY_CYCLES) > 0
                and cycle > 1
                and (cycle % int(self._FULL_RETRAIN_EVERY_CYCLES) == 0)
            ):
                cycle_incremental = False
                force_reason = "periodic_full_retrain"

            self.progress.full_retrain_cycles_remaining = int(
                max(0, forced_full_retrain_cycles)
            )
            if force_reason:
                self._update(
                    message=(
                        f"Cycle {cycle}: using full retrain path "
                        f"({force_reason})"
                    ),
                )

            success = self._run_cycle(
                max_stocks=max_stocks, epochs=current_epochs,
                min_market_cap=min_market_cap, interval=interval,
                horizon=horizon, lookback=lookback,
                incremental=cycle_incremental, cycle_number=cycle,
                priority_stock_codes=priority_stock_codes,
            )

            if success:
                self.progress.total_training_sessions += 1
                self.progress.consecutive_rejections = 0
                current_incremental = bool(base_incremental)
            else:
                streak = int(
                    max(0, self.progress.consecutive_rejections)
                )
                if (
                    streak
                    >= int(self._FORCE_FULL_RETRAIN_AFTER_REJECTIONS)
                ):
                    forced_full_retrain_cycles = max(
                        int(forced_full_retrain_cycles),
                        int(self._FORCE_FULL_RETRAIN_CYCLES),
                    )
                    self.progress.full_retrain_cycles_remaining = int(
                        forced_full_retrain_cycles
                    )
                    current_incremental = False
                    current_epochs = min(
                        200,
                        max(int(epochs), int(current_epochs)) + 2,
                    )
                    warn_msg = (
                        "Model safety guard: repeated rejections "
                        f"(streak={streak}) forcing "
                        f"{forced_full_retrain_cycles} full retrain cycle(s)"
                    )
                    self.progress.add_warning(warn_msg)
                    self._update(
                        stage="risk_guard",
                        message=warn_msg,
                        progress=100.0,
                    )
                if (
                    continuous
                    and streak >= int(self._REJECTION_COOLDOWN_AFTER_STREAK)
                ):
                    cooldown_seconds = int(
                        min(
                            int(self._MAX_REJECTION_COOLDOWN_SECONDS),
                            max(
                                int(self._MIN_REJECTION_COOLDOWN_SECONDS),
                                int(cycle_seconds) * 2,
                            ),
                        )
                    )
                    cooldown_msg = (
                        "Model safety cooldown after repeated rejections "
                        f"(streak={streak}) for {cooldown_seconds}s"
                    )
                    self.progress.add_warning(cooldown_msg)
                    self._update(
                        stage="cooldown",
                        message=cooldown_msg,
                        progress=100.0,
                    )
                    self._interruptible_sleep(cooldown_seconds)
            self._save_state()

            if not continuous:
                break

            if cycle % 5 == 0:
                self._rotator.clear_old_failures()
                self._replay.cleanup_stale_cache()

            self._update(
                stage="waiting",
                message=f"Cycle {cycle} done. Next in {cycle_seconds}s...",
                progress=100.0,
            )
            self._interruptible_sleep(cycle_seconds)

    except CancelledException:
        log.info("Learning cancelled")
    except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
        log.error(f"Learning error: {e}")
        import traceback
        traceback.print_exc()
        self.progress.add_error(str(e))
    finally:
        self.progress.is_running = False
        self._save_state()
        self._notify()

# =========================================================================
# MAIN LOOP - TARGETED MODE
# =========================================================================


def _run_cycle(
    self, max_stocks, epochs, min_market_cap, interval,
    horizon, lookback, incremental, cycle_number,
    priority_stock_codes: list[str] | None = None,
) -> bool:
    start_time = datetime.now()

    try:
        # === 1. Resolve interval ===
        eff_interval, eff_horizon, eff_lookback, min_bars = (
            self._resolve_interval(interval, horizon, lookback)
        )

        # === 2. Setup holdout ===
        self._ensure_holdout(eff_interval, eff_lookback, min_bars, cycle_number)
        if self._should_stop():
            raise CancelledException()

        # === 3. Discover new stocks ===
        self._update(stage="discovering", progress=2.0, message="Discovering stocks...")
        new_codes = self._rotator.discover_new(
            max_stocks=max_stocks, min_market_cap=min_market_cap,
            stop_check=self._should_stop,
            progress_cb=lambda msg, cnt: self._update(message=msg, stocks_found=cnt),
        )
        if self._should_stop():
            raise CancelledException()

        holdout_set = self._get_holdout_set()
        new_codes = [c for c in new_codes if c not in holdout_set]

        if priority_stock_codes:
            usable_priority = self._filter_priority_session_codes(
                list(priority_stock_codes),
                eff_interval,
                min_seconds=3600.0,
            )
            prioritized = []
            seen = set(new_codes)
            for code in usable_priority:
                c = str(code).strip()
                if not c or c in holdout_set or c in seen:
                    continue
                prioritized.append(c)
                seen.add(c)
            if prioritized:
                self._update(
                    message=f"Injecting {len(prioritized)} priority session stocks",
                    progress=4.0,
                )
                new_codes = prioritized + new_codes

        if new_codes:
            try:
                new_codes = self._prioritize_codes_by_news(
                    new_codes,
                    eff_interval,
                    max_probe=min(16, int(max_stocks)),
                )
            except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
                log.debug("News prioritization skipped: %s", e)

        # Recovery: if holdout filters everything and replay is empty,
        # reset rotation/holdout once and re-discover.
        if not new_codes and len(self._replay) == 0:
            self._update(
                message="No candidates after holdout; resetting rotation/holdout",
                progress=3.0,
            )
            self._rotator.reset_processed()
            self._rotator.reset_discovery()
            self._set_holdout_codes([])
            new_codes = self._rotator.discover_new(
                max_stocks=max_stocks, min_market_cap=min_market_cap,
                stop_check=self._should_stop,
                progress_cb=lambda msg, cnt: self._update(
                    message=msg, stocks_found=cnt,
                ),
            )
            if self._should_stop():
                raise CancelledException()
            holdout_set = self._get_holdout_set()
            new_codes = [c for c in new_codes if c not in holdout_set]

        # === 4. Mix with replay ===
        total_learned = len(self._replay)
        if total_learned < 20:
            new_ratio = 0.9
        elif total_learned < 100:
            new_ratio = 0.7
        elif total_learned < 500:
            new_ratio = 0.5
        else:
            new_ratio = 0.3

        num_new = max(3, int(max_stocks * new_ratio))
        num_replay = max_stocks - num_new

        new_batch = new_codes[:num_new]
        replay_batch = self._replay.sample(num_replay)
        replay_batch = [
            c for c in replay_batch
            if c not in new_batch and c not in holdout_set
        ]
        codes = new_batch + replay_batch

        # In VPN mode, large batches can overwhelm upstream providers.
        try:
            from core.network import get_network_env
            env = get_network_env()
            if env.is_vpn_active and len(codes) > 30:
                codes = codes[:30]
                self.progress.add_warning(
                    "VPN mode: batch capped to 30 stocks for fetch stability"
                )
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("VPN batch-cap check skipped: %s", e)

        if not codes:
            self._update(stage="error", message="No stocks available")
            return False

        self.progress.stocks_found = len(codes)
        self.progress.stocks_total = len(codes)
        self.progress.processed_count = self._rotator.processed_count
        self.progress.pool_size = self._rotator.pool_size

        self._update(
            message=f"Batch: {len(new_batch)} new + {len(replay_batch)} replay",
            progress=5.0,
        )

        # === 5. Fetch data ===
        self._update(
            stage="downloading", progress=10.0,
            message=f"Fetching {eff_interval} data...",
            stocks_processed=0,
        )

        ok_codes: list[str] = []
        failed_codes: list[str] = []
        total_codes = max(1, int(len(codes)))
        fetched_so_far = 0
        fetch_groups: list[tuple[str, list[str], bool, bool]] = []
        if new_batch:
            # New stocks: online + DB update (latest 1m window).
            fetch_groups.append(("new", list(new_batch), True, True))
        if replay_batch:
            fetch_groups.append(("replay", list(replay_batch), True, True))
        if not fetch_groups:
            fetch_groups.append(("batch", list(codes), True, True))

        for label, group_codes, group_online, group_update in fetch_groups:
            if not group_codes:
                continue
            base_processed = int(fetched_so_far)
            self._update(
                message=f"Fetching {label} stocks ({len(group_codes)})...",
                stocks_processed=base_processed,
                progress=10.0 + 30.0 * (base_processed / float(total_codes)),
            )
            try:
                group_ok, group_failed = self._fetcher.fetch_batch(
                    group_codes, eff_interval, eff_lookback, min_bars,
                    stop_check=self._should_stop,
                    progress_cb=lambda msg, cnt, b=base_processed, tag=label: self._update(
                        message=f"{tag}: {msg}",
                        stocks_processed=min(len(codes), b + int(cnt)),
                        progress=10.0 + 30.0 * (
                            (b + int(cnt)) / float(total_codes)
                        ),
                    ),
                    allow_online=group_online,
                    update_db=group_update,
                )
            except TypeError:
                group_ok, group_failed = self._fetcher.fetch_batch(
                    group_codes, eff_interval, eff_lookback, min_bars,
                    stop_check=self._should_stop,
                    progress_cb=lambda msg, cnt, b=base_processed, tag=label: self._update(
                        message=f"{tag}: {msg}",
                        stocks_processed=min(len(codes), b + int(cnt)),
                        progress=10.0 + 30.0 * (
                            (b + int(cnt)) / float(total_codes)
                        ),
                    ),
                )
            ok_codes.extend(group_ok)
            failed_codes.extend(group_failed)
            fetched_so_far += len(group_codes)
            if self._should_stop():
                raise CancelledException()

        ok_codes = list(dict.fromkeys(ok_codes))
        ok_set = set(ok_codes)
        failed_codes = [
            c for c in dict.fromkeys(failed_codes)
            if c not in ok_set
        ]

        batch_size = max(1, int(len(codes)))
        min_ok = min(batch_size, max(3, int(batch_size * 0.05)))
        try:
            from core.network import get_network_env
            if get_network_env().is_vpn_active:
                min_ok = min(batch_size, max(2, int(batch_size * 0.03)))
        except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
            log.debug("VPN min-ok adjustment unavailable: %s", e)
        if len(ok_codes) < min_ok and failed_codes and not self._should_stop():
            # FIX 1M: More aggressive retry for intraday data with limited availability
            if eff_interval in ("1m", "2m", "5m"):
                # For 1m data, use much more relaxed thresholds due to limited free data
                relaxed_min_bars = max(CONFIG.SEQUENCE_LENGTH // 3, 15)  # Very relaxed for 1m
            else:
                relaxed_min_bars = max(CONFIG.SEQUENCE_LENGTH + 20, int(min_bars * 0.7))
            retry_base_processed = int(max(0, len(ok_codes)))
            retry_groups: list[tuple[str, list[str], bool, bool]] = []
            retry_cap = min(len(failed_codes), max(8, min_ok * 2))
            new_batch_set = set(new_batch)
            replay_batch_set = set(replay_batch)
            new_failed = [c for c in failed_codes if c in new_batch_set]
            replay_failed = [c for c in failed_codes if c in replay_batch_set]
            if new_failed:
                retry_groups.append((
                    "new",
                    new_failed[: min(len(new_failed), retry_cap)],
                    True,
                    True,
                ))
            used_retry = sum(len(g[1]) for g in retry_groups)
            replay_left = max(0, retry_cap - used_retry)
            if replay_failed and replay_left > 0:
                retry_groups.append((
                    "replay",
                    replay_failed[: min(len(replay_failed), replay_left)],
                    True,
                    True,
                ))
            if not retry_groups:
                retry_groups.append((
                    "batch",
                    failed_codes[:retry_cap],
                    True,
                    True,
                ))
            total_retry = max(1, sum(len(g[1]) for g in retry_groups))
            self._update(
                message=(
                    f"Retrying {sum(len(g[1]) for g in retry_groups)} failed stocks "
                    f"(relaxed min bars {relaxed_min_bars})"
                ),
                progress=36.0,
            )
            retry_ok_all: list[str] = []
            retry_failed_all: list[str] = []
            retry_done = 0
            retried_set: set[str] = set()
            for label, retry_codes, retry_online, retry_update in retry_groups:
                if not retry_codes:
                    continue
                base_retry = int(retry_done)
                retried_set.update(retry_codes)
                try:
                    retry_ok, retry_failed = self._fetcher.fetch_batch(
                        retry_codes, eff_interval, eff_lookback, relaxed_min_bars,
                        stop_check=self._should_stop,
                        progress_cb=lambda msg, cnt, b=base_retry, tag=label: self._update(
                            message=f"Retry {tag}: {msg}",
                            stocks_processed=min(
                                len(codes),
                                retry_base_processed + b + int(cnt),
                            ),
                            progress=36.0 + 4.0 * (
                                (b + int(cnt)) / float(total_retry)
                            ),
                        ),
                        allow_online=retry_online,
                        update_db=retry_update,
                    )
                except TypeError:
                    retry_ok, retry_failed = self._fetcher.fetch_batch(
                        retry_codes, eff_interval, eff_lookback, relaxed_min_bars,
                        stop_check=self._should_stop,
                        progress_cb=lambda msg, cnt, b=base_retry, tag=label: self._update(
                            message=f"Retry {tag}: {msg}",
                            stocks_processed=min(
                                len(codes),
                                retry_base_processed + b + int(cnt),
                            ),
                            progress=36.0 + 4.0 * (
                                (b + int(cnt)) / float(total_retry)
                            ),
                        ),
                    )
                retry_ok_all.extend(retry_ok)
                retry_failed_all.extend(retry_failed)
                retry_done += len(retry_codes)
                if self._should_stop():
                    raise CancelledException()

            ok_set = set(ok_codes)
            for code in retry_ok_all:
                if code not in ok_set:
                    ok_codes.append(code)
                    ok_set.add(code)
            retry_failed_set = set(retry_failed_all)
            failed_codes = [
                c for c in failed_codes
                if c not in ok_set
                and (c not in retried_set or c in retry_failed_set)
            ]

        # FIX 1M: Second-tier ultra-relaxed retry for intraday data
        if len(ok_codes) < min_ok and failed_codes and eff_interval in ("1m", "2m", "5m"):
            ultra_relaxed_min_bars = max(CONFIG.SEQUENCE_LENGTH // 4, 10)  # Ultra relaxed
            ultra_retry_cap = min(len(failed_codes), max(12, min_ok * 3))
            ultra_retry_codes = failed_codes[:ultra_retry_cap]
            if ultra_retry_codes:
                self._update(
                    message=(
                        f"Ultra-relaxed retry for {len(ultra_retry_codes)} stocks "
                        f"(min bars {ultra_relaxed_min_bars})"
                    ),
                    progress=40.0,
                )
                try:
                    ultra_ok, ultra_failed = self._fetcher.fetch_batch(
                        ultra_retry_codes, eff_interval, eff_lookback, ultra_relaxed_min_bars,
                        stop_check=self._should_stop,
                        progress_cb=lambda msg, cnt: self._update(
                            message=f"Ultra retry: {msg}",
                            stocks_processed=min(len(codes), len(ok_codes) + int(cnt)),
                            progress=40.0 + 4.0 * (int(cnt) / float(len(ultra_retry_codes))),
                        ),
                        allow_online=True,
                        update_db=True,
                    )
                except TypeError:
                    ultra_ok, ultra_failed = self._fetcher.fetch_batch(
                        ultra_retry_codes, eff_interval, eff_lookback, ultra_relaxed_min_bars,
                        stop_check=self._should_stop,
                        progress_cb=lambda msg, cnt: self._update(
                            message=f"Ultra retry: {msg}",
                            stocks_processed=min(len(codes), len(ok_codes) + int(cnt)),
                            progress=40.0 + 4.0 * (int(cnt) / float(len(ultra_retry_codes))),
                        ),
                        allow_online=True,
                        update_db=True,
                    )
                ok_set = set(ok_codes)
                for code in ultra_ok:
                    if code not in ok_set:
                        ok_codes.append(code)
                        ok_set.add(code)
                failed_codes = [c for c in failed_codes if c not in ok_set]

        for code in failed_codes:
            self._rotator.mark_failed(code)

        if len(ok_codes) < min_ok:
            for code in new_batch:
                self._rotator.mark_processed([code])

            # FIX HELP: Add helpful message about 1m data limitations
            interval_name = str(eff_interval).lower()
            if interval_name in ("1m", "2m", "5m"):
                help_msg = (
                    f"Too few stocks: {len(ok_codes)}/{len(codes)}. "
                    f"Note: {interval_name} historical data is limited from free sources "
                    f"(typically 1-2 days). Tips: 1) Run during market hours (9:30-15:00 CST) "
                    f"for live session data, 2) Use 1d interval for more data availability, "
                    f"3) Retry later for fresh online data."
                )
                log.warning(help_msg)
                self._update(
                    stage="error",
                    message=help_msg,
                )
            else:
                self._update(
                    stage="error",
                    message=f"Too few stocks: {len(ok_codes)}/{len(codes)}",
                )
            return False

        # === 6. Verify fetched history before training ===
        self._update(
            stage="validating_data",
            progress=41.0,
            message=f"Using latest online {eff_interval} history...",
        )
        self._update(
            message=f"Prepared {len(ok_codes)} stocks for training",
            progress=41.8,
        )

        # === 6. Backup model ===
        self._update(stage="backup", progress=42.0, message="Backing up current model...")
        self._guardian.backup_current(eff_interval, eff_horizon)
        if self._should_stop():
            raise CancelledException()

        # === 7. Pre-training validation ===
        pre_val = None
        holdout_snapshot = list(self._get_holdout_set())
        if holdout_snapshot and len(self._replay) > 10:
            self._update(message="Pre-training validation...", progress=45.0)
            pre_val = self._guardian.validate_model(
                eff_interval, eff_horizon, holdout_snapshot, eff_lookback,
            )
            log.info(f"Pre-validation: {pre_val}")
            if self._should_stop():
                raise CancelledException()

        # === 8. Train ===
        lr = self._lr_scheduler.get_lr(cycle_number, incremental)
        self._update(
            stage="training", progress=50.0,
            message=f"Training {len(ok_codes)} stocks (lr={lr:.6f}, e={epochs})...",
            training_total_epochs=epochs,
        )

        result = self._train(
            ok_codes, epochs, eff_interval, eff_horizon,
            eff_lookback, incremental, lr,
        )

        if result.get("status") == "cancelled":
            raise CancelledException()
        self._emit_model_drift_alarm_if_needed(
            result,
            context=f"auto_cycle_{cycle_number}",
        )

        acc = float(result.get("best_accuracy", 0.0))
        self.progress.training_accuracy = acc
        self._metrics.record(acc)
        self.progress.accuracy_trend = self._metrics.trend

        deployable, deploy_reason = self._trainer_result_is_deployable(result)
        if not deployable:
            log.warning("REJECTED before holdout: %s", deploy_reason)
            self.progress.add_warning(f"Rejected: {deploy_reason}")
            self.progress.model_was_rejected = True
            self._guardian.restore_backup(eff_interval, eff_horizon)
            self._finalize_cycle(
                False, ok_codes, new_batch, replay_batch,
                eff_interval, eff_horizon, eff_lookback,
                acc, cycle_number, start_time,
            )
            return False

        # === 9. Post-training validation ===
        self._update(
            stage="validating", progress=90.0,
            message="Validating on holdout stocks...",
        )

        accepted = self._validate_and_decide(
            eff_interval, eff_horizon, eff_lookback, pre_val, acc,
        )

        # === 10. Update state ===
        self._finalize_cycle(
            accepted, ok_codes, new_batch, replay_batch,
            eff_interval, eff_horizon, eff_lookback,
            acc, cycle_number, start_time,
        )

        return accepted

    except CancelledException:
        raise
    except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
        log.error(f"Cycle error: {e}")
        import traceback
        traceback.print_exc()
        self._update(stage="error", message=str(e))
        self.progress.add_error(str(e))
        return False

# =========================================================================
# SINGLE CYCLE - TARGETED MODE
# =========================================================================


def _run_targeted_cycle(
    self,
    stock_codes: list[str],
    epochs: int,
    interval: str,
    horizon: int,
    lookback: int,
    incremental: bool,
    cycle_number: int,
) -> bool:
    """Single training cycle on user-specified stocks."""
    start_time = datetime.now()

    try:
        # === 1. Resolve interval ===
        eff_interval, eff_horizon, eff_lookback, min_bars = (
            self._resolve_interval(interval, horizon, lookback)
        )

        # === 2. Setup holdout ===
        self._ensure_holdout(eff_interval, eff_lookback, min_bars, cycle_number)
        if self._should_stop():
            raise CancelledException()

        holdout_set = self._get_holdout_set()
        train_codes = [c for c in stock_codes if c not in holdout_set]

        if not train_codes:
            self.progress.add_warning(
                "All selected stocks overlap with holdout set - "
                "training on them anyway"
            )
            train_codes = list(stock_codes)

        self.progress.stocks_found = len(train_codes)
        self.progress.stocks_total = len(train_codes)
        known_codes = set(self._replay.get_all())
        targeted_new = [c for c in train_codes if c not in known_codes]
        targeted_replay = [c for c in train_codes if c in known_codes]

        self._update(
            stage="targeted_training",
            message=(
                f"Targeted batch: {len(targeted_new)} new + "
                f"{len(targeted_replay)} replay"
            ),
            progress=5.0,
        )

        # === 3. Fetch data ===
        self._update(
            stage="downloading",
            progress=10.0,
            message=f"Fetching {eff_interval} data for {len(train_codes)} stocks...",
            stocks_processed=0,
        )

        ok_codes: list[str] = []
        failed_codes: list[str] = []
        total_codes = max(1, int(len(train_codes)))
        fetched_so_far = 0
        fetch_groups: list[tuple[str, list[str], bool, bool]] = []
        if targeted_new:
            fetch_groups.append(("new", list(targeted_new), True, True))
        if targeted_replay:
            fetch_groups.append(("replay", list(targeted_replay), True, True))
        if not fetch_groups:
            fetch_groups.append(("batch", list(train_codes), True, True))

        for label, group_codes, group_online, group_update in fetch_groups:
            if not group_codes:
                continue
            base_processed = int(fetched_so_far)
            self._update(
                message=f"Fetching targeted {label} stocks ({len(group_codes)})...",
                stocks_processed=base_processed,
                progress=10.0 + 30.0 * (base_processed / float(total_codes)),
            )
            try:
                group_ok, group_failed = self._fetcher.fetch_batch(
                    group_codes, eff_interval, eff_lookback, min_bars,
                    stop_check=self._should_stop,
                    progress_cb=lambda msg, cnt, b=base_processed, tag=label: self._update(
                        message=f"{tag}: {msg}",
                        stocks_processed=min(len(train_codes), b + int(cnt)),
                        progress=10.0 + 30.0 * (
                            (b + int(cnt)) / float(total_codes)
                        ),
                    ),
                    allow_online=group_online,
                    update_db=group_update,
                )
            except TypeError:
                group_ok, group_failed = self._fetcher.fetch_batch(
                    group_codes, eff_interval, eff_lookback, min_bars,
                    stop_check=self._should_stop,
                    progress_cb=lambda msg, cnt, b=base_processed, tag=label: self._update(
                        message=f"{tag}: {msg}",
                        stocks_processed=min(len(train_codes), b + int(cnt)),
                        progress=10.0 + 30.0 * (
                            (b + int(cnt)) / float(total_codes)
                        ),
                    ),
                )
            ok_codes.extend(group_ok)
            failed_codes.extend(group_failed)
            fetched_so_far += len(group_codes)
            if self._should_stop():
                raise CancelledException()

        ok_codes = list(dict.fromkeys(ok_codes))
        ok_set = set(ok_codes)
        failed_codes = [
            c for c in dict.fromkeys(failed_codes)
            if c not in ok_set
        ]

        if not ok_codes and failed_codes and not self._should_stop():
            # FIX 1M: More aggressive retry for intraday data with limited availability
            if eff_interval in ("1m", "2m", "5m"):
                relaxed_min_bars = max(CONFIG.SEQUENCE_LENGTH // 3, 15)  # Very relaxed for 1m
            else:
                relaxed_min_bars = max(CONFIG.SEQUENCE_LENGTH + 20, int(min_bars * 0.7))
            retry_base_processed = int(max(0, len(ok_codes)))
            retry_groups: list[tuple[str, list[str], bool, bool]] = []
            retry_cap = min(len(failed_codes), 12)
            targeted_new_set = set(targeted_new)
            targeted_replay_set = set(targeted_replay)
            new_failed = [c for c in failed_codes if c in targeted_new_set]
            replay_failed = [c for c in failed_codes if c in targeted_replay_set]
            if new_failed:
                retry_groups.append((
                    "new",
                    new_failed[: min(len(new_failed), retry_cap)],
                    True,
                    True,
                ))
            used_retry = sum(len(g[1]) for g in retry_groups)
            replay_left = max(0, retry_cap - used_retry)
            if replay_failed and replay_left > 0:
                retry_groups.append((
                    "replay",
                    replay_failed[: min(len(replay_failed), replay_left)],
                    True,
                    True,
                ))
            if not retry_groups:
                retry_groups.append((
                    "batch",
                    failed_codes[:retry_cap],
                    True,
                    True,
                ))
            total_retry = max(1, sum(len(g[1]) for g in retry_groups))
            self._update(
                message=(
                    f"Retrying targeted fetch for "
                    f"{sum(len(g[1]) for g in retry_groups)} stocks "
                    f"(min bars {relaxed_min_bars})"
                ),
                progress=36.0,
            )
            retry_ok_all: list[str] = []
            retry_failed_all: list[str] = []
            retry_done = 0
            retried_set: set[str] = set()
            for label, retry_codes, retry_online, retry_update in retry_groups:
                if not retry_codes:
                    continue
                base_retry = int(retry_done)
                retried_set.update(retry_codes)
                try:
                    retry_ok, retry_failed = self._fetcher.fetch_batch(
                        retry_codes, eff_interval, eff_lookback, relaxed_min_bars,
                        stop_check=self._should_stop,
                        progress_cb=lambda msg, cnt, b=base_retry, tag=label: self._update(
                            message=f"Retry {tag}: {msg}",
                            stocks_processed=min(
                                len(train_codes),
                                retry_base_processed + b + int(cnt),
                            ),
                            progress=36.0 + 4.0 * (
                                (b + int(cnt)) / float(total_retry)
                            ),
                        ),
                        allow_online=retry_online,
                        update_db=retry_update,
                    )
                except TypeError:
                    retry_ok, retry_failed = self._fetcher.fetch_batch(
                        retry_codes, eff_interval, eff_lookback, relaxed_min_bars,
                        stop_check=self._should_stop,
                        progress_cb=lambda msg, cnt, b=base_retry, tag=label: self._update(
                            message=f"Retry {tag}: {msg}",
                            stocks_processed=min(
                                len(train_codes),
                                retry_base_processed + b + int(cnt),
                            ),
                            progress=36.0 + 4.0 * (
                                (b + int(cnt)) / float(total_retry)
                            ),
                        ),
                    )
                retry_ok_all.extend(retry_ok)
                retry_failed_all.extend(retry_failed)
                retry_done += len(retry_codes)
                if self._should_stop():
                    raise CancelledException()

            ok_set = set(ok_codes)
            for code in retry_ok_all:
                if code not in ok_set:
                    ok_codes.append(code)
                    ok_set.add(code)
            retry_failed_set = set(retry_failed_all)
            failed_codes = [
                c for c in failed_codes
                if c not in ok_set
                and (c not in retried_set or c in retry_failed_set)
            ]

        if failed_codes:
            failed_display = ', '.join(failed_codes[:10])
            extra = f" (+{len(failed_codes) - 10} more)" if len(failed_codes) > 10 else ""
            self.progress.add_warning(
                f"Failed to fetch: {failed_display}{extra}"
            )

        if not ok_codes:
            self._update(
                stage="error",
                message=(
                    f"No valid data for any of the {len(train_codes)} stocks. "
                    f"Check codes and network connection."
                ),
            )
            self.progress.add_error(
                f"All {len(train_codes)} stocks failed data fetch"
            )
            return False

        # === 4. Verify fetched history before training ===
        self._update(
            stage="validating_data",
            progress=41.0,
            message=f"Using latest online {eff_interval} history...",
        )
        self._update(
            message=f"Prepared {len(ok_codes)} stocks for training",
            progress=41.8,
        )

        # === 4. Backup model ===
        self._update(
            stage="backup", progress=42.0,
            message="Backing up current model...",
        )
        self._guardian.backup_current(eff_interval, eff_horizon)
        if self._should_stop():
            raise CancelledException()

        # === 5. Pre-training validation ===
        pre_val = None
        holdout_snapshot = list(self._get_holdout_set())
        if holdout_snapshot and len(self._replay) > 10:
            self._update(message="Pre-training validation...", progress=45.0)
            pre_val = self._guardian.validate_model(
                eff_interval, eff_horizon, holdout_snapshot, eff_lookback,
            )
            log.info(f"Pre-validation: {pre_val}")
            if self._should_stop():
                raise CancelledException()

        # === 6. Train ===
        lr = self._lr_scheduler.get_lr(cycle_number, incremental)
        self._update(
            stage="training",
            progress=50.0,
            message=(
                f"Training on {len(ok_codes)} targeted stocks "
                f"(lr={lr:.6f}, epochs={epochs})..."
            ),
            training_total_epochs=epochs,
        )

        result = self._train(
            ok_codes, epochs, eff_interval, eff_horizon,
            eff_lookback, incremental, lr,
        )

        if result.get("status") == "cancelled":
            raise CancelledException()
        self._emit_model_drift_alarm_if_needed(
            result,
            context=f"targeted_cycle_{cycle_number}",
        )

        acc = float(result.get("best_accuracy", 0.0))
        self.progress.training_accuracy = acc
        self._metrics.record(acc)
        self.progress.accuracy_trend = self._metrics.trend

        deployable, deploy_reason = self._trainer_result_is_deployable(result)
        if not deployable:
            log.warning("REJECTED before holdout: %s", deploy_reason)
            self.progress.add_warning(f"Rejected: {deploy_reason}")
            self.progress.model_was_rejected = True
            self._guardian.restore_backup(eff_interval, eff_horizon)
            self._finalize_cycle(
                False, ok_codes, ok_codes, [],
                eff_interval, eff_horizon, eff_lookback,
                acc, cycle_number, start_time,
            )
            return False

        # === 7. Post-training validation ===
        self._update(
            stage="validating",
            progress=90.0,
            message="Validating on holdout stocks...",
        )

        accepted = self._validate_and_decide(
            eff_interval, eff_horizon, eff_lookback, pre_val, acc,
        )

        # === 8. Update state ===
        self._finalize_cycle(
            accepted, ok_codes, ok_codes, [],
            eff_interval, eff_horizon, eff_lookback,
            acc, cycle_number, start_time,
        )

        return accepted

    except CancelledException:
        raise
    except _AUTO_LEARNER_RECOVERABLE_EXCEPTIONS as e:
        log.error(f"Targeted cycle error: {e}")
        import traceback
        traceback.print_exc()
        self._update(stage="error", message=str(e))
        self.progress.add_error(str(e))
        return False

# =========================================================================
# SHARED CYCLE FINALIZATION (DRY)
# =========================================================================
