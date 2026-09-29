"""Single source of truth for feature names and window sizes.

Both the offline batch pipeline (offline.py, used to build training data)
and the online Redis-backed pipeline (store.py, used at serving time) import
these constants instead of hardcoding window sizes or feature names. That's
what "training and serving pull from the same feature logic" actually means
here: the two implementations differ (vectorized pandas replay vs
incremental Redis updates, because training needs to process 500K historical
rows fast and serving needs to process one transaction in milliseconds), but
they compute the same features, over the same windows, with the same
point-in-time semantics. test_feature_parity.py checks that this is actually
true rather than just asserted in a comment.

Point-in-time semantics: every feature for a transaction is computed from
that entity's state strictly BEFORE this transaction (closed='left' / a
get-before-update ordering). A transaction's own amount, address, or device
never leaks into its own features. This mirrors how a real serving system
works: when a transaction arrives, you look up existing entity state, score
it, and only then record the new transaction into that state.
"""

# Entity resolution (see entity.py). Vesta's card1-6/addr1/addr2 fields are
# coarse categorical bins, not real account numbers, so this is a proxy for
# "the same person/card", not a guaranteed one-to-one mapping. Rows missing
# any key field get a singleton ID instead of colliding into a shared
# "unknown" bucket (see entity.py docstring for why that matters).
ENTITY_KEY_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]

# Rolling window sizes, in seconds. TransactionDT in IEEE-CIS is a relative
# offset in seconds from an arbitrary reference point, not a wall-clock
# timestamp, but it's monotonic and correctly spaced, so second-denominated
# windows behave the same as they would on real timestamps.
WINDOW_SHORT_SECONDS = 3600  # 1 hour
WINDOW_LONG_SECONDS = 86400  # 24 hours

# Note: there is deliberately no "address changed" feature. addr1 is one of
# ENTITY_KEY_COLS, so an entity by definition never has a different addr1 from
# its previous transaction; the flag was 0.0 on every row (found by looking
# at the Phase 2 feature summary: std 0). It was removed rather than kept as
# a constant column.
#
# The engineered feature columns this pipeline produces, and what each one
# means. Both offline.py and store.py must produce exactly these columns
# with this meaning.
FEATURE_COLUMNS = [
    "entity_txn_seq_num",       # count of this entity's transactions strictly before this one (0 = first time we've seen this entity)
    "entity_time_since_last_sec",  # seconds since this entity's previous transaction; NaN if this is the first (or arrived out of order, online only)
    "entity_txn_count_1h",      # count of this entity's transactions in the trailing 1h, excluding this one
    "entity_txn_count_24h",     # count of this entity's transactions in the trailing 24h, excluding this one
    "entity_amt_sum_24h",       # sum of TransactionAmt for this entity in the trailing 24h, excluding this one
    "entity_amt_mean_24h",      # mean of TransactionAmt for this entity in the trailing 24h, excluding this one
    "entity_device_changed",    # 1 if DeviceInfo differs from this entity's last-seen DeviceInfo, 0 if same, NaN if first txn or DeviceInfo missing on either side
]
