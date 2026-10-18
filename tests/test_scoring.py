"""bidkv.scoring 单元测试。

ps aux | grep python | grep -v grep | grep -v conda | head -10
- ScoringStrategy Protocol 合规性
- PositionalScoring 策略
- generate_bids() 返回合法 CompressionBid
- build_bids() 统一生成器
"""

from __future__ import annotations

import math

import pytest

from bidkv.protocol.bid import CompressionBid
from bidkv.scoring.base import ScoringStrategy
from bidkv.scoring.positional import PositionalScoring

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_IDS = list(range(100))
DEFAULT_COMPRESSION_LEVELS = [0.2, 0.4, 0.6, 0.8]


def _validate_scores(scores: list[float], n: int) -> None:
    """验证 scores 列表的基本属性。"""
    assert len(scores) == n, f"Expected {n} scores, got {len(scores)}"
    for i, s in enumerate(scores):
        assert 0.0 <= s <= 1.0, f"Score at index {i} out of range: {s}"


def _validate_bids(bids: list[CompressionBid], request_id: str) -> None:
    """验证 bids 列表的合法性。"""
    for bid in bids:
        assert isinstance(bid, CompressionBid)
        assert bid.request_id == request_id
        assert bid.tokens_freed > 0
        assert 0.0 <= bid.quality_delta <= 1.0
        assert bid.compress_latency_ms >= 0.0
        assert 0.0 <= bid.confidence <= 1.0
        assert bid.bid_id
        assert bid.algorithm_id
        # 三层体系：metadata 应为 dict
        assert isinstance(bid.metadata, dict)


def _spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """计算 Spearman rank correlation coefficient。

    使用手动实现，不依赖 scipy。
    """
    n = len(x)
    assert n == len(y), "Lists must have the same length"
    if n < 2:
        return 1.0

    def _ranks(values: list[float]) -> list[float]:
        """计算排名（处理并列情况使用平均排名）。"""
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)

    # Pearson correlation on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


# ===========================================================================
# ScoringStrategy Protocol 合规性
# ===========================================================================


class TestScoringStrategyProtocol:
    """测试所有策略是否满足 ScoringStrategy Protocol。"""

    def test_positional_is_scoring_strategy(self) -> None:
        scorer = PositionalScoring()
        assert isinstance(scorer, ScoringStrategy)


# ===========================================================================
# PositionalScoring 测试
# ===========================================================================


class TestPositionalScoring:
    """Positional 评分策略测试。"""

    def test_init_defaults(self) -> None:
        scorer = PositionalScoring()
        assert scorer.heavy_ratio == 0.2
        assert scorer.recent_ratio == 0.2
        assert scorer.decode_steps == 0

    def test_init_custom_params(self) -> None:
        scorer = PositionalScoring(heavy_ratio=0.3, recent_ratio=0.1)
        assert scorer.heavy_ratio == 0.3
        assert scorer.recent_ratio == 0.1

    def test_invalid_heavy_ratio(self) -> None:
        with pytest.raises(ValueError, match="heavy_ratio"):
            PositionalScoring(heavy_ratio=1.5)

    def test_invalid_recent_ratio(self) -> None:
        with pytest.raises(ValueError, match="recent_ratio"):
            PositionalScoring(recent_ratio=-0.1)

    def test_heavy_plus_recent_exceeds_one(self) -> None:
        with pytest.raises(ValueError, match="heavy_ratio \\+ recent_ratio"):
            PositionalScoring(heavy_ratio=0.6, recent_ratio=0.5)

    def test_score_empty(self) -> None:
        scorer = PositionalScoring()
        assert scorer.score([]) == []

    def test_score_positional_heuristic(self) -> None:
        """无 decode 数据时使用位置启发式。"""
        scorer = PositionalScoring()
        scores = scorer.score(DEFAULT_TOKEN_IDS)
        _validate_scores(scores, len(DEFAULT_TOKEN_IDS))
        # 第一个 token（attention sink）应比中间 token 更重要
        assert scores[0] > scores[len(scores) // 2]

    def test_score_with_cumulative_attention(self) -> None:
        """有 decode 数据时使用累积注意力评分。"""
        scorer = PositionalScoring()
        n = 20
        token_ids = list(range(n))

        # 模拟 3 个 decode step：token 5 和 15 被频繁 attend to
        for _ in range(3):
            pattern = [0.01] * n
            pattern[5] = 0.5  # heavy hitter
            pattern[15] = 0.3  # heavy hitter
            pattern[-1] = 0.2  # recent
            scorer.update_from_decode_step(pattern)

        assert scorer.decode_steps == 3
        scores = scorer.score(token_ids)
        _validate_scores(scores, n)
        # Token 5 应该有高分（heavy hitter）
        assert scores[5] > 0.5

    def test_score_via_context(self) -> None:
        """通过 context 传递 attention_pattern。"""
        scorer = PositionalScoring()
        n = 10
        pattern = [0.1] * n
        pattern[3] = 0.9
        scores = scorer.score(list(range(n)), attention_pattern=pattern)
        _validate_scores(scores, n)
        assert scorer.decode_steps == 1

    def test_generate_bids(self) -> None:
        scorer = PositionalScoring()
        bids = scorer.generate_bids("req-1", DEFAULT_TOKEN_IDS, DEFAULT_COMPRESSION_LEVELS)
        assert len(bids) == len(DEFAULT_COMPRESSION_LEVELS)
        _validate_bids(bids, "req-1")
        # bid_id 格式检查
        for i, bid in enumerate(bids):
            assert bid.bid_id == f"req-1:bid:{i}"
            assert bid.algorithm_id == "positional"
            assert "scoring_method" in bid.metadata
            assert bid.metadata["scoring_method"] == "positional"

    def test_generate_bids_empty(self) -> None:
        scorer = PositionalScoring()
        bids = scorer.generate_bids("req-1", [], DEFAULT_COMPRESSION_LEVELS)
        assert bids == []

    def test_generate_bids_with_decode_data(self) -> None:
        """有 decode 数据时，confidence 应更高。"""
        scorer = PositionalScoring()
        n = 50
        token_ids = list(range(n))
        # 添加 decode 数据
        for _ in range(5):
            scorer.update_from_decode_step([0.1] * n)

        bids = scorer.generate_bids("req-1", token_ids, [0.3])
        assert len(bids) == 1
        assert bids[0].confidence > 0.3  # 有 decode 数据时 confidence 应 > 0.3

    def test_reset(self) -> None:
        scorer = PositionalScoring()
        scorer.update_from_decode_step([0.1, 0.2, 0.3])
        assert scorer.decode_steps == 1
        scorer.reset()
        assert scorer.decode_steps == 0

    def test_tokens_freed_monotonically_increases(self) -> None:
        """更高压缩级别应释放更多 token。"""
        scorer = PositionalScoring()
        bids = scorer.generate_bids("req-1", DEFAULT_TOKEN_IDS, DEFAULT_COMPRESSION_LEVELS)
        freed_values = [b.tokens_freed for b in bids]
        for i in range(len(freed_values) - 1):
            assert freed_values[i] <= freed_values[i + 1]


# ===========================================================================
# generate_bids 通用行为测试
# ===========================================================================


class TestGenerateBidsCommon:
    """所有策略的 generate_bids 通用行为测试。"""

    @pytest.mark.parametrize(
        "scorer_cls,kwargs",
        [
            (PositionalScoring, {}),
        ],
    )
    def test_bid_fields_complete(self, scorer_cls: type, kwargs: dict) -> None:
        """CompressionBid 包含三层体系所有必要字段。"""
        scorer = scorer_cls(**kwargs)
        bids = scorer.generate_bids("req-test", DEFAULT_TOKEN_IDS, DEFAULT_COMPRESSION_LEVELS)
        _validate_bids(bids, "req-test")
        for bid in bids:
            # Layer 1
            assert bid.tokens_freed > 0
            assert 0.0 <= bid.quality_delta <= 1.0
            # Layer 2
            assert bid.compress_latency_ms >= 0.0
            assert bid.request_id == "req-test"
            # Layer 3
            assert 0.0 <= bid.confidence <= 1.0
            assert isinstance(bid.metadata, dict)
            assert "compression_level" in bid.metadata

    @pytest.mark.parametrize(
        "scorer_cls,kwargs",
        [
            (PositionalScoring, {}),
        ],
    )
    def test_bid_id_format(self, scorer_cls: type, kwargs: dict) -> None:
        """bid_id 格式为 '{request_id}:bid:{level}'。"""
        scorer = scorer_cls(**kwargs)
        bids = scorer.generate_bids("req-format", DEFAULT_TOKEN_IDS, [0.3, 0.6])
        for i, bid in enumerate(bids):
            assert bid.bid_id == f"req-format:bid:{i}"

    @pytest.mark.parametrize(
        "scorer_cls,kwargs",
        [
            (PositionalScoring, {}),
        ],
    )
    def test_empty_token_ids_returns_empty(self, scorer_cls: type, kwargs: dict) -> None:
        scorer = scorer_cls(**kwargs)
        bids = scorer.generate_bids("req-empty", [], DEFAULT_COMPRESSION_LEVELS)
        assert bids == []

    @pytest.mark.parametrize(
        "scorer_cls,kwargs",
        [
            (PositionalScoring, {}),
        ],
    )
    def test_single_token(self, scorer_cls: type, kwargs: dict) -> None:
        """单个 token 不应生成 bid（至少保留 1 个）。"""
        scorer = scorer_cls(**kwargs)
        bids = scorer.generate_bids("req-single", [42], [0.5])
        # tokens_freed 不能 >= total tokens（至少保留1）
        # 对于 1 个 token，tokens_freed = min(int(1*0.5)=0... 实际 max(1, 0) = 1，
        # 但 min(1, 1-1) = 0，所以 skip
        assert bids == []


# ===========================================================================
# build_bids 统一生成器测试
# ===========================================================================


class TestBuildBids:
    """测试 build_bids() 统一 bids 生成逻辑。"""

    def test_basic(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        bids = build_bids(
            request_id="req-bb",
            token_ids=[10, 20, 30, 40, 50],
            scores=scores,
            compression_levels=[0.4],
            algorithm_id="test",
        )
        assert len(bids) == 1
        bid = bids[0]
        assert bid.request_id == "req-bb"
        assert bid.algorithm_id == "test"
        assert bid.tokens_freed == 2  # int(5 * 0.4) = 2
        assert bid.bid_id == "req-bb:bid:0"
        # 被移除的应是分数最低的 2 个：0.1, 0.3 → δ = 0.2
        assert abs(bid.quality_delta - 0.2) < 1e-9

    def test_empty_tokens(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=[],
            scores=[],
            compression_levels=[0.5],
            algorithm_id="test",
        )
        assert bids == []

    def test_scores_length_mismatch(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        with pytest.raises(ValueError, match="scores length"):
            build_bids(
                request_id="req-bb",
                token_ids=[1, 2, 3],
                scores=[0.5, 0.5],
                compression_levels=[0.3],
                algorithm_id="test",
            )

    def test_multiple_levels(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=list(range(10)),
            scores=[0.5] * 10,
            compression_levels=[0.2, 0.5, 0.8],
            algorithm_id="test",
        )
        assert len(bids) == 3
        freed = [b.tokens_freed for b in bids]
        assert freed[0] <= freed[1] <= freed[2]

    def test_confidence_fn(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=[1, 2, 3, 4, 5],
            scores=[0.2, 0.4, 0.6, 0.8, 1.0],
            compression_levels=[0.4],
            algorithm_id="test",
            confidence_fn=lambda: 0.85,
        )
        assert bids[0].confidence == 0.85

    def test_extra_metadata(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=[1, 2, 3, 4, 5],
            scores=[0.1, 0.2, 0.3, 0.4, 0.5],
            compression_levels=[0.4],
            algorithm_id="test",
            extra_metadata={"custom_key": "custom_value"},
        )
        assert bids[0].metadata["custom_key"] == "custom_value"
        assert "compression_level" in bids[0].metadata

    def test_default_confidence(self) -> None:
        from bidkv.scoring.bid_builder import build_bids

        bids = build_bids(
            request_id="req-bb",
            token_ids=[1, 2, 3, 4, 5],
            scores=[0.5] * 5,
            compression_levels=[0.4],
            algorithm_id="test",
        )
        assert bids[0].confidence == 0.5


# ===========================================================================
# BidKVStrategy scorer-agnostic 测试
# ===========================================================================


class TestBidKVStrategyScorerAgnostic:
    """测试 BidKVStrategy 接受任意 ScoringStrategy 实现。"""

    def test_default_is_h2o(self) -> None:
        from bidkv.baselines.bidkv_strategy import BidKVStrategy

        strategy = BidKVStrategy()
        assert isinstance(strategy.scoring, PositionalScoring)
